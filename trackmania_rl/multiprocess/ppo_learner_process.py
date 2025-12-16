"""
trackmania_rl/multiprocess/ppo_learner_process.py

PPO Learner process - fully integrated with Trackmania's training infrastructure.
"""

import copy
import importlib
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from multiprocessing.connection import wait
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.optim as optim
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from config_files import config_copy
from trackmania_rl import utilities
from trackmania_rl.agents.ppo import make_untrained_ppo_network
from trackmania_rl.map_reference_times import reference_times


def ppo_learner_process_fn(
    rollout_queues,
    uncompiled_shared_network,
    shared_network_lock,
    shared_steps: mp.Value,
    base_dir: Path,
    save_dir: Path,
    tensorboard_base_dir: Path,
):
    """
    PPO Learner Process - performs network updates using PPO algorithm.
    Fully integrated with Trackmania's training infrastructure.
    """
    layout_version = "lay_ppo"
    SummaryWriter(log_dir=str(tensorboard_base_dir / layout_version)).add_custom_scalars(
        {
            layout_version: {
                "eval_race_time_robust": [
                    "Multiline",
                    [
                        "eval_race_time_robust",
                    ],
                ],
                "explo_race_time_finished": [
                    "Multiline",
                    [
                        "explo_race_time_finished",
                    ],
                ],
                "ppo_losses": [
                    "Multiline",
                    ["ppo/policy_loss", "ppo/value_loss"],
                ],
                "ppo_metrics": [
                    "Multiline",
                    ["ppo/entropy", "ppo/approx_kl", "ppo/clipfrac"],
                ],
                "single_zone_reached": [
                    "Multiline",
                    [
                        "single_zone_reached",
                    ],
                ],
            },
        }
    )

    # ========================================================
    # Create PPO network
    # ========================================================
    online_network, uncompiled_online_network = make_untrained_ppo_network(
        input_channels=config_copy.ppo_input_channels,
        num_actions=config_copy.ppo_num_actions,
        float_input_dim=config_copy.ppo_float_input_dim,
        img_height=config_copy.H_downsized,
        img_width=config_copy.W_downsized,
        jit=config_copy.use_jit,
        is_inference=False
    )

    print(online_network)
    utilities.count_parameters(online_network)

    accumulated_stats: defaultdict = defaultdict(int)
    accumulated_stats["alltime_min_ms"] = {}
    accumulated_stats["rolling_mean_ms"] = {}
    previous_alltime_min = None
    time_last_save = time.perf_counter()
    queue_check_order = list(range(len(rollout_queues)))
    rollout_queue_readers = [q._reader for q in rollout_queues]
    time_waited_for_workers_since_last_tensorboard_write = 0
    time_training_since_last_tensorboard_write = 0

    # ========================================================
    # Load existing weights
    # ========================================================
    try:
        online_network.load_state_dict(torch.load(f=save_dir / "weights1.torch", weights_only=False))
        print(" =====================  PPO Learner weights loaded !  ============================")
    except:
        print(" PPO Learner could not load weights")

    with shared_network_lock:
        uncompiled_shared_network.load_state_dict(uncompiled_online_network.state_dict())

    try:
        accumulated_stats = joblib.load(save_dir / "accumulated_stats.joblib")
        shared_steps.value = accumulated_stats["cumul_number_frames_played"]
        print(" =====================   PPO Learner stats loaded !   ============================")
    except:
        print(" PPO Learner could not load stats")

    if "rolling_mean_ms" not in accumulated_stats.keys():
        accumulated_stats["rolling_mean_ms"] = {}

    accumulated_stats["cumul_number_ppo_updates"] = accumulated_stats.get("cumul_number_ppo_updates", 0)
    accumulated_stats["cumul_number_samples_trained"] = accumulated_stats.get("cumul_number_samples_trained", 0)

    # Create optimizer
    optimizer = optim.Adam(
        online_network.parameters(),
        lr=config_copy.ppo_learning_rate,
        eps=1e-5
    )

    try:
        optimizer.load_state_dict(torch.load(f=save_dir / "optimizer1.torch", weights_only=False))
        print(" =========================  PPO Optimizer loaded !  ================================")
    except:
        print(" Could not load PPO optimizer")

    tensorboard_suffix = utilities.from_staircase_schedule(
        config_copy.tensorboard_suffix_schedule,
        accumulated_stats["cumul_number_frames_played"],
    )
    tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_base_dir / (config_copy.run_name + tensorboard_suffix)))

    # PPO-specific tracking
    policy_loss_history = []
    value_loss_history = []
    entropy_history = []
    kl_history = []
    clipfrac_history = []
    update_duration_history = []

    print("[PPO Learner] Waiting for rollouts...")

    while True:  # Trainer loop
        before_wait_time = time.perf_counter()
        wait(rollout_queue_readers)
        time_waited = time.perf_counter() - before_wait_time
        if time_waited > 1:
            print(f"Warning: PPO learner waited {time_waited:.2f} seconds for workers to provide rollouts")
        time_waited_for_workers_since_last_tensorboard_write += time_waited
        
        for idx in queue_check_order:
            if not rollout_queues[idx].empty():
                (
                    buffer_data,       # PPO rollout data with advantages/returns
                    rollout_results,   # Original rollout results
                    end_race_stats,    # End race statistics
                    fill_buffer,       # Whether to use this rollout
                    is_explo,          # Is exploration
                    map_name,          # Map name
                    map_status,        # Map status (trained/blind)
                    rollout_duration,  # Rollout duration
                    loop_number,       # Loop number
                ) = rollout_queues[idx].get()
                queue_check_order.append(queue_check_order.pop(queue_check_order.index(idx)))
                break

        importlib.reload(config_copy)

        new_tensorboard_suffix = utilities.from_staircase_schedule(
            config_copy.tensorboard_suffix_schedule,
            accumulated_stats["cumul_number_frames_played"],
        )
        if new_tensorboard_suffix != tensorboard_suffix:
            tensorboard_suffix = new_tensorboard_suffix
            tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_base_dir / (config_copy.run_name + tensorboard_suffix)))

        accumulated_stats["cumul_number_frames_played"] += len(rollout_results["frames"])

        # ===============================================
        #   WRITE SINGLE RACE RESULTS TO TENSORBOARD
        # ===============================================
        race_stats_to_write = {
            f"race_time_ratio_{map_name}": end_race_stats["race_time_for_ratio"] / (rollout_duration * 1000),
            f"explo_race_time_{map_status}_{map_name}" if is_explo else f"eval_race_time_{map_status}_{map_name}": end_race_stats["race_time"] / 1000,
            f"explo_race_finished_{map_status}_{map_name}" if is_explo else f"eval_race_finished_{map_status}_{map_name}": end_race_stats["race_finished"],
            f"single_zone_reached_{map_status}_{map_name}": rollout_results["furthest_zone_idx"],
            "instrumentation__answer_normal_step": end_race_stats["instrumentation__answer_normal_step"],
            "instrumentation__answer_action_step": end_race_stats["instrumentation__answer_action_step"],
            "instrumentation__between_run_steps": end_race_stats["instrumentation__between_run_steps"],
            "instrumentation__grab_frame": end_race_stats["instrumentation__grab_frame"],
            "instrumentation__convert_frame": end_race_stats["instrumentation__convert_frame"],
            "instrumentation__grab_floats": end_race_stats["instrumentation__grab_floats"],
            "instrumentation__exploration_policy": end_race_stats["instrumentation__exploration_policy"],
            "instrumentation__request_inputs_and_speed": end_race_stats["instrumentation__request_inputs_and_speed"],
            "tmi_protection_cutoff": end_race_stats["tmi_protection_cutoff"],
            "worker_time_in_rollout_percentage": rollout_results["worker_time_in_rollout_percentage"],
        }
        print("Race time ratio  ", race_stats_to_write[f"race_time_ratio_{map_name}"])

        if end_race_stats["race_finished"]:
            race_stats_to_write[f"{'explo' if is_explo else 'eval'}_race_time_finished_{map_status}_{map_name}"] = (
                end_race_stats["race_time"] / 1000
            )
            if not is_explo:
                accumulated_stats["rolling_mean_ms"][map_name] = (
                    accumulated_stats["rolling_mean_ms"].get(map_name, config_copy.cutoff_rollout_if_race_not_finished_within_duration_ms)
                    * 0.9
                    + end_race_stats["race_time"] * 0.1
                )
        
        if (
            (not is_explo)
            and end_race_stats["race_finished"]
            and end_race_stats["race_time"] < 1.02 * accumulated_stats["rolling_mean_ms"].get(map_name, float('inf'))
        ):
            race_stats_to_write[f"eval_race_time_robust_{map_status}_{map_name}"] = end_race_stats["race_time"] / 1000
            if map_name in reference_times:
                for reference_time_name in ["author", "gold"]:
                    if reference_time_name in reference_times[map_name]:
                        reference_time = reference_times[map_name][reference_time_name]
                        race_stats_to_write[f"eval_ratio_{map_status}_{reference_time_name}_{map_name}"] = (
                            100 * (end_race_stats["race_time"] / 1000) / reference_time
                        )
                        race_stats_to_write[f"eval_agg_ratio_{map_status}_{reference_time_name}"] = (
                            100 * (end_race_stats["race_time"] / 1000) / reference_time
                        )

        for i in [0]:
            race_stats_to_write[f"q_value_{i}_starting_frame_{map_name}"] = end_race_stats[f"q_value_{i}_starting_frame"]
        
        if not is_explo:
            for i, split_time in enumerate(
                [
                    (e - s) / 1000
                    for s, e in zip(
                        end_race_stats["cp_time_ms"][:-1],
                        end_race_stats["cp_time_ms"][1:],
                    )
                ]
            ):
                race_stats_to_write[f"split_{map_name}_{i}"] = split_time

        walltime_tb = time.time()
        for tag, value in race_stats_to_write.items():
            tensorboard_writer.add_scalar(
                tag=tag,
                scalar_value=value,
                global_step=accumulated_stats["cumul_number_frames_played"],
                walltime=walltime_tb,
            )

        # ===============================================
        #   SAVE STUFF IF THIS WAS A GOOD RACE
        # ===============================================
        if end_race_stats["race_time"] < accumulated_stats["alltime_min_ms"].get(map_name, 99999999999):
            accumulated_stats["alltime_min_ms"][map_name] = end_race_stats["race_time"]
            if accumulated_stats["cumul_number_frames_played"] > config_copy.frames_before_save_best_runs:
                name = f"{map_name}_{end_race_stats['race_time']}"
                utilities.save_run(
                    base_dir,
                    save_dir / "best_runs" / name,
                    rollout_results,
                    f"{name}.inputs",
                    inputs_only=False,
                )
                utilities.save_checkpoint(
                    save_dir / "best_runs",
                    online_network,
                    None,  # No target network in PPO
                    optimizer,
                    None,  # No scaler in PPO
                )

        if end_race_stats["race_time"] < config_copy.threshold_to_save_all_runs_ms:
            name = f"{map_name}_{end_race_stats['race_time']}_{datetime.now().strftime('%m%d_%H%M%S')}_{accumulated_stats['cumul_number_frames_played']}_{'explo' if is_explo else 'eval'}"
            utilities.save_run(
                base_dir,
                save_dir / "good_runs",
                rollout_results,
                f"{name}.inputs",
                inputs_only=True,
            )

        # ===============================================
        #   PERFORM PPO UPDATE
        # ===============================================
        if fill_buffer and buffer_data is not None:
            if not online_network.training:
                online_network.train()

            update_start_time = time.perf_counter()
            
            # Perform PPO update
            metrics = ppo_update(
                online_network,
                optimizer,
                buffer_data,
                config_copy
            )
            
            update_duration = time.perf_counter() - update_start_time
            update_duration_history.append(update_duration)
            time_training_since_last_tensorboard_write += update_duration
            
            # Track metrics
            policy_loss_history.append(metrics['policy_loss'])
            value_loss_history.append(metrics['value_loss'])
            entropy_history.append(metrics['entropy'])
            kl_history.append(metrics['approx_kl'])
            clipfrac_history.append(metrics['clipfrac'])
            
            accumulated_stats["cumul_number_ppo_updates"] += 1
            accumulated_stats["cumul_number_samples_trained"] += buffer_data['observations'].shape[0] * config_copy.ppo_epochs
            
            print(f"PPO Update | Policy Loss: {metrics['policy_loss']:.4f} | "
                  f"Value Loss: {metrics['value_loss']:.4f} | "
                  f"Entropy: {metrics['entropy']:.4f} | "
                  f"KL: {metrics['approx_kl']:.6f} | "
                  f"Duration: {update_duration*1000:.1f}ms")

            # Update shared network periodically
            if accumulated_stats["cumul_number_ppo_updates"] % config_copy.ppo_shared_network_update_freq == 0:
                with shared_network_lock:
                    uncompiled_shared_network.load_state_dict(uncompiled_online_network.state_dict())

        sys.stdout.flush()

        # ===============================================
        #   WRITE AGGREGATED STATISTICS TO TENSORBOARD
        # ===============================================
        save_frequency_s = 5 * 60
        if time.perf_counter() - time_last_save >= save_frequency_s:
            accumulated_stats["cumul_training_hours"] = accumulated_stats.get("cumul_training_hours", 0) + (time.perf_counter() - time_last_save) / 3600
            time_since_last_save = time.perf_counter() - time_last_save
            waited_percentage = time_waited_for_workers_since_last_tensorboard_write / time_since_last_save
            trained_percentage = time_training_since_last_tensorboard_write / time_since_last_save
            time_waited_for_workers_since_last_tensorboard_write = 0
            time_training_since_last_tensorboard_write = 0
            time_last_save = time.perf_counter()

            # ===============================================
            #   COLLECT VARIOUS STATISTICS
            # ===============================================
            step_stats = {
                "ppo_learning_rate": config_copy.ppo_learning_rate,
                "ppo_gamma": config_copy.ppo_gamma,
                "ppo_gae_lambda": config_copy.ppo_gae_lambda,
                "ppo_clip_coef": config_copy.ppo_clip_coef,
                "ppo_ent_coef": config_copy.ppo_ent_coef,
                "ppo_vf_coef": config_copy.ppo_vf_coef,
                "learner_percentage_waiting_for_workers": waited_percentage,
                "learner_percentage_training": trained_percentage,
            }
            
            if len(policy_loss_history) > 0:
                step_stats.update({
                    "ppo/policy_loss": np.mean(policy_loss_history),
                    "ppo/value_loss": np.mean(value_loss_history),
                    "ppo/entropy": np.mean(entropy_history),
                    "ppo/approx_kl": np.mean(kl_history),
                    "ppo/clipfrac": np.mean(clipfrac_history),
                    "ppo/update_duration": np.median(update_duration_history),
                })

            for key, value in accumulated_stats.items():
                if key not in ["alltime_min_ms", "rolling_mean_ms"]:
                    step_stats[key] = value
            
            for key, value in accumulated_stats["alltime_min_ms"].items():
                step_stats[f"alltime_min_ms_{key}"] = value

            # Clear histories
            policy_loss_history = []
            value_loss_history = []
            entropy_history = []
            kl_history = []
            clipfrac_history = []
            update_duration_history = []

            # ===============================================
            #   WRITE TO TENSORBOARD
            # ===============================================
            walltime_tb = time.time()
            for name, param in online_network.named_parameters():
                tensorboard_writer.add_scalar(
                    tag=f"layer_{name}_L2",
                    scalar_value=np.sqrt((param**2).mean().detach().cpu().item()),
                    global_step=accumulated_stats["cumul_number_frames_played"],
                    walltime=walltime_tb,
                )

            for k, v in step_stats.items():
                tensorboard_writer.add_scalar(
                    tag=k,
                    scalar_value=v,
                    global_step=accumulated_stats["cumul_number_frames_played"],
                    walltime=walltime_tb,
                )

            previous_alltime_min = previous_alltime_min or copy.deepcopy(accumulated_stats["alltime_min_ms"])

            tensorboard_writer.add_text(
                "times_summary",
                f"{datetime.now().strftime('%Y/%m/%d, %H:%M:%S')} "
                + " ".join(
                    [
                        f"{'**' if v < previous_alltime_min.get(k, 99999999) else ''}{k}: {v / 1000:.2f}{'**' if v < previous_alltime_min.get(k, 99999999) else ''}"
                        for k, v in accumulated_stats["alltime_min_ms"].items()
                    ]
                ),
                global_step=accumulated_stats["cumul_number_frames_played"],
                walltime=walltime_tb,
            )

            previous_alltime_min = copy.deepcopy(accumulated_stats["alltime_min_ms"])

            # ===============================================
            #   SAVE CHECKPOINT
            # ===============================================
            torch.save(online_network.state_dict(), save_dir / "weights1.torch")
            torch.save(optimizer.state_dict(), save_dir / "optimizer1.torch")
            joblib.dump(accumulated_stats, save_dir / "accumulated_stats.joblib")
            print(f"[PPO Learner] Checkpoint saved at {accumulated_stats['cumul_number_frames_played']} frames")


def ppo_update(network, optimizer, buffer_data, config):
    """
    Perform PPO update on the network.
    
    Args:
        network: PPO network
        optimizer: Optimizer
        buffer_data: Dictionary containing rollout data (NUMPY arrays)
        config: Configuration object
    
    Returns:
        Dictionary of training metrics
    """
    # Move data to GPU and convert to tensors
    device = next(network.parameters()).device
    
    observations = torch.from_numpy(buffer_data['observations']).float().to(device)
    float_inputs = torch.from_numpy(buffer_data['float_inputs']).float().to(device) if config.ppo_float_input_dim > 0 else None
    actions = torch.from_numpy(buffer_data['actions']).long().to(device)
    old_log_probs = torch.from_numpy(buffer_data['log_probs']).float().to(device)
    advantages = torch.from_numpy(buffer_data['advantages']).float().to(device)
    returns = torch.from_numpy(buffer_data['returns']).float().to(device)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Get batch size
    batch_size = observations.shape[0]
    indices = np.arange(batch_size)
    
    # Training metrics
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    total_clipfrac = 0
    total_kl = 0
    num_updates = 0
    
    # Multiple epochs of updates
    for epoch in range(config.ppo_epochs):
        # Shuffle data
        np.random.shuffle(indices)
        
        # Mini-batch updates
        for start in range(0, batch_size, config.ppo_batch_size):
            end = start + config.ppo_batch_size
            if end > batch_size:
                continue
            
            mb_indices = indices[start:end]
            
            # Get mini-batch
            mb_obs = observations[mb_indices]
            mb_float = float_inputs[mb_indices] if float_inputs is not None else None
            mb_actions = actions[mb_indices]
            mb_old_log_probs = old_log_probs[mb_indices]
            mb_advantages = advantages[mb_indices]
            mb_returns = returns[mb_indices]
            
            # Ensure observations have correct shape (B, C, H, W)
            if mb_obs.ndim == 3:  # (B, H, W)
                mb_obs = mb_obs.unsqueeze(1)  # Add channel dim -> (B, 1, H, W)
            
            # Forward pass
            action_logits, values = network(mb_obs, mb_float)
            values = values.squeeze(-1)
            
            # Compute policy loss
            dist = torch.distributions.Categorical(logits=action_logits)
            log_probs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()
            
            # Compute ratio and clipped objective
            ratio = torch.exp(log_probs - mb_old_log_probs)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - config.ppo_clip_coef, 1.0 + config.ppo_clip_coef) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Compute value loss
            if config.ppo_clip_vloss:
                old_values = torch.from_numpy(buffer_data['values'][mb_indices]).float().to(device)
                values_clipped = old_values + torch.clamp(
                    values - old_values,
                    -config.ppo_clip_coef,
                    config.ppo_clip_coef
                )
                value_loss_unclipped = (values - mb_returns) ** 2
                value_loss_clipped = (values_clipped - mb_returns) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
            else:
                value_loss = 0.5 * ((values - mb_returns) ** 2).mean()
            
            # Total loss
            loss = policy_loss + config.ppo_vf_coef * value_loss - config.ppo_ent_coef * entropy
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), config.ppo_max_grad_norm)
            optimizer.step()
            
            # Metrics
            with torch.no_grad():
                clipfrac = ((ratio - 1.0).abs() > config.ppo_clip_coef).float().mean()
                approx_kl = (mb_old_log_probs - log_probs).mean()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            total_clipfrac += clipfrac.item()
            total_kl += approx_kl.item()
            num_updates += 1
    
    # Return average metrics
    return {
        'policy_loss': total_policy_loss / num_updates,
        'value_loss': total_value_loss / num_updates,
        'entropy': total_entropy / num_updates,
        'clipfrac': total_clipfrac / num_updates,
        'approx_kl': total_kl / num_updates,
    }
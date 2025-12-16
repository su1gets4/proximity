"""
trackmania_rl/multiprocess/ppo_collector_process.py

PPO Collector - Fixed to match IQN's logic flow.
"""

import importlib
import time
from itertools import chain, count, cycle
from pathlib import Path

import numpy as np
import torch
from torch import multiprocessing as mp

from config_files import config_copy
from trackmania_rl import utilities
from trackmania_rl.utilities import set_random_seed


def ppo_collector_process_fn(
    rollout_queue,
    uncompiled_shared_network,
    shared_network_lock,
    game_spawning_lock,
    shared_steps: mp.Value,
    base_dir: Path,
    save_dir: Path,
    tmi_port: int,
    process_number: int,
):
    """
    PPO Collector Process for Trackmania.
    Matches IQN's logic flow: collect rollout, THEN process it.
    """
    from trackmania_rl.map_loader import analyze_map_cycle, load_next_map_zone_centers
    from trackmania_rl.tmi_interaction import game_instance_manager
    from trackmania_rl.agents.ppo import make_untrained_ppo_network

    set_random_seed(process_number)

    # Initialize TMInterface
    tmi = game_instance_manager.GameInstanceManager(
        game_spawning_lock=game_spawning_lock,
        running_speed=config_copy.running_speed,
        run_steps_per_action=config_copy.tm_engine_step_per_action,
        max_overall_duration_ms=config_copy.cutoff_rollout_if_race_not_finished_within_duration_ms,
        max_minirace_duration_ms=config_copy.cutoff_rollout_if_no_vcp_passed_within_duration_ms,
        tmi_port=tmi_port,
    )

    # Create PPO network
    _, inference_network = make_untrained_ppo_network(
        input_channels=config_copy.ppo_input_channels,
        num_actions=config_copy.ppo_num_actions,
        float_input_dim=config_copy.ppo_float_input_dim,
        img_height=config_copy.H_downsized,
        img_width=config_copy.W_downsized,
        jit=config_copy.use_jit,
        is_inference=True,
    )

    try:
        checkpoint = torch.load(save_dir / "weights1.torch", weights_only=False, map_location="cpu")
        inference_network.load_state_dict(checkpoint)
        print(f"[PPO Collector {process_number}] Loaded weights")
    except Exception as e:
        print(f"[PPO Collector {process_number}] Starting from scratch:", e)

    device = torch.device(f"cuda:{process_number % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")
    inference_network.to(device)
    inference_network.train()  # PPO uses train mode for stochastic sampling

    def update_network():
        """Update inference network from shared network"""
        with shared_network_lock:
            inference_network.load_state_dict(uncompiled_shared_network.state_dict())

    # ========================================
    # PPO EXPLORATION POLICY
    # ========================================
    class PPOInferer:
        """Wrapper class to match IQN's Inferer interface"""
        
        def __init__(self, network, device):
            self.network = network
            self.device = device
            self.is_explo = True
        
        def get_exploration_action(self, obs, float_input):
            """
            Get action from PPO policy.
            Returns: (action_idx, is_greedy, value, action_probs)
            """
            # Prepare observation
            if obs.ndim == 2:
                obs = obs[np.newaxis, :, :]
            
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
            
            float_tensor = None
            if config_copy.ppo_float_input_dim > 0:
                float_tensor = torch.from_numpy(float_input).unsqueeze(0).float().to(self.device)
            
            # Get action from network
            with torch.no_grad():
                action, log_prob, entropy, value = self.network.get_action_and_value(obs_tensor, float_tensor)
                
                # Get action probabilities
                action_logits, _ = self.network(obs_tensor, float_tensor)
                action_probs = torch.softmax(action_logits, dim=-1).squeeze().cpu().numpy()
            
            action_np = action.cpu().item()
            value_np = value.cpu().item()
            
            # Return tuple matching IQN: (action, is_greedy, value, probs)
            # PPO samples from policy, so we consider it "greedy" w.r.t. the current policy
            return (action_np, True, value_np, action_probs)
    
    inferer = PPOInferer(inference_network, device)

    # ========================================
    # MAP CYCLE SETUP
    # ========================================
    map_cycle_str = str(config_copy.map_cycle)
    set_maps_trained, set_maps_blind = analyze_map_cycle(config_copy.map_cycle)
    map_cycle_iter = cycle(chain(*config_copy.map_cycle))
    zone_centers_filename = None

    # ========================================
    # WARMUP
    # ========================================
    for _ in range(5):
        dummy_obs = torch.randint(
            0, 255,
            (1, config_copy.ppo_input_channels, config_copy.H_downsized, config_copy.W_downsized),
            dtype=torch.uint8
        ).float().to(device)
        
        dummy_float = (
            torch.randn(1, config_copy.ppo_float_input_dim).to(device)
            if config_copy.ppo_float_input_dim > 0
            else None
        )
        
        with torch.no_grad():
            _ = inference_network.get_action_and_value(dummy_obs, dummy_float)

    time_since_last_queue_push = time.perf_counter()

    # ========================================
    # MAIN TRAINING LOOP
    # ========================================
    for loop_number in count(1):
        importlib.reload(config_copy)
        
        tmi.max_minirace_duration_ms = config_copy.cutoff_rollout_if_no_vcp_passed_within_duration_ms

        # Check if map cycle changed
        if str(config_copy.map_cycle) != map_cycle_str:
            map_cycle_str = str(config_copy.map_cycle)
            set_maps_trained, set_maps_blind = analyze_map_cycle(config_copy.map_cycle)
            map_cycle_iter = cycle(chain(*config_copy.map_cycle))

        # Get next map
        next_map_tuple = next(map_cycle_iter)
        if next_map_tuple[2] != zone_centers_filename:
            zone_centers = load_next_map_zone_centers(next_map_tuple[2], base_dir)
        
        map_name, map_path, zone_centers_filename, is_explo, fill_buffer = next_map_tuple
        map_status = "trained" if map_name in set_maps_trained else "blind"
        
        inferer.is_explo = is_explo

        # Update network before rollout
        update_network()

        # ========================================
        # COLLECT ROLLOUT
        # ========================================
        rollout_start_time = time.perf_counter()
        
        rollout_results, end_race_stats = tmi.rollout(
            exploration_policy=inferer.get_exploration_action,
            map_path=map_path,
            zone_centers=zone_centers,
            update_network=update_network,
        )
        
        rollout_end_time = time.perf_counter()
        rollout_duration = rollout_end_time - rollout_start_time
        
        print(f"[PPO Collector {process_number}] Rollout complete: {len(rollout_results['frames'])} steps", flush=True)

        # ========================================
        # PROCESS ROLLOUT FOR PPO
        # ========================================
        if not tmi.last_rollout_crashed and fill_buffer and len(rollout_results['frames']) > 0:
            # Extract data from rollout_results
            observations = np.array([frame for frame in rollout_results['frames']], dtype=np.uint8)
            float_inputs = np.array([state for state in rollout_results['state_float']], dtype=np.float32)
            actions = np.array(rollout_results['actions'], dtype=np.int64)
            rewards = np.array(rollout_results['rewards'], dtype=np.float32)
            
            # Get log_probs and values by re-running network (since we didn't store them during rollout)
            log_probs = []
            values = []
            
            with torch.no_grad():
                for i in range(len(observations)):
                    obs = observations[i]
                    if obs.ndim == 2:
                        obs = obs[np.newaxis, :, :]
                    
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(device)
                    float_tensor = None
                    if config_copy.ppo_float_input_dim > 0:
                        float_tensor = torch.from_numpy(float_inputs[i]).unsqueeze(0).float().to(device)
                    
                    action_logits, value = inference_network(obs_tensor, float_tensor)
                    dist = torch.distributions.Categorical(logits=action_logits)
                    log_prob = dist.log_prob(torch.tensor([actions[i]]).to(device))
                    
                    log_probs.append(log_prob.cpu().item())
                    values.append(value.squeeze().cpu().item())
            
            log_probs = np.array(log_probs, dtype=np.float32)
            values = np.array(values, dtype=np.float32)
            
            # Compute last value for bootstrapping
            last_obs = observations[-1]
            if last_obs.ndim == 2:
                last_obs = last_obs[np.newaxis, :, :]
            last_obs_tensor = torch.from_numpy(last_obs).unsqueeze(0).float().to(device)
            last_float_tensor = None
            if config_copy.ppo_float_input_dim > 0:
                last_float_tensor = torch.from_numpy(float_inputs[-1]).unsqueeze(0).float().to(device)
            
            with torch.no_grad():
                last_value = inference_network.get_value(last_obs_tensor, last_float_tensor).cpu().item()
            
            # Compute returns and advantages using GAE
            advantages = np.zeros(len(rewards), dtype=np.float32)
            last_gae_lam = 0.0
            dones = np.zeros(len(rewards), dtype=np.float32)  # No dones mid-episode
            dones[-1] = 1.0 if end_race_stats['race_finished'] else 0.0
            
            for t in reversed(range(len(rewards))):
                next_non_terminal = 1.0 - dones[t]
                next_value = last_value if t == len(rewards) - 1 else values[t + 1]
                
                delta = rewards[t] + config_copy.ppo_gamma * next_value * next_non_terminal - values[t]
                advantages[t] = last_gae_lam = delta + config_copy.ppo_gamma * config_copy.ppo_gae_lambda * next_non_terminal * last_gae_lam
            
            returns = advantages + values
            
            # Create buffer data
            buffer_data = {
                'observations': observations,
                'float_inputs': float_inputs,
                'actions': actions,
                'log_probs': log_probs,
                'advantages': advantages,
                'returns': returns,
                'values': values,
            }
            
            # Update shared step counter
            with shared_steps.get_lock():
                shared_steps.value += len(observations)
            
            # Send to learner
            rollout_results["worker_time_in_rollout_percentage"] = rollout_duration / (time.perf_counter() - time_since_last_queue_push)
            time_since_last_queue_push = time.perf_counter()
            
            try:
                rollout_queue.put(
                    (
                        buffer_data,
                        rollout_results,
                        end_race_stats,
                        fill_buffer,
                        is_explo,
                        map_name,
                        map_status,
                        rollout_duration,
                        loop_number,
                    ),
                    timeout=5
                )
            except:
                print(f"[PPO Collector {process_number}] Warning: Queue full, dropping rollout")
        
        print("", flush=True)
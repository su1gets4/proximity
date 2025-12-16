"""
trackmania_rl/agents/ppo.py

PPO Actor-Critic Network for Trackmania
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PPONetwork(nn.Module):
    """
    Combined Actor-Critic network for PPO.
    Actor outputs action probabilities, Critic outputs state value.
    Handles both image observations and float inputs.
    """
    
    def __init__(self, input_channels=3, num_actions=12, float_input_dim=0, img_height=120, img_width=160):
        super(PPONetwork, self).__init__()
        
        self.float_input_dim = float_input_dim
        
        # Shared convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size after convolutions dynamically
        # After conv1: (H-8)/4+1, (W-8)/4+1
        # After conv2: (H-4)/2+1, (W-4)/2+1
        # After conv3: (H-3)/1+1, (W-3)/1+1
        def conv_output_shape(h, w):
            h = (h - 8) // 4 + 1  # conv1
            w = (w - 8) // 4 + 1
            h = (h - 4) // 2 + 1  # conv2
            w = (w - 4) // 2 + 1
            h = (h - 3) // 1 + 1  # conv3
            w = (w - 3) // 1 + 1
            return h, w
        
        conv_h, conv_w = conv_output_shape(img_height, img_width)
        conv_output_size = 64 * conv_h * conv_w
        
        # Shared fully connected layer (combines CNN features + float inputs)
        self.fc_shared = nn.Linear(conv_output_size + float_input_dim, 512)
        
        # Actor head (policy)
        self.fc_actor = nn.Linear(512, num_actions)
        
        # Critic head (value function)
        self.fc_critic = nn.Linear(512, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Special initialization for actor output layer
        nn.init.orthogonal_(self.fc_actor.weight, gain=0.01)
        nn.init.constant_(self.fc_actor.bias, 0)
    
    def forward(self, x, float_input=None):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (batch, channels, height, width)
            float_input: Additional float features (batch, float_input_dim)
        
        Returns:
            action_logits: Logits for action distribution
            value: State value estimate
        """
        # Normalize input if needed (assuming pixel values 0-255)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        
        # Shared feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        
        # Concatenate float inputs if provided
        if float_input is not None and self.float_input_dim > 0:
            x = torch.cat([x, float_input], dim=1)
        
        x = F.relu(self.fc_shared(x))
        
        # Actor and Critic outputs
        action_logits = self.fc_actor(x)
        value = self.fc_critic(x)
        
        return action_logits, value
    
    def get_action_and_value(self, x, float_input=None, action=None):
        """
        Get action, log probability, entropy, and value.
        Used during rollout collection.
        
        Args:
            x: Input observation
            float_input: Additional float features
            action: If provided, compute log prob for this action
        
        Returns:
            action: Selected action
            log_prob: Log probability of the action
            entropy: Entropy of the action distribution
            value: State value estimate
        """
        action_logits, value = self.forward(x, float_input)
        dist = Categorical(logits=action_logits)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value.squeeze(-1)
    
    def get_value(self, x, float_input=None):
        """Get only the value estimate (used in critic-only forward pass)"""
        _, value = self.forward(x, float_input)
        return value.squeeze(-1)


def make_untrained_ppo_network(input_channels=3, num_actions=12, float_input_dim=0, img_height=120, img_width=160, jit=False, is_inference=False):
    """
    Factory function to create PPO network, matching IQN interface.
    
    Args:
        input_channels: Number of input channels (e.g., 3 for RGB, 1 for grayscale)
        num_actions: Number of possible actions
        float_input_dim: Dimension of additional float features
        img_height: Height of input images
        img_width: Width of input images
        jit: Whether to use JIT compilation
        is_inference: Whether this is for inference only
    
    Returns:
        compiled_network: JIT compiled network (or None if jit=False)
        uncompiled_network: Raw PyTorch network
    """
    network = PPONetwork(
        input_channels=input_channels,
        num_actions=num_actions,
        float_input_dim=float_input_dim,
        img_height=img_height,
        img_width=img_width
    )
    
    if is_inference:
        network.eval()
    
    compiled_network = None
    if jit:
        # Create dummy input for JIT tracing
        dummy_input = torch.randn(1, input_channels, img_height, img_width)
        dummy_float = torch.randn(1, float_input_dim) if float_input_dim > 0 else None
        compiled_network = torch.jit.trace(network, (dummy_input, dummy_float))
    
    return compiled_network, network
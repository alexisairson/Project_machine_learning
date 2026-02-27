import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
import copy
import pickle
import os
from datetime import datetime


# ==========================================================
#       CONFIGURATION PARSERS
# ==========================================================

def _parse_value(value_str):
    """Helper function to parse a single value from string."""
    value_str = value_str.strip()
    try:
        if value_str.lower() == 'true':
            return True
        elif value_str.lower() == 'false':
            return False
        elif '.' in value_str:
            return float(value_str)
        else:
            return int(value_str)
    except ValueError:
        return value_str


def _load_config_dict(filepath):
    """Load configuration file into a dictionary."""
    config = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = _parse_value(value)
    except FileNotFoundError:
        pass
    return config


def parse_creature_config(filepath="Creature.txt"):
    """Parse creature configuration file."""
    config = _load_config_dict(filepath)
    
    defaults = {
        'segment_length': 3.0,
        'size_penalty_factor': 0.25,
        'fixed_segments': True,
        'num_segments': 2,
        'min_segments': 2,
        'max_segments': 10,
        'target_angle_deg': 0,
        'max_angle_deg': 180.0,
        'use_nn': True,
        'base_energy': 100.0,
        'energy_cost_angle_move': 0.03,
        'energy_cost_add_segment': 3.0,
        'energy_cost_remove_segment': 1.5,
        'metabolism_rate': 0.002
    }
    
    for key, default_val in defaults.items():
        if key not in config:
            config[key] = default_val
    
    return config


def parse_physics_config(filepath="PhysicsModel.txt"):
    """Parse physics model configuration file."""
    config = _load_config_dict(filepath)
    
    defaults = {
        'water_drag': 0.05,
        'thrust_coefficient': 3.5,
        'angle_speed': 0.4,
        'drag_spread_factor': 2.0
    }
    
    for key, default_val in defaults.items():
        if key not in config:
            config[key] = default_val
    
    return config


def parse_learning_config(filepath="LearningProcess.txt"):
    """Parse learning process configuration file."""
    config = _load_config_dict(filepath)
    
    defaults = {
        'learning_rate_nn': 0.001,
        'gamma': 0.95,
        'epsilon_min': 0.05
    }
    
    for key, default_val in defaults.items():
        if key not in config:
            config[key] = default_val
    
    return config


def parse_environment_config(filepath="Environment.txt"):
    """Parse environment configuration file."""
    config = _load_config_dict(filepath)
    
    defaults = {
        'min_segments': 2,
        'max_segments': 10,
        'duration_simulation': 30.0,
        'dt': 0.01,
        'substeps': 5
    }
    
    for key, default_val in defaults.items():
        if key not in config:
            config[key] = default_val
    
    return config


# ==========================================================
#       LEARNING PROCESS
# ==========================================================

class QNetwork(nn.Module):
    """Neural network for Q-value approximation."""
    
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)
    
    def get_q_values(self, state):
        with torch.no_grad():
            state_tensor = torch.from_numpy(state) if isinstance(state, np.ndarray) else state
            return self.network(state_tensor).numpy()


class QTabular:
    """Tabular Q-learning with state discretization."""
    
    def __init__(self, state_dim, action_dim, num_bins=8):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.q_table = {}
        self.learning_rate = 0.15
    
    def discretize_state(self, state):
        discrete = []
        for s in state:
            s_clipped = np.clip(s, -1, 1)
            bin_idx = int((s_clipped + 1) / 2 * (self.num_bins - 1))
            bin_idx = np.clip(bin_idx, 0, self.num_bins - 1)
            discrete.append(bin_idx)
        return tuple(discrete)
    
    def get_q_values(self, state):
        discrete_state = self.discretize_state(state)
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.action_dim)
        return self.q_table[discrete_state]
    
    def update(self, state, action, target):
        discrete_state = self.discretize_state(state)
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.action_dim)
        current = self.q_table[discrete_state][action]
        self.q_table[discrete_state][action] = current + self.learning_rate * (target - current)


class LearningProcess:
    """Manages Q-learning components."""
    
    def __init__(self, state_dim, action_dim, use_neural_network=True, 
                 learning_rate_nn=0.001, gamma=0.95, epsilon_min=0.05):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_neural_network = use_neural_network
        self.learning_rate_nn = learning_rate_nn
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        
        if use_neural_network:
            self.model = QNetwork(state_dim, action_dim)
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate_nn)
        else:
            self.model = QTabular(state_dim, action_dim)
            self.optimizer = None
    
    @classmethod
    def from_config_file(cls, state_dim, action_dim, use_neural_network=True, 
                         filepath="LearningProcess.txt"):
        config = parse_learning_config(filepath)
        
        return cls(
            state_dim=state_dim,
            action_dim=action_dim,
            use_neural_network=use_neural_network,
            learning_rate_nn=float(config['learning_rate_nn']),
            gamma=float(config['gamma']),
            epsilon_min=float(config['epsilon_min'])
        )
    
    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            if self.use_neural_network:
                with torch.no_grad():
                    state_tensor = torch.from_numpy(state)
                    q_values = self.model(state_tensor)
                    return torch.argmin(q_values).item()
            else:
                q_values = self.model.get_q_values(state)
                return int(np.argmin(q_values))
    
    def get_epsilon(self, episode, total_episodes):
        return max(self.epsilon_min, 1.0 - (episode / total_episodes) * (1.0 - self.epsilon_min))
    
    def calculate_reward(self, creature_state, direction_vector, start_pos, energy_cost):
        head_pos = creature_state['head_pos']
        head_vel = creature_state['head_vel']
        
        delta = head_pos - start_pos
        distance = np.dot(delta, direction_vector)
        
        speed = np.linalg.norm(head_vel)
        if speed > 1e-6:
            vel_dir = head_vel / speed
            alignment = np.dot(vel_dir, direction_vector)
        else:
            alignment = 0.0
        
        reward_speed = alignment * speed * 0.5
        reward_distance = distance * 0.05
        penalty_energy = energy_cost * 0.005
        reward = reward_speed + reward_distance - penalty_energy
        
        info = {
            'distance': distance,
            'speed': speed,
            'alignment': alignment
        }
        
        return reward, info
    
    def update_q_values(self, state, action, cost, next_state, done):
        if self.use_neural_network:
            with torch.no_grad():
                q_next = self.model(torch.from_numpy(next_state))
                target = cost + self.gamma * torch.min(q_next).item()
            
            current_q = self.model(torch.from_numpy(state))[action]
            loss = nn.MSELoss()(current_q, torch.tensor(target, dtype=torch.float32))
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            return loss.item()
        else:
            q_next = self.model.get_q_values(next_state)
            target = cost + self.gamma * np.min(q_next)
            self.model.update(state, action, target)
            return None
    
    def get_model_state(self):
        if self.use_neural_network:
            return self.model.state_dict()
        else:
            return self.model.q_table.copy()


# ==========================================================
#       PHYSICS MODEL
# ==========================================================

class PhysicsModel:
    
    """
    Aquatic propulsion physics.
    """
    
    def __init__(self, water_drag=0.05, thrust_coefficient=3.5, angle_speed=0.4,
                 drag_spread_factor=2.0):
        self.base_drag = water_drag
        self.thrust_coefficient = thrust_coefficient
        self.angle_speed = angle_speed
        self.drag_spread_factor = drag_spread_factor
    
    @classmethod
    def from_config_file(cls, filepath="PhysicsModel.txt"):
        config = parse_physics_config(filepath)
        
        return cls(
            water_drag=float(config['water_drag']),
            thrust_coefficient=float(config['thrust_coefficient']),
            angle_speed=float(config['angle_speed']),
            drag_spread_factor=float(config['drag_spread_factor'])
        )
    
    def update(self, creature, dt):
        old_angles = creature.segment_angles.copy()
        
        # 1. Update angles towards target
        for i in range(creature.num_segments):
            diff = creature.target_angles[i] - creature.segment_angles[i]
            creature.segment_angles[i] += diff * self.angle_speed
        
        # 2. Apply thrust from angle changes
        # The head_velocity vector is thus modified and amplified
        self._apply_thrust(creature, old_angles, dt)
        
        # 3. Apply drag
        # The head_velocity vector is thus reduced
        self._apply_drag(creature, dt)
        
        # 5. Update position of the head
        creature.head_pos = creature.head_pos + creature.head_vel * dt
    
    def _apply_thrust(self, creature, old_angles, dt):
        
        """
        Calculate thrust from segment angle changes.
        It propulses the creature in a certain direction
        
        When segment i rotates, it sweeps a circular sector,
        displacing water perpendicular to its orientation.
        """
        
        thrust = np.array([0.0, 0.0])
        
        for i in range(creature.num_segments):
            
            # Compute the angle of the movement
            angle_change = creature.segment_angles[i] - old_angles[i]
            
            if abs(angle_change) < 1e-8:
                continue
            
            # Swept area = (1/2) × L² × |dθ|
            swept_area = 0.5 * abs(angle_change) * (creature.segment_length)**2
            thrust_magnitude = self.thrust_coefficient * swept_area
            
            # Segment orientation in world frame
            # seg_world_angle is thus the angle between the origin x=0 and the new position of the segment CW
            seg_world_angle = creature.base_orientation + creature.segment_angles[i]
            
            # Thrust is perpendicular to segment
            if angle_change > 0:
                thrust_angle = seg_world_angle + math.pi / 2
            else:
                thrust_angle = seg_world_angle - math.pi / 2
            
            # thrust is thus a vector that will propulse the creature in a certain direction
            thrust_dir = np.array([math.cos(thrust_angle), math.sin(thrust_angle)])
            thrust += thrust_magnitude * thrust_dir
            
        # Apply total thrust
        creature.head_vel = creature.head_vel + thrust * dt
    
    def _apply_drag(self, creature, dt):
        
        """
        Calculate drag based on the new segment orientations.
        It slows down the progression of the creature in its movement progression.
        
        It computes the component of each segment perpendicular to the movement.
        """
        
        # The speed of progression is the norm of the velocity vector
        speed = np.linalg.norm(creature.head_vel)
        if speed < 0.001:
            return
        
        # This angle gives the direction of progression relative to x=0
        vel_angle = math.atan2(creature.head_vel[1], creature.head_vel[0])
        
        # Each segment slows down a bit the progression
        total_projected_area = 0.0
        for i in range(creature.num_segments):
            
            # Segment orientation in world frame
            # seg_world_angle is thus the angle between the origin x=0 and the new position of the segment CW
            seg_world_angle = creature.base_orientation + creature.segment_angles[i]
            
            # The perpendicular component of the segment is:
            angle_to_velocity     = seg_world_angle - vel_angle
            projected_area        = abs(math.sin(angle_to_velocity)) * creature.segment_length
            
            # We add a factor because the tail segments are less subjects to the drag
            total_projected_area += projected_area # * ((creature.num_segments - i) / creature.num_segments)
        
        # We normalize the total projection to apply a constant spread factor
        normalized_area = total_projected_area / creature.segment_length
        current_drag_coeff = self.base_drag * (1.0 + self.drag_spread_factor * normalized_area)
        
        # The drag is proportional to - coeff x speed^2 in the direction of the velocity
        # Thus, drag = - coeff x speed^2 x normalized(head_vel) = - coeff x speed x head_vel
        drag = -current_drag_coeff * creature.head_vel * speed
        creature.head_vel = creature.head_vel + drag * dt

# ==========================================================
#       CREATURE
# ==========================================================

class Creature:
    
    """
    Creature with segment angles.
    """
    
    def __init__(self,
                 segment_length=3.0, size_penalty_factor=0.25,
                 fixed_segments=True, num_segments=2,
                 min_segments=2, max_segments=10,
                 target_angle_deg=0, max_angle_deg=180.0,
                 use_nn=True,
                 base_energy=100.0, metabolism_rate=0.002,
                 energy_cost_angle_move=0.03,
                 energy_cost_add_segment=3.0, energy_cost_remove_segment=1.5):
        
        self.segment_length = segment_length
        self.size_penalty_factor = size_penalty_factor
        
        self.fixed_segments = fixed_segments
        self.num_segments = num_segments
        self.min_segments = min_segments
        self.max_segments = max_segments
        
        self.target_angle_deg = target_angle_deg
        self.target_angle_rad = math.radians(target_angle_deg)
        self.base_orientation = math.radians(target_angle_deg)
        
        self.max_angle = math.radians(max_angle_deg)
        self.use_nn = use_nn
        
        self.base_energy = base_energy
        self.max_energy = base_energy
        self.energy = base_energy
        
        self.energy_cost_angle_move = energy_cost_angle_move
        self.energy_cost_add_segment = energy_cost_add_segment
        self.energy_cost_remove_segment = energy_cost_remove_segment
        self.metabolism_rate = metabolism_rate
        
        if not fixed_segments:
            self.num_segments = min_segments
        
        # Segment angles: orientation of each segment relative to base_orientation
        # For N segments, there are N angles
        # Stored in array of size max_segments
        self.segment_angles = np.zeros(max_segments, dtype=np.float64)
        self.target_angles = np.zeros(max_segments, dtype=np.float64)
        
        self.head_pos = np.array([0.0, 0.0], dtype=np.float64)
        self.head_vel = np.array([0.0, 0.0], dtype=np.float64)
        
        self.direction_vector = np.array([
            math.cos(self.target_angle_rad),
            math.sin(self.target_angle_rad)
        ])
    
    @classmethod
    def from_config_file(cls, filepath="Creature.txt", 
                         min_segments_env=2, max_segments_env=10):
        config = parse_creature_config(filepath)
        
        return cls(
            segment_length=float(config['segment_length']),
            size_penalty_factor=float(config['size_penalty_factor']),
            fixed_segments=bool(config['fixed_segments']),
            num_segments=int(config['num_segments']),
            min_segments=int(max(min_segments_env, config['min_segments'])),
            max_segments=int(min(max_segments_env, config['max_segments'])),
            target_angle_deg=float(config['target_angle_deg']),
            max_angle_deg=float(config['max_angle_deg']),
            use_nn=bool(config['use_nn']),
            base_energy=float(config['base_energy']),
            energy_cost_angle_move=float(config['energy_cost_angle_move']),
            energy_cost_add_segment=float(config['energy_cost_add_segment']),
            energy_cost_remove_segment=float(config['energy_cost_remove_segment']),
            metabolism_rate=float(config['metabolism_rate'])
        )
    
    def reset(self, x=0.0, y=0.0, orientation=None):
        """
        Reset the creature for a new episode.
        """
        
        # Reset the orientation of the head
        self.base_orientation = orientation if orientation is not None else self.target_angle_rad
        
        # Straight body: all segment angles are 0 (aligned with base_orientation)
        self.segment_angles[:self.num_segments] = 0.0
        self.target_angles[:self.num_segments]  = 0.0
        
        # Replace the head to its initial position and set the velocity vector to 0
        self.head_pos = np.array([x, y], dtype=np.float64)
        self.head_vel = np.array([0.0, 0.0], dtype=np.float64)
        
        # Reset the energy level to the maximum level given the size of the creature
        self.max_energy = self.base_energy * self.get_body_size_factor()
        self.energy     = self.max_energy
    
    def get_body_size_factor(self):
        """Bigger creatures have a higher energy consumption."""
        return 1.0 + (self.num_segments - 2) * self.size_penalty_factor
    
    def consume_energy_metabolism(self):
        "Staying alive requires metabolism energy consumption."
        
        # Computes the cost of staying alive for a creature of this size
        size_factor = self.get_body_size_factor()
        energy_cost = self.metabolism_rate * size_factor
        
        # If the creature doesn't have enough energy left, it dies (energy to 0)
        self.energy = max(0, self.energy - energy_cost)
        return energy_cost
    
    def consume_energy_angle_move(self, angle_delta):
        """Changing the orientation of a segment requires energy."""
        
        # Computes the cost of modifying the orientation of a segment
        size_factor = self.get_body_size_factor()
        energy_cost = self.energy_cost_angle_move * abs(angle_delta) * size_factor
        
        # Ensures that the creature still has enough energy to make the move
        if self.energy - energy_cost < 0:
            return 0.0
        
        # If it can move, decreases the energy level
        self.energy -= energy_cost
        return energy_cost
    
    def add_segment(self):
        """Add a new member to the creature if possible."""
        
        # Ensures that we don't exceed the maximal number of members allowed
        if self.num_segments < self.max_segments:
            return False
        
        # Ensures that the creature still has enough energy to add a new member
        if self.energy - energy_cost < 0:
            return False
            
        # Computes the cost of adding a new member
        size_factor = self.get_body_size_factor()
        energy_cost = self.energy_cost_add_segment * size_factor
        
        # If it can add the new member,
        self.energy       -= energy_cost # Decreases the energy
        self.num_segments += 1           # Increases the number of members
        
        # As the maximal level of energy depends on the size of the creature,
        # we check that it doesn't overshoot it
        self.max_energy = self.base_energy * self.get_body_size_factor()
        self.energy = min(self.energy, self.max_energy)
        
        # Initialize the angle of this new segment angle to 0
        # It's thus aligned with the desired orientation
        self.segment_angles[self.num_segments - 1] = 0.0
        self.target_angles[self.num_segments - 1]  = 0.0
        
        return True
    
    def remove_segment(self):
        """Remove a member of the creature if possible."""
        
        # Ensures that we don't go below the minimal number of members allowed
        if self.num_segments > self.min_segments:
            return False
        
        # Ensures that the creature still has enough energy to remove a member
        if self.energy - energy_cost < 0:
            return False
        
        # Computes the cost of removing a member
        size_factor = self.get_body_size_factor()
        energy_cost = self.energy_cost_add_segment * size_factor
        
        # If it can be removed,
        self.energy       -= energy_cost # Decrease the energy level
        self.num_segments -= 1           # Decrease the number of members
        
        # As the maximal level of energy depends on the size of the creature,
        # we check that it doesn't overshoot it
        self.max_energy = self.base_energy * self.get_body_size_factor()
        self.energy = min(self.energy, self.max_energy)
        
        return True
    
    # def set_target_angle(self, segment_idx, angle):
    #     """Set target angle for a segment."""
    #     if 0 <= segment_idx < self.num_segments:
    #         self.target_angles[segment_idx] = np.clip(angle, -self.max_angle, self.max_angle)  
    
    # def get_state(self):
    #     """
    #     Get state vector: the segment angles.
        
    #     Returns angles normalized to [-1, 1] range.
    #     State size = num_segments
    #     """
    #     # Normalize angles to [-1, 1] for neural network
    #     normalized_angles = self.segment_angles[:self.num_segments] / self.max_angle
    #     return normalized_angles.astype(np.float32)

# ==========================================================
#       ENVIRONMENT
# ==========================================================

class Environment:
    """Environment orchestrator with dynamic state/action dimensions."""
    
    def __init__(self,
                 min_segments=2, max_segments=10,
                 duration_simulation=30.0, dt=0.01,
                 substeps=5):
        
        self.min_segments = min_segments
        self.max_segments = max_segments
        
        self.duration_simulation = duration_simulation
        self.dt = dt
        self.max_steps = int(duration_simulation / dt)
        self.substeps = substeps
        
        self.creature = Creature.from_config_file(
            "Creature.txt",
            min_segments_env=min_segments,
            max_segments_env=max_segments
        )
        
        self.physics_model = PhysicsModel.from_config_file("PhysicsModel.txt")
        
        self.learning_process = None
        
        self.step_count = 0
        self.start_pos = np.array([0.0, 0.0])
        self.trajectory = []
    
    @classmethod
    def from_config_file(cls, filepath="Environment.txt"):
        config = parse_environment_config(filepath)
        
        return cls(
            min_segments=int(config['min_segments']),
            max_segments=int(config['max_segments']),
            duration_simulation=float(config['duration_simulation']),
            dt=float(config['dt']),
            substeps=int(config['substeps'])
        )
    
    def _get_action_dim(self):
        """
        Action dimension:
        - For each segment: open, close, hold (3 actions)
        - If variable segments: add segment, remove segment (2 actions)
        """
        base_actions = self.creature.num_segments * 3
        if self.creature.fixed_segments:
            return base_actions
        return base_actions + 2
    
    def init_learning_process(self, use_nn):
        """Initialize learning process with current dimensions."""
        state_dim = self.creature.num_segments
        action_dim = self._get_action_dim()
        
        self.learning_process = LearningProcess.from_config_file(
            state_dim, action_dim, use_neural_network=use_nn
        )
    
    def reset(self):
        """Reset environment for new episode."""
        self.creature.reset(0.0, 0.0, self.creature.target_angle_rad)
        
        # Reinitialize learning process for current segment count
        use_nn = self.learning_process.use_neural_network if self.learning_process else True
        self.init_learning_process(use_nn)
        
        self.step_count = 0
        self.start_pos = self.creature.head_pos.copy()
        
        self.trajectory = [{
            'pos': self.start_pos.copy(),
            'segment_angles': self.creature.segment_angles[:self.creature.num_segments].copy(),
            'num_segments': self.creature.num_segments,
            'energy': self.creature.energy,
            'base_orientation': self.creature.base_orientation
        }]
        
        return self.creature.get_state()
    
    def step(self, action):
        """Execute one environment step."""
        energy_cost, segment_changed = self._apply_action(action)
        
        for _ in range(self.substeps):
            self.physics_model.update(self.creature, self.dt / self.substeps)
        
        metabolism_cost = self.creature.consume_energy_metabolism()
        total_energy_cost = energy_cost + metabolism_cost
        
        self.trajectory.append({
            'pos': self.creature.head_pos.copy(),
            'segment_angles': self.creature.segment_angles[:self.creature.num_segments].copy(),
            'num_segments': self.creature.num_segments,
            'energy': self.creature.energy,
            'base_orientation': self.creature.base_orientation
        })
        
        creature_state = {
            'head_pos': self.creature.head_pos,
            'head_vel': self.creature.head_vel
        }
        
        reward, info = self.learning_process.calculate_reward(
            creature_state,
            self.creature.direction_vector,
            self.start_pos,
            total_energy_cost
        )
        info['segment_changed'] = segment_changed
        info['energy_cost'] = total_energy_cost
        
        self.step_count += 1
        done = self.step_count >= self.max_steps or not (self.creature.energy > 0)
        
        next_state = self.creature.get_state()
        return next_state, float(-reward), done, info
    
    def _apply_action(self, action):
        """Apply action to creature."""
        segment_changed = None
        angle_step = 0.2
        
        # Check for add/remove segment actions
        if not self.creature.fixed_segments:
            add_action = self.creature.num_segments * 3
            remove_action = self.creature.num_segments * 3 + 1
            
            if action == add_action:
                if self.creature.add_segment():
                    segment_changed = ("ADD", self.creature.num_segments)
                    self.init_learning_process(self.learning_process.use_neural_network)
                    return self.creature.energy_cost_add_segment * self.creature.get_body_size_factor(), segment_changed
                return 0.0, segment_changed
            
            elif action == remove_action:
                if self.creature.remove_segment():
                    segment_changed = ("REMOVE", self.creature.num_segments)
                    self.init_learning_process(self.learning_process.use_neural_network)
                    return self.creature.energy_cost_remove_segment * self.creature.get_body_size_factor(), segment_changed
                return 0.0, segment_changed
        
        # Segment rotation actions
        segment_idx = action // 3
        action_type = action % 3
        
        if segment_idx < self.creature.num_segments:
            if action_type == 0:  # Rotate counter-clockwise (open)
                current = self.creature.target_angles[segment_idx]
                self.creature.set_target_angle(segment_idx, current + angle_step)
                return self.creature.consume_energy_angle_move(angle_step), segment_changed
            
            elif action_type == 1:  # Rotate clockwise (close)
                current = self.creature.target_angles[segment_idx]
                self.creature.set_target_angle(segment_idx, current - angle_step)
                return self.creature.consume_energy_angle_move(angle_step), segment_changed
        
        # Hold action (action_type == 2)
        return 0.0, segment_changed


# ==========================================================
#       TRAINING SESSION
# ==========================================================

class TrainingSession:
    """Training session manager."""
    
    def __init__(self, env, episodes=300, checkpoint_interval=20):
        self.env = env
        self.episodes = episodes
        self.checkpoint_interval = checkpoint_interval
        self.checkpoints = []
        self.best_distance = -float('inf')
        self.best_trajectory = []
    
    def run(self):
        print("=" * 60)
        print("TRAINING AQUATIC CREATURE V11")
        print("=" * 60)
        
        learning = self.env.learning_process
        use_nn = learning.use_neural_network
        
        print(f"Using {'NEURAL NETWORK' if use_nn else 'TABULAR'} Q-learning")
        print(f"Initial state dim: {self.env.creature.num_segments}")
        print(f"Initial action dim: {self.env._get_action_dim()}")
        
        for ep in range(self.episodes + 1):
            state = self.env.reset()
            epsilon = learning.get_epsilon(ep, self.episodes)
            max_distance = 0
            
            for step in range(self.env.max_steps):
                action = learning.select_action(state, epsilon)
                next_state, cost, done, info = self.env.step(action)
                max_distance = max(max_distance, info['distance'])
                
                # Update Q-values (dimensions may have changed)
                if len(state) == len(next_state):
                    learning.update_q_values(state, action, cost, next_state, done)
                
                state = next_state
                if done:
                    break
            
            if max_distance > self.best_distance:
                self.best_distance = max_distance
                self.best_trajectory = copy.deepcopy(self.env.trajectory)
            
            if ep % self.checkpoint_interval == 0:
                checkpoint = TrainingCheckpoint(
                    episode=ep,
                    model_state=learning.get_model_state(),
                    trajectory=copy.deepcopy(self.best_trajectory),
                    max_distance=self.best_distance,
                    use_nn=use_nn,
                    num_segments=self.env.creature.num_segments
                )
                self.checkpoints.append(checkpoint)
                print(f"Ep {ep:3d} | Best: {self.best_distance:6.1f}m | Segs: {self.env.creature.num_segments} | State dim: {self.env.creature.num_segments}")
        
        print("-" * 60)
        print(f"Done! {len(self.checkpoints)} checkpoints saved.")
        
        return learning.model, self.checkpoints


class TrainingCheckpoint:
    """Checkpoint container for saving training state."""
    
    def __init__(self, episode, model_state, trajectory, max_distance, use_nn, num_segments):
        self.episode = episode
        self.model_state = copy.deepcopy(model_state)
        self.trajectory = copy.deepcopy(trajectory)
        self.max_distance = max_distance
        self.use_nn = use_nn
        self.num_segments = num_segments


# ==========================================================
#       MAIN EXECUTION
# ==========================================================

def ensure_result_folder():
    result_dir = os.path.join(os.getcwd(), "Result")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir


def save_simulation(checkpoints, use_nn, target_angle, duration, num_segments, fixed_segments):
    result_dir = ensure_result_folder()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method = "nn" if use_nn else "tabular"
    seg_info = f"seg{num_segments}" + ("" if fixed_segments else "var")
    filename = os.path.join(result_dir, f"aquatic_V11_{method}_{seg_info}_{timestamp}.pkl")
    
    save_data = {
        'checkpoints': checkpoints,
        'use_nn': use_nn,
        'target_angle': target_angle,
        'duration': duration,
        'num_segments': num_segments,
        'fixed_segments': fixed_segments,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(save_data, f)
    
    return filename


def main():
    print("\n" + "=" * 60)
    print("   AQUATIC CREATURE TRAINER - V11")
    print("=" * 60)
    
    try:
        episodes = int(input("Episodes [300]: ").strip() or 300)
        duration = float(input("Episode duration (s) [30]: ").strip() or 30)
        angle = float(input("Target direction (°) [0]: ").strip() or 0)
        checkpoint_interval = int(input("Checkpoint interval [20]: ").strip() or 20)
        
        seg_choice = input("Fix number of segments? (y/n) [y]: ").strip().lower()
        fixed_segments = seg_choice != 'n'
        
        if fixed_segments:
            num_segments = int(input("Number of segments (2-10) [5]: ").strip() or 5)
            num_segments = max(2, min(10, num_segments))
        else:
            num_segments = int(input("Initial number of segments [2]: ").strip() or 2)
            num_segments = max(2, min(10, num_segments))
        
        nn_choice = input("Use Neural Network? (y/n) [y]: ").strip().lower()
        use_nn = nn_choice != 'n'
    except:
        episodes, duration, angle, num_segments, fixed_segments, use_nn = 300, 30.0, 0.0, 5, True, True
    
    print()
    
    env = Environment.from_config_file("Environment.txt")
    
    if fixed_segments:
        env.creature.fixed_segments = True
        env.creature.num_segments = num_segments
    else:
        env.creature.fixed_segments = False
        env.creature.num_segments = num_segments
    
    env.creature.target_angle_deg = angle
    env.creature.target_angle_rad = math.radians(angle)
    env.creature.base_orientation = math.radians(angle)
    env.creature.direction_vector = np.array([
        math.cos(math.radians(angle)),
        math.sin(math.radians(angle))
    ])
    
    env.init_learning_process(use_nn)
    
    session = TrainingSession(env, episodes=episodes, checkpoint_interval=checkpoint_interval)
    model, checkpoints = session.run()
    
    saved_file = save_simulation(checkpoints, use_nn, angle, duration, num_segments, fixed_segments)
    print(f"\nSimulation saved to: {saved_file}")


if __name__ == "__main__":
    main()

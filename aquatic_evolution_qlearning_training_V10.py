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
#       LEARNING PROCESS (PINK/RED) - FOR STUDENTS
#  Contains: Q-Network, Q-Tabular, Action Selection, Reward
# ==========================================================

class QNetwork(nn.Module):
    
    """
    Neural network for Q-value approximation.
    Students can modify architecture for experimentation.
    """
    
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        
        # ===== STUDENT SECTION: Network Architecture =====
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
        # ==================================================
        
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def get_q_values(self, state):
        """Get Q-values for a given state."""
        with torch.no_grad():
            state_tensor = torch.from_numpy(state) if isinstance(state, np.ndarray) else state
            return self.network(state_tensor).numpy()

class QTabular:
    
    """
    Tabular Q-learning with state discretization.
    Good for understanding Q-learning fundamentals.
    """
    
    def __init__(self, state_dim, action_dim, num_bins=8):
        
        # The bins are a discretization of the continuous state
        # 8 bins = 8 classes 
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.num_bins   = num_bins
        self.q_table    = {}
        
        # ===== STUDENT SECTION: Learning Rate =====
        self.learning_rate = 0.15
        # ===========================================
    
    def discretize_state(self, state):
        """Convert continuous state to discrete representation."""
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
    
    """
    ============================================================
    LEARNING PROCESS - MANAGES Q-LEARNING COMPONENTS
    ============================================================
    
    This class encapsulates all Q-learning logic:
    - Action selection (exploration vs exploitation)
    - Reward calculation
    - Q-value updates
    
    Students should understand and modify the marked sections.
    """
    
    def __init__(self, state_dim, action_dim, use_neural_network=True):
        
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.use_neural_network = use_neural_network
        
        # Initialize appropriate Q-model
        if use_neural_network:
            self.model = QNetwork(state_dim, action_dim)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        else:
            self.model = QTabular(state_dim, action_dim)
            self.optimizer = None
        
        # ===== STUDENT SECTION: Discount Factor =====
        self.gamma = 0.95
        # =============================================
    
    # =====================
    # ACTION SELECTION
    # =====================
    
    def select_action(self, state, epsilon):
        """
        Select action using epsilon-greedy strategy.
        
        Args:
            state: Current state observation
            epsilon: Exploration rate (0=exploit, 1=explore)
        
        Returns:
            int: Selected action index
        """
        # ===== STUDENT SECTION: Exploration Strategy =====
        
        if random.random() < epsilon:
            # EXPLORATION: Random action
            action = random.randint(0, self.action_dim - 1)
        else:
            # EXPLOITATION: Best action according to Q-values
            if self.use_neural_network:
                with torch.no_grad():
                    state_tensor = torch.from_numpy(state)
                    q_values = self.model(state_tensor)
                    action = torch.argmin(q_values).item()
            else:
                q_values = self.model.get_q_values(state)
                action = int(np.argmin(q_values))
        
        # ==================================================
        return action
    
    def get_epsilon(self, episode, total_episodes):
        """
        Calculate exploration rate for current episode.
        
        Args:
            episode: Current episode number
            total_episodes: Total training episodes
        
        Returns:
            float: Epsilon value in [0.05, 1.0]
        """
        # ===== STUDENT SECTION: Decay Schedule =====
        
        # Linear decay
        epsilon = max(0.05, 1.0 - (episode / total_episodes) * 0.95)
        
        # Alternative: Exponential decay
        # epsilon = max(0.05, 0.99 ** episode)
        
        # ============================================
        return epsilon
    
    # =====================
    # REWARD FUNCTION
    # =====================
    
    def calculate_reward(self, creature_state, direction_vector, start_pos, energy_cost):
        """
        Calculate reward based on creature behavior.
        
        Args:
            creature_state: Dict with 'head_pos', 'head_vel', 'num_segments'
            direction_vector: Target direction unit vector
            start_pos: Starting position for distance calculation
            energy_cost: Energy consumed this step
        
        Returns:
            float: Reward value (negative = cost)
            dict: Additional info for logging
        """
        head_pos = creature_state['head_pos']
        head_vel = creature_state['head_vel']
        num_segments = creature_state['num_segments']
        
        # Distance in target direction
        delta = head_pos - start_pos
        distance = np.dot(delta, direction_vector)
        
        # Velocity alignment with target
        speed = np.linalg.norm(head_vel)
        if speed > 1e-6:
            vel_dir = head_vel / speed
            alignment = np.dot(vel_dir, direction_vector)
        else:
            alignment = 0.0
        
        # ===== STUDENT SECTION: Reward Components =====
        
        # Reward for moving in correct direction
        reward_speed = alignment * speed * 0.5
        
        # Reward for progress
        reward_distance = distance * 0.05
        
        # Penalty for energy use
        penalty_energy = energy_cost * 0.005
        
        reward = reward_speed + reward_distance - penalty_energy
        
        # ===============================================
        
        info = {
            'distance': distance,
            'speed': speed,
            'alignment': alignment,
            'num_segments': num_segments
        }
        
        return reward, info
    
    # =====================
    # Q-VALUE UPDATE
    # =====================
    
    def update_q_values(self, state, action, cost, next_state, done):
        """
        Update Q-values using Bellman equation.
        
        Q(s,a) = cost + gamma * min(Q(s', a'))
        
        Args:
            state, action, next_state: Transition
            cost: Immediate cost (negative reward)
            done: Episode termination flag
        
        Returns:
            float or None: Loss value (NN only)
        """
        # ===== STUDENT SECTION: Q-Update Logic =====
        
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
        
        # =============================================
    
    def get_model_state(self):
        """Get current model state for checkpointing."""
        if self.use_neural_network:
            return self.model.state_dict()
        else:
            return self.model.q_table.copy()


# ==========================================================
#       PHYSICS MODEL (ORANGE) - REPLACEABLE MODULE
#  Contains: water_drag, thrust_coeff, angle_speed
#  Function: update creature position and velocity
# ==========================================================

class PhysicsModel:
    
    def __init__(self, water_drag=0.05, thrust_coefficient=3.5, angle_speed=0.4):
        self.base_drag = water_drag
        self.thrust_coefficient = thrust_coefficient
        self.angle_speed = angle_speed
    
    def update(self, creature, dt):
        
        old_angles = creature.angles.copy()
        
        # 1. Update Angles (Same as before)
        for i in range(creature.num_segments):
            diff = creature.target_angles[i] - creature.angles[i]
            creature.angles[i] += diff * self.angle_speed
        
        thrust = np.array([0.0, 0.0])
        
        # Calculate the "Spread" of the creature for Drag calculations
        # Assuming angles are relative to the body axis
        total_spread = sum(abs(a) for a in creature.angles)
        
        # 2. Calculate Thrust (The "Jet" Effect)
        # We approximate the volume change by the change in angle sum
        # This assumes a simple V where angle change = volume change
        
        # Sum of angles determines the "volume" state roughly
        # Actually, for a simple V, the angle of the last segment relative to center is key
        # Let's simplify: Thrust is generated based on the velocity of the segments closing.
        
        for i in range(creature.num_segments):
            angle_change = creature.angles[i] - old_angles[i]
            
            # We only care about the rate of closure
            # If angle_change < 0 (closing), we want positive thrust (forward)
            # If angle_change > 0 (opening), we want negative thrust (suction/backward)
            
            # Direction of the segment (body frame)
            # Note: This logic assumes the creature tries to align with base_orientation
            # and the angles define the V shape relative to the centerline.
            
            # Let's use a simplified Jet Model:
            # Force = -k * (dAngle/dt) * Forward_Direction
            
            # Determine forward direction
            forward_dir = np.array([math.cos(creature.base_orientation), 
                                    math.sin(creature.base_orientation)])
            
            # Positive angle_change = Opening = Suction (Negative Thrust)
            # Negative angle_change = Closing = Jet (Positive Thrust)
            # We use -angle_change so that closing gives positive force
            thrust_mag = -angle_change * self.thrust_coefficient
            
            # Apply this as an axial force (along the body axis)
            thrust += thrust_mag * forward_dir
            
            # Optional: Add a small lateral component if you want it to wiggle
            # But for pure V-propulsion, the force is axial.
            
        # 3. Apply Thrust
        creature.head_vel = creature.head_vel + thrust * dt
        
        # 4. Apply Dynamic Drag (The "Parachute" Effect)
        speed = np.linalg.norm(creature.head_vel)
        if speed > 0.001:
            # Drag coefficient increases as the creature opens up
            # When spread is 0 (closed), drag is low.
            # When spread is large (open), drag is high.
            current_drag_coeff = self.base_drag * (1.0 + 2.0 * total_spread)
            
            drag = -current_drag_coeff * creature.head_vel * speed
            creature.head_vel = creature.head_vel + drag * dt
        
        # 5. Update Position
        creature.head_pos = creature.head_pos + creature.head_vel * dt
        
        return thrust

# ==========================================================
#       CREATURE (PURPLE/BLUE) - STATE CONTAINER
#  Contains: segments, angles, position, velocity, orientation
# ==========================================================

def parse_creature_config(filepath):
    """
    Parses creature configuration from a text file.
    Returns a dictionary with the extracted values.
    """
    config = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            # Skip empty lines and comments
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Split on '=' and clean up
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Try to convert to appropriate type
                try:
                    # Try int first, then float, then bool, then string
                    if value.lower() == 'true':
                        config[key] = True
                    elif value.lower() == 'false':
                        config[key] = False
                    elif '.' in value:
                        config[key] = float(value)
                    else:
                        config[key] = int(value)
                except ValueError:
                    config[key] = value
    
    return config

class Creature:
    
    """
    Creature state container.
    """
    
    def __init__(self,
                 segment_length=3, size_penalty_factor=0.25,
                 fixed_segments=True, num_segments=2,
                 min_segments=2, max_segments=10,
                 target_angle_deg=0, max_angle=180,
                 use_nn=True):
        
        # ===== CREATURE PARAMETERS =====
        self.segment_length      = segment_length       # The size of each member of the creature
        self.size_penalty_factor = size_penalty_factor  # The factor for each new member in energy consumption
        
        self.fixed_segments = fixed_segments # Can the creature grow/lose members?
        self.num_segments   = num_segments   # If no, there's a fixed number of members
        self.min_segments   = min_segments   # If yes, there's a max/min number of members
        self.max_segments   = max_segments
        
        # The direction in which the creature should progress
        self.target_angle_deg = target_angle_deg 
        self.target_angle_rad = math.radians(target_angle_deg)
        self.base_orientation = math.radians(target_angle_deg)
        
        self.max_angle = math.radians(max_angle) # Maximum angle between 2 members
        self.use_nn    = use_nn                  # Can the creature use a NN to evolve?
        
        # ===== STATE VARIABLES =====
        
        # If the creature can grow, the initial number of states is set to the minimum one
        if not fixed_segments: self.num_segments = min_segments
        
        # The angles between the members are used as states
        # There are N-1 angles in the creature with N members + the orientation of the head
        # Thus, potentially N different angles to modify
        self.angles = np.zeros(max_segments, dtype=np.float64)
        self.target_angles = np.zeros(max_segments, dtype=np.float64)
        
        # Position and velocity vectors of the head
        self.head_pos = np.array([0.0, 0.0], dtype=np.float64)
        self.head_vel = np.array([0.0, 0.0], dtype=np.float64)
        
        # Direction vector for movement
        self.direction_vector = np.array([math.cos(self.target_angle_rad),math.sin(self.target_angle_rad)])
    
    def reset(self, x=0.0, y=0.0, orientation=None):
        
        """
        Reset creature to initial straight-line configuration.
        
        Args:
            x, y: Starting position
            orientation: Heading direction (default: target direction)
        """
        
        # Reset to the desired progress direction
        if orientation is None:
            orientation = self.target_angle_rad
        
        # Straight body: all relative angles are zero
        self.angles        = np.zeros(self.max_segments, dtype=np.float64)
        self.target_angles = np.zeros(self.max_segments, dtype=np.float64)
        
        # Set heading
        self.base_orientation = orientation
        
        # Reset position and velocity
        self.head_pos = np.array([x, y], dtype=np.float64)
        self.head_vel = np.array([0.0, 0.0], dtype=np.float64)
    
    def add_segment(self):
        """Add a segment if possible."""
        if self.num_segments < self.max_segments:
            self.num_segments += 1
            return True
        return False
    
    def remove_segment(self):
        """Remove a segment if possible."""
        if self.num_segments > self.min_segments:
            self.num_segments -= 1
            return True
        return False
    
    def get_body_size_factor(self):
        """Get size factor based on segment count."""
        return 1.0 + (self.num_segments - 2) * self.size_penalty_factor
    
    def set_target_angle(self, angle_idx, angle):
        """Set target angle for an angle between 2 segments."""
        if 0 <= angle_idx < self.num_segments:
            self.target_angles[angle_idx] = np.clip(angle, -self.max_angle, self.max_angle)
    
    def get_state(self, max_segments):
        """Get state vector for Q-learning."""
        
        # The current angular values and errors
        angles_sin   = np.sin(self.angles[:self.num_segments])
        angles_cos   = np.cos(self.angles[:self.num_segments])
        angle_errors = (self.target_angles[:self.num_segments] - self.angles[:self.num_segments]) / self.max_angle
        
        # Add 0 for potential angles that could appear by adding members
        pad_size     = max_segments - self.num_segments
        angles_sin   = np.pad(angles_sin, (0, pad_size))
        angles_cos   = np.pad(angles_cos, (0, pad_size))
        angle_errors = np.pad(angle_errors, (0, pad_size))
        
        return np.concatenate([angles_sin, angles_cos, angle_errors, self.head_pos, self.head_vel, self.num_segments / max_segments]).astype(np.float32)

# ==========================================================
#       ENVIRONMENT (GREEN/TEAL) - ORCHESTRATOR
#  Contains: simulation parameters, physics, learning processes
# ==========================================================

class Environment:
    
    """
    Environment orchestrator.
    """
    
    def __init__(self,
                 min_segments=2, max_segments=10,
                 duration_simulation=30.0, dt=0.01):
        
        # ===== ENVIRONMENT PARAMETERS =====
        
        # The number of segments of the creatures are bounded by the environnement
        self.min_segments = min_segments
        self.max_segments = max_segments
        
        # Defines the duration of the simulation in seconds
        self.duration_simulation = duration_simulation
        self.dt                  = dt
        self.max_steps           = int(duration_simulation / dt)
        
        # ===== CREATURE =====
        
        # Create creature from a txt file
        creature_dic = parse_creature_config("Creature.txt")
        
        self.creature = Creature(
            segment_length      = float(creature_dic['segment_length']),
            size_penalty_factor = float(creature_dic['size_penalty_factor']),
            
            fixed_segments = bool(creature_dic['fixed_segments']),
            num_segments   = int(creature_dic['num_segments']),
            min_segments   = int(max(min_segments, creature_dic['min_segments'])),
            max_segments   = int(min(max_segments, creature_dic['max_segments'])),
            
            target_angle_deg = float(creature_dic['target_angle_deg']),
            max_angle        = float(creature_dic['max_angle']),
            use_nn           = bool(creature_dic['use_nn']))
        
        # ===== PHYSIC MODEL =====
        self.physics_model = PhysicsModel()
        
        # ===== LEARNING PROCESSES =====
        self.learning_process_nn = None
        self.learning_process_tabular = None
        
        # ======================
        
        # Energy system
        self.base_energy = 100.0
        self.max_energy  = 100.0
        self.energy      = self.max_energy
        
        # Tracking
        self.step_count = 0
        self.start_pos = np.array([0.0, 0.0])
        self.trajectory = []
    
    def _init_learning_process(self, use_nn):
        """Initialize learning process with correct dimensions."""
        state_dim = self._get_state_dim()
        action_dim = self._get_action_dim()
        
        # Create both learning processes (for flexibility)
        self.learning_process_nn = LearningProcess(state_dim, action_dim, use_neural_network=True)
        self.learning_process_tabular = LearningProcess(state_dim, action_dim, use_neural_network=False)
        
        # Set active learning process
        self.active_learning = self.learning_process_nn if use_nn else self.learning_process_tabular
    
    def _get_state_dim(self):
        return self.max_segments * 3 + 3
    
    def _get_action_dim(self):
        if self.fixed_segments:
            return self.max_segments * 3
        return self.max_segments * 3 + 2
    
    def reset(self):
        """Reset environment for new episode."""
        # Reset creature
        self.creature.reset(0.0, 0.0, self.creature.target_angle_rad)
        
        # Reset energy
        self.max_energy = self.base_energy
        self.energy = self.max_energy
        
        # Reset counters
        self.step_count = 0
        self.start_pos = self.creature.head_pos.copy()
        
        # Reset trajectory
        self.trajectory = [{
            'pos': self.start_pos.copy(),
            'angles': self.creature.angles[:self.num_segments].copy(),
            'num_segments': self.creature.num_segments,
            'energy': self.energy,
            'base_orientation': self.creature.base_orientation
        }]
        
        return self.creature.get_state(self.max_segments)
    
    def step(self, action):
        """
        Execute one environment step.
        
        Args:
            action: Action index
        
        Returns:
            next_state, cost, done, info
        """
        # Parse action
        segment_idx = action // 3
        action_type = action % 3
        
        size_factor = self.creature.get_body_size_factor()
        energy_cost = 0.0
        segment_changed = None
        
        # Apply action to creature
        energy_cost, segment_changed = self._apply_action(
            action, segment_idx, action_type, size_factor
        )
        
        # Update physics
        substeps = 5
        for _ in range(substeps):
            self.physics_model.update(self.creature, self.dt / substeps)
        
        # Update energy
        metabolism = 0.002 * size_factor
        self.energy -= (energy_cost + metabolism)
        self.energy = max(0, self.energy)
        
        # Record trajectory
        head_pos = self.creature.head_pos.copy()
        self.trajectory.append({
            'pos': head_pos.copy(),
            'angles': self.creature.angles[:self.num_segments].copy(),
            'num_segments': self.creature.num_segments,
            'energy': self.energy,
            'base_orientation': self.creature.base_orientation
        })
        
        # Calculate reward via learning process
        creature_state = {
            'head_pos': self.creature.head_pos,
            'head_vel': self.creature.head_vel,
            'num_segments': self.creature.num_segments
        }
        
        reward, info = self.active_learning.calculate_reward(
            creature_state,
            self.creature.direction_vector,
            self.start_pos,
            energy_cost
        )
        info['segment_changed'] = segment_changed
        
        self.step_count += 1
        done = self.step_count >= self.max_steps or self.energy <= 0
        
        next_state = self.creature.get_state(self.max_segments)
        return next_state, float(-reward), done, info
    
    def _apply_action(self, action, segment_idx, action_type, size_factor):
        """Apply action to creature. Returns (energy_cost, segment_changed)."""
        energy_cost = 0.0
        segment_changed = None
        
        max_angle = self.physics_model.max_angle
        
        if not self.fixed_segments:
            add_action = self.max_segments * 3
            remove_action = self.max_segments * 3 + 1
            
            if action == add_action:
                if self.creature.add_segment():
                    segment_changed = ("ADD", self.creature.num_segments)
                    energy_cost = 3.0 * size_factor
                    self.max_energy = self.base_energy * size_factor
                    self.energy = min(self.energy, self.max_energy)
            
            elif action == remove_action:
                if self.creature.remove_segment():
                    segment_changed = ("REMOVE", self.creature.num_segments)
                    energy_cost = 1.5 * size_factor
                    self.max_energy = self.base_energy * self.creature.get_body_size_factor()
                    self.energy = min(self.energy, self.max_energy)
            
            elif segment_idx < self.creature.num_segments:
                energy_cost = self._modify_angle(segment_idx, action_type, size_factor, max_angle)
        else:
            if segment_idx < self.creature.num_segments:
                energy_cost = self._modify_angle(segment_idx, action_type, size_factor, max_angle)
        
        return energy_cost, segment_changed
    
    def _modify_angle(self, segment_idx, action_type, size_factor, max_angle):
        """Modify creature angle. Returns energy cost."""
        if action_type == 0:  # Increase
            current = self.creature.target_angles[segment_idx]
            self.creature.set_target_angle(segment_idx, current + 0.2, max_angle)
            return 0.03 * size_factor
        elif action_type == 1:  # Decrease
            current = self.creature.target_angles[segment_idx]
            self.creature.set_target_angle(segment_idx, current - 0.2, max_angle)
            return 0.03 * size_factor
        return 0.0  # Hold
    
    def get_state(self):
        return self.creature.get_state(self.max_segments)
    
    def get_distance_from_start(self, pos):
        """Calculate distance traveled in target direction."""
        delta = pos - self.start_pos
        return np.dot(delta, self.creature.direction_vector)


class TrainingSession:
    """
    Training session manager.
    Handles the training loop and checkpointing.
    """
    
    def __init__(self, env, episodes=300, checkpoint_interval=20):
        self.env = env
        self.episodes = episodes
        self.checkpoint_interval = checkpoint_interval
        self.checkpoints = []
        self.best_distance = -float('inf')
        self.best_trajectory = []
    
    def run(self):
        """Run training loop."""
        print("=" * 60)
        print("TRAINING AQUATIC CREATURE V9 (Modular Architecture)")
        print("=" * 60)
        
        learning = self.env.active_learning
        use_nn = learning.use_neural_network
        
        print(f"Using {'NEURAL NETWORK' if use_nn else 'TABULAR'} Q-learning")
        
        for ep in range(self.episodes + 1):
            state = self.env.reset()
            epsilon = learning.get_epsilon(ep, self.episodes)
            max_distance = 0
            
            for step in range(self.env.max_steps):
                # Select action via learning process
                action = learning.select_action(state, epsilon)
                
                # Execute action in environment
                next_state, cost, done, info = self.env.step(action)
                max_distance = max(max_distance, info['distance'])
                
                # Update Q-values via learning process
                learning.update_q_values(state, action, cost, next_state, done)
                
                state = next_state
                if done:
                    break
            
            # Track best performance
            if max_distance > self.best_distance:
                self.best_distance = max_distance
                self.best_trajectory = copy.deepcopy(self.env.trajectory)
            
            # Save checkpoint
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
                print(f"Ep {ep:3d} | Best: {self.best_distance:6.1f}m | Segs: {self.env.creature.num_segments}")
        
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
    filename = os.path.join(result_dir, f"aquatic_V9_{method}_{seg_info}_{timestamp}.pkl")
    
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
    print("   AQUATIC CREATURE TRAINER - V9 (Modular)")
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
    
    # Create environment
    env = Environment(
        min_segments=num_segments if not fixed_segments else num_segments,
        max_segments=10,
        duration_simulation=duration
    )
    
    # Create and run training session
    session = TrainingSession(env, episodes=episodes, checkpoint_interval=checkpoint_interval)
    model, checkpoints = session.run()
    
    # Save results
    saved_file = save_simulation(checkpoints, use_nn, angle, duration, num_segments, fixed_segments)
    print(f"\nSimulation saved to: {saved_file}")


if __name__ == "__main__":
    main()
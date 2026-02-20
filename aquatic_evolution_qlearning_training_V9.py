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
#                    PHYSICS MODEL SECTION
#        (This section contains the physical simulation)
#      Can be replaced with alternative physics models
# ==========================================================

class AquaticCreaturePhysics:
    
    """
    Physics model for aquatic creature movement.
    This class handles all physical properties and calculations.
    Can be replaced with alternative physics implementations.
    """
    
    def __init__(self, num_segments=5, max_segments=10, min_segments=2):
        
        self.num_segments = num_segments
        self.max_segments = max_segments
        self.min_segments = min_segments
        
        # Physical dimensions
        self.segment_length = 10.0
        
        # Relative angles between consecutive segments (NOT absolute orientations)
        self.angles = np.zeros(max_segments, dtype=np.float64)
        self.target_angles = np.zeros(max_segments, dtype=np.float64)
        
        # Base orientation: the direction the creature's head points (radians)
        # This is the creature's overall heading, separate from relative angles
        self.base_orientation = 0.0
        
        # Head position and velocity in world coordinates
        self.head_pos = np.array([0.0, 0.0], dtype=np.float64)
        self.head_vel = np.array([0.0, 0.0], dtype=np.float64)
        
        # Physics parameters
        self.water_drag = 0.05
        self.thrust_coefficient = 3.5
        self.angle_speed = 0.4
        self.max_angle = 1.2
    
    def reset(self, x=0.0, y=0.0, orientation=0.0):
        
        # All relative angles are 0 (straight body)
        self.angles = np.zeros(self.max_segments, dtype=np.float64)
        self.target_angles = np.zeros(self.max_segments, dtype=np.float64)
        
        # Set the creature's overall heading
        self.base_orientation = orientation
        
        self.head_pos = np.array([x, y], dtype=np.float64)
        self.head_vel = np.array([0.0, 0.0], dtype=np.float64)
    
    def set_target_angle(self, segment_idx, angle):
        if 0 <= segment_idx < self.num_segments:
            self.target_angles[segment_idx] = np.clip(angle, -self.max_angle, self.max_angle)
    
    def add_segment(self):
        if self.num_segments < self.max_segments:
            self.num_segments += 1
            return True
        return False
    
    def remove_segment(self):
        if self.num_segments > self.min_segments:
            self.num_segments -= 1
            return True
        return False
    
    def get_body_size_factor(self):
        return 1.0 + (self.num_segments - 2) * 0.25
    
    def get_segment_positions(self):
        
        positions = [self.head_pos.copy()]
        cum_angle = self.base_orientation
        
        for i in range(self.num_segments):
            cum_angle += self.angles[i]
            prev_pos = positions[-1]
            # Each segment extends backward from the head
            new_pos = prev_pos - self.segment_length * np.array([math.cos(cum_angle),math.sin(cum_angle)])
            positions.append(new_pos)
        
        return positions
    
    def update(self, dt):
        
        old_angles = self.angles.copy()
        
        # Smooth angle transitions
        for i in range(self.num_segments):
            diff = self.target_angles[i] - self.angles[i]
            self.angles[i] += diff * self.angle_speed
        
        # Calculate thrust from angular motion
        thrust = np.array([0.0, 0.0])
        
        for i in range(self.num_segments):
            angle_change = self.angles[i] - old_angles[i]
            
            if abs(angle_change) > 0.001:
                # Cumulative angle includes base orientation
                cum_angle = self.base_orientation + sum(self.angles[:i+1])
                seg_dir = np.array([math.cos(cum_angle), math.sin(cum_angle)])
                perp_dir = np.array([-seg_dir[1], seg_dir[0]])
                
                position_factor = (i + 1) / self.num_segments
                thrust_mag = abs(angle_change) * self.thrust_coefficient * position_factor
                
                thrust += thrust_mag * perp_dir * np.sign(angle_change)
                thrust[0] += thrust_mag * 0.5 * seg_dir[0]
        
        # Apply thrust to velocity
        self.head_vel += thrust * dt
        
        # Apply water drag
        speed = np.linalg.norm(self.head_vel)
        if speed > 0.001:
            drag = -self.water_drag * self.head_vel * speed
            self.head_vel += drag * dt
        
        # Update position
        self.head_pos += self.head_vel * dt
        
        return thrust
    
    def get_state(self, max_segments):
        
        angles_sin = np.sin(self.angles[:self.num_segments])
        angles_cos = np.cos(self.angles[:self.num_segments])
        angle_errors = (self.target_angles[:self.num_segments] - self.angles[:self.num_segments]) / self.max_angle
        
        pad_size = max_segments - self.num_segments
        angles_sin = np.pad(angles_sin, (0, pad_size))
        angles_cos = np.pad(angles_cos, (0, pad_size))
        angle_errors = np.pad(angle_errors, (0, pad_size))
        
        return np.concatenate([angles_sin, angles_cos, angle_errors, self.head_vel / 10.0, [self.num_segments / max_segments]]).astype(np.float32)


# ==========================================================
#                  CREATURE INTERFACE
#    (Simplified interface for environment interaction)
# ==========================================================

class AquaticCreature:
    
    """
    Simplified creature interface that wraps the physics model.
    Provides cached values to reduce redundant calculations.
    """
    
    def __init__(self, num_segments=5, max_segments=10, min_segments=2):
        
        # Use provided physics
        self.physics = AquaticCreaturePhysics(num_segments, max_segments, min_segments)
        
        # Cached values
        self._cached_size_factor = None
        self._cache_valid = False
    
    def _invalidate_cache(self):
        self._cache_valid = False
    
    def _get_size_factor(self):
        if not self._cache_valid:
            self._cached_size_factor = self.physics.get_body_size_factor()
            self._cache_valid = True
        return self._cached_size_factor
    
    # Delegate properties to physics model
    @property
    def num_segments(self):
        return self.physics.num_segments
    
    @property
    def angles(self):
        return self.physics.angles
    
    @property
    def target_angles(self):
        return self.physics.target_angles
    
    @property
    def head_pos(self):
        return self.physics.head_pos
    
    @property
    def head_vel(self):
        return self.physics.head_vel
    
    def reset(self, x=0.0, y=0.0, orientation=0.0):
        self.physics.reset(x, y, orientation)
        self._invalidate_cache()
    
    def set_target_angle(self, segment_idx, angle):
        self.physics.set_target_angle(segment_idx, angle)
    
    def add_segment(self):
        result = self.physics.add_segment()
        if result:
            self._invalidate_cache()
        return result
    
    def remove_segment(self):
        result = self.physics.remove_segment()
        if result:
            self._invalidate_cache()
        return result
    
    def get_body_size_factor(self):
        return self._get_size_factor()
    
    def update(self, dt):
        return self.physics.update(dt)
    
    def get_state(self, max_segments):
        return self.physics.get_state(max_segments)
    
    def get_head_position(self):
        return self.physics.head_pos.copy()
    
    def get_segment_positions(self):
        return self.physics.get_segment_positions()


# ==========================================================
#            REWARD FUNCTION SECTION (FOR STUDENTS)
#   Students should understand and potentially modify this
# ==========================================================

def calculate_reward(env, head_pos, energy_cost, segment_changed):
    
    """
    ============================================================
    REWARD FUNCTION - TO BE IMPLEMENTED/MODIFIED BY STUDENTS
    ============================================================
    
    This function calculates the reward for the creature based on
    its current state. Students should understand how different
    reward components affect learning behavior.
    
    Current implementation considers:
    - Distance traveled in target direction
    - Velocity alignment with target direction
    - Energy cost penalty
    
    Args:
        env: The environment containing creature and target info
        head_pos: Current head position [x, y]
        energy_cost: Energy consumed this step
        segment_changed: Whether body structure changed
    
    Returns:
        float: The reward value (negative = cost, positive = reward)
    """
    
    # Get distance traveled in target direction
    distance = env.get_distance(head_pos)
    
    # Get velocity and speed
    vel = env.creature.head_vel
    speed = np.linalg.norm(vel)
    
    # Calculate velocity alignment with target direction
    # alignment ranges from -1 (opposite) to 1 (same direction)
    if speed > 1e-6:
        vel_dir = vel / speed
        alignment = np.dot(vel_dir, env.direction_vector)
    else:
        alignment = 0.0
    
    # ===== STUDENT SECTION: Modify reward components =====
    # Current reward combines:
    # 1. Speed in correct direction (alignment * speed)
    # 2. Distance progress
    # 3. Energy cost penalty
    
    reward_speed = alignment * speed * 0.5
    reward_distance = distance * 0.05
    penalty_energy = energy_cost * 0.005
    
    reward = reward_speed + reward_distance - penalty_energy
    # =====================================================
    
    return reward, {'distance': distance,'speed': speed,'alignment': alignment,'segment_changed': segment_changed,'num_segments': env.creature.num_segments}


# ==========================================================
#          ACTION SELECTION SECTION (FOR STUDENTS)
#    Students implement exploration vs exploitation
# ==========================================================

def select_action(model, state, epsilon, action_dim, use_neural_network=True):
    
    """
    ============================================================
    ACTION SELECTION - TO BE IMPLEMENTED BY STUDENTS
    ============================================================
    
    This function implements the epsilon-greedy action selection
    strategy for exploration vs exploitation trade-off.
    
    Args:
        model: The Q-learning model (neural network or tabular)
        state: Current state observation
        epsilon: Exploration rate (0 = pure exploitation, 1 = pure exploration)
        action_dim: Number of possible actions
        use_neural_network: Whether using neural network or tabular Q
    
    Returns:
        int: Selected action index
    """
    # ===== STUDENT SECTION: Implement exploration strategy =====
    # Current implementation: epsilon-greedy
    
    if random.random() < epsilon:
        # EXPLORATION: Random action
        action = random.randint(0, action_dim - 1)
    else:
        # EXPLOITATION: Choose best action according to Q-values
        if use_neural_network:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state)
                q_values = model(state_tensor)
                # Note: Using argmin because we're minimizing cost
                action = torch.argmin(q_values).item()
        else:
            q_values = model.get_q_values(state)
            action = int(np.argmin(q_values))
    
    # ============================================================
    return action


def get_epsilon(episode, total_episodes):
    """
    ============================================================
    EXPLORATION DECAY - TO BE IMPLEMENTED BY STUDENTS
    ============================================================
    
    Calculate exploration rate for current episode.
    Students can experiment with different decay schedules.
    
    Args:
        episode: Current episode number
        total_episodes: Total number of training episodes
    
    Returns:
        float: Epsilon value between 0 and 1
    """
    # ===== STUDENT SECTION: Implement decay schedule =====
    # Current: Linear decay from 1.0 to 0.05
    
    epsilon = max(0.05, 1.0 - (episode / total_episodes) * 0.95)
    
    # Alternative: Exponential decay
    # epsilon = max(0.05, 0.99 ** episode)
    
    # Alternative: Step decay
    # epsilon = max(0.05, 1.0 - (episode // 100) * 0.2)
    
    # ======================================================
    return epsilon


# ==========================================================
#             Q-LEARNING SECTION (FOR STUDENTS)
#   Core Q-learning algorithm implementation
# ==========================================================

class QNetwork(nn.Module):
    """
    ============================================================
    Q-NETWORK - NEURAL NETWORK FOR Q-VALUE APPROXIMATION
    ============================================================
    
    This neural network approximates Q(s,a) for continuous states.
    Students can modify the architecture (layers, activations).
    """
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        
        # ===== STUDENT SECTION: Modify network architecture =====
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
        # =========================================================
        
        # Initialize weights
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


class TabularQ:
    """
    ============================================================
    TABULAR Q-LEARNING - DISCRETE STATE-ACTION VALUES
    ============================================================
    
    Simple tabular Q-learning with state discretization.
    Good for understanding Q-learning fundamentals.
    """
    def __init__(self, state_dim, action_dim, num_bins=8):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.q_table = {}
        
        # ===== STUDENT SECTION: Adjust learning rate =====
        self.learning_rate = 0.15
        # =================================================
    
    def discretize_state(self, state):
        """Convert continuous state to discrete bin indices."""
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
    
    def get_state_dict(self):
        return self.q_table.copy()


def update_q_values(model, state, action, cost, next_state, done,
                    use_neural_network=True, optimizer=None, gamma=0.95):
    """
    ============================================================
    Q-VALUE UPDATE - CORE Q-LEARNING ALGORITHM
    ============================================================
    
    Update Q-values using the Bellman equation:
    Q(s,a) = cost + gamma * min(Q(s', a'))
    
    Note: We minimize cost (negative reward), hence min instead of max.
    
    Args:
        model: Q-learning model
        state, action, next_state: Transition data
        cost: Immediate cost (negative reward)
        done: Whether episode terminated
        use_neural_network: Model type flag
        optimizer: PyTorch optimizer (for neural network)
        gamma: Discount factor
    
    Returns:
        float: Loss value (for neural network) or None
    """
    # ===== STUDENT SECTION: Implement Q-value update =====
    
    if use_neural_network:
        # Neural network Q-learning update
        with torch.no_grad():
            q_next = model(torch.from_numpy(next_state))
            target = cost + gamma * torch.min(q_next).item()
        
        current_q = model(torch.from_numpy(state))[action]
        loss = nn.MSELoss()(current_q, torch.tensor(target, dtype=torch.float32))
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        return loss.item()
    else:
        # Tabular Q-learning update
        q_next = model.get_q_values(next_state)
        target = cost + gamma * np.min(q_next)
        model.update(state, action, target)
        return None
    
    # ======================================================


# ==========================================================
#                    ENVIRONMENT
# ==========================================================

class AquaticEnv:
    """
    Aquatic environment for creature simulation.
    Coordinates the creature, target direction, and simulation loop.
    """
    
    def __init__(self, num_segments=5, max_segments=10, min_segments=2,
                 fixed_segments=True, target_angle_deg=0.0,
                 simulation_duration=30.0, dt=0.01):
        
        self.initial_segments = num_segments
        self.max_segments = max_segments if not fixed_segments else num_segments
        self.min_segments = min_segments if not fixed_segments else num_segments
        self.fixed_segments = fixed_segments
        
        # Target direction setup
        self.target_angle_deg = target_angle_deg
        self.target_angle_rad = math.radians(target_angle_deg)
        self.direction_vector = np.array([
            math.cos(self.target_angle_rad),
            math.sin(self.target_angle_rad)
        ])
        
        # Simulation parameters
        self.simulation_duration = simulation_duration
        self.dt = dt
        self.max_steps = int(simulation_duration / dt)
        
        # Energy system
        self.base_energy = 100.0
        self.max_energy = 100.0
        self.energy = self.max_energy
        
        # Create creature
        self.creature = AquaticCreature(num_segments, self.max_segments, self.min_segments)
        self.reset()
    
    def reset(self):
        """Reset environment and creature for new episode."""
        self.creature = AquaticCreature(
            self.initial_segments, self.max_segments, self.min_segments
        )
        
        # FIX: Reset creature with orientation, not individual angles
        # Creature starts as straight line pointing in target direction
        self.creature.reset(0.0, 0.0, self.target_angle_rad)
        
        self.max_energy = self.base_energy
        self.energy = self.max_energy
        self.step_count = 0
        self.start_pos = self.creature.get_head_position().copy()
        
        self.trajectory = [{
            'pos': self.start_pos.copy(),
            'angles': self.creature.angles[:self.creature.num_segments].copy(),
            'num_segments': self.creature.num_segments,
            'energy': self.energy,
            'base_orientation': self.creature.physics.base_orientation
        }]
        
        return self.get_state()
    
    def get_state_dim(self):
        return self.max_segments * 3 + 3
    
    def get_action_dim(self):
        if self.fixed_segments:
            return self.max_segments * 3
        return self.max_segments * 3 + 2
    
    def get_state(self):
        return self.creature.get_state(self.max_segments)
    
    def get_distance(self, pos):
        """Get distance traveled in target direction."""
        delta = pos - self.start_pos
        return np.dot(delta, self.direction_vector)
    
    def step(self, action):
        """
        Execute one environment step.
        Consolidates creature interactions to reduce redundancy.
        """
        # Parse action (cached values)
        segment_idx = action // 3
        action_type = action % 3
        
        # Get size factor once (cached)
        size_factor = self.creature.get_body_size_factor()
        energy_cost = 0.0
        segment_changed = None
        
        # Handle action types
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
                energy_cost = self._apply_angle_action(segment_idx, action_type, size_factor)
        else:
            if segment_idx < self.creature.num_segments:
                energy_cost = self._apply_angle_action(segment_idx, action_type, size_factor)
        
        # Physics update
        substeps = 5
        for _ in range(substeps):
            self.creature.update(self.dt / substeps)
        
        # Energy consumption
        metabolism = 0.002 * size_factor
        self.energy -= (energy_cost + metabolism)
        self.energy = max(0, self.energy)
        
        # Record trajectory
        head_pos = self.creature.get_head_position()
        self.trajectory.append({
            'pos': head_pos.copy(),
            'angles': self.creature.angles[:self.creature.num_segments].copy(),
            'num_segments': self.creature.num_segments,
            'energy': self.energy,
            'base_orientation': self.creature.physics.base_orientation
        })
        
        # Calculate reward using separate function
        reward, info = calculate_reward(self, head_pos, energy_cost, segment_changed)
        
        self.step_count += 1
        done = self.step_count >= self.max_steps or self.energy <= 0
        
        return self.get_state(), float(-reward), done, info
    
    def _apply_angle_action(self, segment_idx, action_type, size_factor):
        """Apply angle modification action. Returns energy cost."""
        current_target = self.creature.target_angles[segment_idx]
        
        if action_type == 0:  # Increase angle
            self.creature.set_target_angle(segment_idx, current_target + 0.2)
        elif action_type == 1:  # Decrease angle
            self.creature.set_target_angle(segment_idx, current_target - 0.2)
        # action_type == 2: No change (hold)
        
        return 0.03 * size_factor if action_type in [0, 1] else 0.0


# ==========================================================
#                    TRAINING
# ==========================================================

class TrainingCheckpoint:
    def __init__(self, episode, model_state, trajectory, max_distance, use_nn, num_segments):
        self.episode = episode
        self.model_state = copy.deepcopy(model_state)
        self.trajectory = copy.deepcopy(trajectory)
        self.max_distance = max_distance
        self.use_nn = use_nn
        self.num_segments = num_segments


def train_with_checkpoints(episodes=300, checkpoint_interval=20,
                           target_angle_deg=0.0, simulation_duration=30.0,
                           use_neural_network=False, num_segments=5, fixed_segments=True):
    
    env = AquaticEnv(num_segments=num_segments,fixed_segments=fixed_segments,
                     target_angle_deg=target_angle_deg,simulation_duration=simulation_duration)
    
    # Initialize Q-learning model
    if use_neural_network:
        model = QNetwork(env.get_state_dim(), env.get_action_dim())
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        print("Using NEURAL NETWORK for Q-estimation")
    else:
        model = TabularQ(env.get_state_dim(), env.get_action_dim())
        optimizer = None
        print("Using TABULAR Q-learning")
    
    checkpoints = []
    best_distance = -float('inf')
    best_trajectory = []
    
    print("=" * 60)
    print("TRAINING AQUATIC CREATURE V9 (Refactored)")
    print("=" * 60)
    
    for ep in range(episodes + 1):
        state = env.reset()
        
        # Get exploration rate using separate function
        epsilon = get_epsilon(ep, episodes)
        max_distance = 0
        
        for step in range(env.max_steps):
            # Action selection using separate function
            action = select_action(
                model, state, epsilon,
                env.get_action_dim(), use_neural_network
            )
            
            next_state, cost, done, info = env.step(action)
            max_distance = max(max_distance, info['distance'])
            
            # Q-value update using separate function
            update_q_values(
                model, state, action, cost, next_state, done,
                use_neural_network, optimizer
            )
            
            state = next_state
            if done:
                break
        
        if max_distance > best_distance:
            best_distance = max_distance
            best_trajectory = copy.deepcopy(env.trajectory)
        
        if ep % checkpoint_interval == 0:
            if use_neural_network:
                model_state = model.state_dict()
            else:
                model_state = model.get_state_dict()
            
            checkpoint = TrainingCheckpoint(
                episode=ep,
                model_state=model_state,
                trajectory=copy.deepcopy(best_trajectory),
                max_distance=best_distance,
                use_nn=use_neural_network,
                num_segments=num_segments
            )
            checkpoints.append(checkpoint)
            print(f"Ep {ep:3d} | Best: {best_distance:6.1f}m | Segs: {num_segments}")
    
    print("-" * 60)
    print(f"Done! {len(checkpoints)} checkpoints saved.")
    
    return model, checkpoints


# ==========================================================
#                    MAIN EXECUTION
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
    
    save_data = {'checkpoints': checkpoints,'use_nn': use_nn,'target_angle': target_angle,'duration': duration,
                 'num_segments': num_segments,'fixed_segments': fixed_segments,'timestamp': datetime.now().isoformat()}
    
    with open(filename, 'wb') as f:
        pickle.dump(save_data, f)
    
    return filename


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   AQUATIC CREATURE TRAINER - V9 (Refactored)")
    print("=" * 60)
    
    try:
        episodes            = int(input("Episodes [300]: ").strip() or 300)
        duration            = float(input("Episode duration (s) [30]: ").strip() or 30)
        angle               = float(input("Target direction (°) [0]: ").strip() or 0)
        checkpoint_interval = int(input("Checkpoint interval for simulation [20]: ").strip() or 20)
        
        seg_choice = input("Fix number of segments? (y/n) [n]: ").strip().lower()
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
    model, checkpoints = train_with_checkpoints(episodes=episodes, checkpoint_interval=checkpoint_interval,
                                                target_angle_deg=angle, simulation_duration=duration,
                                                use_neural_network=use_nn, num_segments=num_segments,
                                                fixed_segments=fixed_segments)
    
    saved_file = save_simulation(checkpoints, use_nn, angle, duration, num_segments, fixed_segments)
    print(f"\nSimulation saved to: {saved_file}")
    print("Run visualization script to see results.")

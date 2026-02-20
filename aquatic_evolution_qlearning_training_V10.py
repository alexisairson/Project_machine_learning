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
    
    def forward(self, state):
        return self.network(state)
    
    def get_q_values(self, state):
        """Get Q-values for a given state."""
        with torch.no_grad():
            state_tensor = torch.from_numpy(state) if isinstance(state, np.ndarray) else state
            return self.forward(state_tensor).numpy()
    
    def get_state_dict(self):
        return self.state_dict()


class QTabular:
    """
    Tabular Q-learning with state discretization.
    Good for understanding Q-learning fundamentals.
    """
    
    def __init__(self, state_dim, action_dim, num_bins=8):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.q_table = {}
        
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
    
    def get_state_dict(self):
        return self.q_table.copy()


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
        self.state_dim = state_dim
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
        
        # ===== STUDENT SECTION: Exploration Strategy =====
        
        if random.random() < epsilon:
            # EXPLORATION: Random action
            action = random.randint(0, self.action_dim - 1)
        else:
            # EXPLOITATION: Best action according to Q-values
            if self.use_neural_network:
                with torch.no_grad():
                    q_values = self.model(torch.from_numpy(state))
                    action = torch.argmin(q_values).item()
            else:
                q_values = self.model.get_q_values(state)
                action = int(np.argmin(q_values))
        
        # ==================================================
        return action
    
    def get_epsilon(self, episode, total_episodes):
        
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
        
        info = {'distance': distance,'speed': speed,'alignment': alignment,'num_segments': num_segments}
        return reward, info
    
    # =====================
    # Q-VALUE UPDATE
    # =====================
    
    def update_q_values(self, state, action, cost, next_state, done):
        
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
        if self.use_neural_network:
            return self.model.state_dict()
        else:
            return self.model.get_state_dict()


# ==========================================================
#       PHYSICS MODEL (ORANGE) - REPLACEABLE MODULE
#  Contains: water_drag, thrust_coeff, angle_speed, max_angle
#  Function: update creature position and velocity
# ==========================================================

class PhysicsModel:
    
    """
    Physics model for aquatic creature simulation.
    This module can be replaced with alternative physics implementations.
    
    Responsibilities:
    - Apply thrust based on angular motion
    - Apply water drag
    - Update position from velocity
    """
    
    def __init__(self):
        
        # ===== PHYSICS PARAMETERS =====
        self.water_drag         = 0.05
        self.thrust_coefficient = 3.5
        self.angle_speed        = 0.4
        self.max_angle          = 1.2
        # ==============================
    
    def update(self, creature, dt):
        
        """
        Update creature position and velocity based on physics.
        
        Args:
            creature: Creature object to update
            dt: Time step
        
        Returns:
            np.array: Thrust vector generated
        """
        
        # Creature attributes
        num_segments     = creature.num_segments
        base_orientation = creature.base_orientation
        
        # Vectorized angle update
        old_angles  = creature.angles[:num_segments].copy()
        angle_diffs = creature.target_angles[:num_segments] - old_angles
        new_angles  = old_angles + angle_diffs * self.angle_speed
        
        # Vectorized cumulative angles (cumsum)
        cum_angles = base_orientation + np.cumsum(new_angles)
        
        # Vectorized thrust calculation
        angle_changes = new_angles - old_angles
        valid_mask = np.abs(angle_changes) > 0.001
        
        seg_dirs  = np.column_stack([np.cos(cum_angles), np.sin(cum_angles)])
        perp_dirs = np.column_stack([-seg_dirs[:, 1], seg_dirs[:, 0]])
        
        position_factors = np.arange(1, num_segments + 1) / num_segments
        thrust_mags = np.abs(angle_changes) * self.thrust_coefficient * position_factors
        
        # Only sum valid contributions
        signs = np.sign(angle_changes)
        lateral_thrust = np.sum((thrust_mags * signs)[:, None] * perp_dirs * valid_mask[:, None], axis=0)
        forward_thrust = np.sum(thrust_mags * 0.5 * seg_dirs[:, 0] * valid_mask)
        
        thrust = lateral_thrust + np.array([forward_thrust, 0.0])
        
        # Physics update (vectorized)
        head_vel = creature.head_vel + thrust * dt
        speed = np.linalg.norm(head_vel)
        if speed > 0.001:
            head_vel += -self.water_drag * head_vel * speed * dt
        
        head_pos = creature.head_pos + head_vel * dt
        
        # Write back
        creature.angles[:num_segments] = new_angles
        creature.head_vel = head_vel
        creature.head_pos = head_pos
        
        return thrust

# ==========================================================
#       CREATURE (PURPLE/BLUE) - STATE CONTAINER
#  Contains: segments, angles, position, velocity, orientation
# ==========================================================

class Creature:
    
    """
    Creature state container.
    
    Holds all state information for an aquatic creature:
    - Segment configuration
    - Joint angles
    - Position and velocity
    - Target direction
    """
    
    def __init__(self,
                 min_segments=2, max_segments=10,
                 fixed_segments=True, segment_length=10,
                 target_angle_deg=0.0, use_nn=True):
        
        # ===== CREATURE PARAMETERS =====
        self.min_segments   = min_segments
        self.max_segments   = max_segments
        self.fixed_segments = fixed_segments
        self.segment_length = segment_length
        
        self.target_angle_deg = target_angle_deg
        self.target_angle_rad = math.radians(target_angle_deg)
        self.use_nn = use_nn
        
        # ===== STATE VARIABLES =====
        self.num_segments = min_segments
        
        self.base_orientation = 0.0
        self.angles           = np.zeros(max_segments, dtype=np.float64)
        self.target_angles    = np.zeros(max_segments, dtype=np.float64)
        
        self.head_pos = np.array([0.0, 0.0], dtype=np.float64)
        self.head_vel = np.array([0.0, 0.0], dtype=np.float64)
        
        # ============================
        
        # Direction vector for movement
        self.direction_vector = np.array([math.cos(self.target_angle_rad),math.sin(self.target_angle_rad)])
    
    def reset(self, x=0.0, y=0.0):
        
        """
        Reset creature to initial straight-line configuration.
        
        Args:
            x, y: Starting position
        """
        
        # Straight body: all relative angles are zero
        self.angles = np.zeros(self.max_segments, dtype=np.float64)
        self.target_angles = np.zeros(self.max_segments, dtype=np.float64)
        
        # Set heading
        self.base_orientation = self.target_angle_rad
        
        # Reset position and velocity
        self.head_pos = np.array([x, y], dtype=np.float64)
        self.head_vel = np.array([0.0, 0.0], dtype=np.float64)
        
    # ==========================================================
    
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
    
    # ==========================================================
    
    def set_target_angle(self, segment_idx, angle, max_angle=1.2):
        if 0 <= segment_idx < self.num_segments:
            self.target_angles[segment_idx] = np.clip(angle, -max_angle, max_angle)
    
    def get_body_size_factor(self):
        return 1.0 + (self.num_segments - 2) * 0.25
    
    def get_state(self, max_segments, max_angle):
        
        angles_sin   = np.sin(self.angles[:self.num_segments])
        angles_cos   = np.cos(self.angles[:self.num_segments])
        angle_errors = (self.target_angles[:self.num_segments] - self.angles[:self.num_segments]) / max_angle
        
        pad_size     = max_segments - self.num_segments
        angles_sin   = np.pad(angles_sin, (0, pad_size))
        angles_cos   = np.pad(angles_cos, (0, pad_size))
        angle_errors = np.pad(angle_errors, (0, pad_size))
        
        # Dividing the velocity by 10 for neural networks efficiency
        return np.concatenate([angles_sin, angles_cos, angle_errors,
                               self.head_vel / 10.0, [self.num_segments / max_segments]]).astype(np.float32)

# ==========================================================
#       ENVIRONMENT (GREEN/TEAL) - ORCHESTRATOR
#  Contains: simulation parameters, physics, learning processes
# ==========================================================

class Environment:
    """
    Environment orchestrator.
    
    Coordinates:
    - Creature state
    - Physics simulation
    - Learning process
    - Training loop
    
    This class only calls methods of other classes when necessary.
    """
    
    def __init__(self, min_segments=2, max_segments=10, fixed_segments=True,
                 target_angle_deg=0.0, use_nn=True, duration_simulation=30.0, dt=0.01):
        
        # ===== ENVIRONMENT PARAMETERS =====
        self.min_segments = min_segments
        self.max_segments = max_segments
        self.fixed_segments = fixed_segments
        self.duration_simulation = duration_simulation
        self.dt = dt
        self.max_steps = int(duration_simulation / dt)
        self.target_angle_deg = target_angle_deg
        # ===================================
        
        # ===== COMPONENTS =====
        # Create creature
        self.creature = Creature(
            min_segments=min_segments,
            max_segments=max_segments,
            fixed_segments=fixed_segments,
            target_angle_deg=target_angle_deg,
            use_nn=use_nn
        )
        
        # Create physics model
        self.physics_model = PhysicsModel()
        
        # Create learning process (will be initialized with proper dimensions)
        self.learning_process_nn = None
        self.learning_process_tabular = None
        self._init_learning_process(use_nn)
        # ======================
        
        # Energy system
        self.base_energy = 100.0
        self.max_energy = 100.0
        self.energy = self.max_energy
        
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
        self.episode      = episode
        self.model_state  = copy.deepcopy(model_state)
        self.trajectory   = copy.deepcopy(trajectory)
        self.max_distance = max_distance
        self.use_nn       = use_nn
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
    
    save_data = {'checkpoints': checkpoints,'use_nn': use_nn,'target_angle': target_angle,'duration': duration,
                 'num_segments': num_segments,'fixed_segments': fixed_segments,'timestamp': datetime.now().isoformat()}
    
    with open(filename, 'wb') as f:
        pickle.dump(save_data, f)
    
    return filename


def main():
    print("\n" + "=" * 60)
    print("   AQUATIC CREATURE TRAINER - V9 (Modular)")
    print("=" * 60)
    
    try:
        episodes            = int(input("Episodes [300]: ").strip() or 300)
        duration            = float(input("Episode duration (s) [30]: ").strip() or 30)
        angle               = float(input("Target direction (°) [0]: ").strip() or 0)
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
    env = Environment(min_segments=num_segments,max_segments=10,fixed_segments=fixed_segments,
                      target_angle_deg=angle,use_nn=use_nn,duration_simulation=duration)
    
    # Create and run training session
    session = TrainingSession(env, episodes=episodes, checkpoint_interval=checkpoint_interval)
    _ , checkpoints = session.run()
    
    # Save results
    saved_file = save_simulation(checkpoints, use_nn, angle, duration, num_segments, fixed_segments)
    print(f"\nSimulation saved to: {saved_file}")


if __name__ == "__main__":
    main()

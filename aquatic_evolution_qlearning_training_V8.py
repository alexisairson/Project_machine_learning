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
# 1. Neural Network (Q-Network) - V8 Enhanced
# ==========================================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256), # Added LayerNorm for stability
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

# ==========================================================
# 2. Tabular Q-Learning
# ==========================================================
class TabularQ:
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
    
    def get_state_dict(self):
        return self.q_table.copy()

# ==========================================================
# 3. Aquatic Creature
# ==========================================================
# class AquaticCreature:
#     def __init__(self, num_segments=5, max_segments=10, min_segments=2):
#         self.num_segments = num_segments
#         self.max_segments = max_segments
#         self.min_segments = min_segments
        
#         self.segment_length = 15.0
#         self.angles = np.zeros(max_segments, dtype=np.float64)
#         self.target_angles = np.zeros(max_segments, dtype=np.float64)
        
#         self.head_pos = np.array([0.0, 0.0], dtype=np.float64)
#         self.head_vel = np.array([0.0, 0.0], dtype=np.float64)
        
#         # V8 Tuning: Better physics
#         self.water_drag = 0.05        # Reduced drag
#         self.thrust_coefficient = 3.5 # Increased thrust
#         self.angle_speed = 0.4        # Faster reaction
#         self.max_angle = 1.2
        
#     def reset(self, x=0.0, y=0.0, initial_angle=0.0):
#         self.angles = np.zeros(self.max_segments, dtype=np.float64)
#         self.target_angles = np.zeros(self.max_segments, dtype=np.float64)
#         for i in range(self.num_segments):
#             self.angles[i] = initial_angle
#             self.target_angles[i] = initial_angle
#         self.head_pos = np.array([x, y], dtype=np.float64)
#         self.head_vel = np.array([0.0, 0.0], dtype=np.float64)
    
#     def get_joint_positions(self):
#         positions = [self.head_pos.copy()]
#         cum_angle = 0.0
#         for i in range(self.num_segments):
#             cum_angle += self.angles[i]
#             prev = positions[-1]
#             new_x = prev[0] - self.segment_length * math.cos(cum_angle)
#             new_y = prev[1] - self.segment_length * math.sin(cum_angle)
#             positions.append(np.array([new_x, new_y]))
#         return positions
    
#     def set_target_angle(self, segment_idx, angle):
#         if 0 <= segment_idx < self.num_segments:
#             self.target_angles[segment_idx] = np.clip(angle, -self.max_angle, self.max_angle)
    
#     def add_segment(self):
#         if self.num_segments < self.max_segments:
#             self.num_segments += 1
#             return True
#         return False
    
#     def remove_segment(self):
#         if self.num_segments > self.min_segments:
#             self.num_segments -= 1
#             return True
#         return False
    
#     def get_body_size_factor(self):
#         return 1.0 + (self.num_segments - 2) * 0.25
    
#     def update(self, dt):
#         old_angles = self.angles.copy()
        
#         for i in range(self.num_segments):
#             diff = self.target_angles[i] - self.angles[i]
#             self.angles[i] += diff * self.angle_speed
        
#         thrust = np.array([0.0, 0.0])
        
#         for i in range(self.num_segments):
#             angle_change = self.angles[i] - old_angles[i]
            
#             if abs(angle_change) > 0.001:
#                 cum_angle = sum(self.angles[:i+1])
#                 seg_dir = np.array([math.cos(cum_angle), math.sin(cum_angle)])
#                 perp_dir = np.array([-seg_dir[1], seg_dir[0]])
                
#                 position_factor = (i + 1) / self.num_segments
#                 thrust_mag = abs(angle_change) * self.thrust_coefficient * position_factor
                
#                 thrust += thrust_mag * perp_dir * np.sign(angle_change)
#                 thrust[0] += thrust_mag * 0.5 * seg_dir[0]
        
#         self.head_vel += thrust * dt
        
#         speed = np.linalg.norm(self.head_vel)
#         if speed > 0.001:
#             drag = -self.water_drag * self.head_vel * speed
#             self.head_vel += drag * dt
        
#         self.head_pos += self.head_vel * dt
    
#     def get_head_position(self):
#         return self.head_pos.copy()
    
#     def get_state(self, max_segments):
#         angles_sin = np.sin(self.angles[:self.num_segments])
#         angles_cos = np.cos(self.angles[:self.num_segments])
#         angle_errors = (self.target_angles[:self.num_segments] - self.angles[:self.num_segments]) / self.max_angle
        
#         pad_size = max_segments - self.num_segments
#         angles_sin = np.pad(angles_sin, (0, pad_size))
#         angles_cos = np.pad(angles_cos, (0, pad_size))
#         angle_errors = np.pad(angle_errors, (0, pad_size))
        
#         return np.concatenate([
#             angles_sin,
#             angles_cos,
#             angle_errors,
#             self.head_vel / 10.0,
#             [self.num_segments / max_segments]
#         ]).astype(np.float32)

# ==========================================================
# 4. Environment
# ==========================================================
class AquaticEnv:
    
    def __init__(self,
                 num_segments=5, max_segments=10, min_segments=2, fixed_segments=True,
                 target_angle_deg=0.0, simulation_duration=10.0, dt=0.01):
        
        self.initial_segments = num_segments
        self.max_segments     = max_segments if not fixed_segments else num_segments
        self.min_segments     = min_segments if not fixed_segments else num_segments
        self.fixed_segments   = fixed_segments
        
        self.target_angle_deg = target_angle_deg
        self.target_angle_rad = math.radians(target_angle_deg)
        self.direction_vector = np.array([math.cos(self.target_angle_rad),math.sin(self.target_angle_rad)])
        
        self.simulation_duration = simulation_duration
        self.dt = dt
        self.max_steps = int(simulation_duration / dt)
        
        self.base_energy = 100.0
        self.max_energy  = 100.0
        self.energy      = self.max_energy
        
        self.creature = AquaticCreature(num_segments, max_segments, self.min_segments)
        self.reset()
    
    def reset(self):
        self.creature = AquaticCreature(self.initial_segments, self.max_segments, self.min_segments)
        self.creature.reset(0.0, 0.0, self.target_angle_rad)
        
        self.max_energy = self.base_energy
        self.energy     = self.max_energy
        
        self.step_count = 0
        self.start_pos = self.creature.get_head_position().copy()
        self.trajectory = [{'pos': self.start_pos.copy(), 
                           'angles': self.creature.angles[:self.creature.num_segments].copy(),
                           'num_segments': self.creature.num_segments,
                           'energy': self.energy}]
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
        delta = pos - self.start_pos
        return np.dot(delta, self.direction_vector)
    
    # def step(self, action):
    #     segment_idx = action // 3
    #     action_type = action % 3
        
    #     size_factor = self.creature.get_body_size_factor()
    #     energy_cost = 0.0
    #     segment_changed = None
        
    #     if not self.fixed_segments:
    #         add_action = self.max_segments * 3
    #         remove_action = self.max_segments * 3 + 1
            
    #         if action == add_action:
    #             if self.creature.add_segment():
    #                 segment_changed = ("ADD", self.creature.num_segments)
    #                 energy_cost = 3.0 * size_factor
    #                 self.max_energy = self.base_energy * size_factor
    #                 self.energy = min(self.energy, self.max_energy)
    #         elif action == remove_action:
    #             if self.creature.remove_segment():
    #                 segment_changed = ("REMOVE", self.creature.num_segments)
    #                 energy_cost = 1.5 * size_factor
    #                 self.max_energy = self.base_energy * self.creature.get_body_size_factor()
    #                 self.energy = min(self.energy, self.max_energy)
    #         elif segment_idx < self.creature.num_segments:
    #             current_target = self.creature.target_angles[segment_idx]
    #             if action_type == 0:
    #                 self.creature.set_target_angle(segment_idx, current_target + 0.2)
    #                 energy_cost = 0.03 * size_factor # Reduced cost
    #             elif action_type == 1:
    #                 self.creature.set_target_angle(segment_idx, current_target - 0.2)
    #                 energy_cost = 0.03 * size_factor
    #     else:
    #         if segment_idx < self.creature.num_segments:
    #             current_target = self.creature.target_angles[segment_idx]
    #             if action_type == 0:
    #                 self.creature.set_target_angle(segment_idx, current_target + 0.2)
    #                 energy_cost = 0.03 * size_factor
    #             elif action_type == 1:
    #                 self.creature.set_target_angle(segment_idx, current_target - 0.2)
    #                 energy_cost = 0.03 * size_factor
        
    #     # Update physics
    #     substeps = 4 # More substeps for stability
    #     for _ in range(substeps):
    #         self.creature.update(self.dt / substeps)
        
    #     # Metabolism cost reduced
    #     metabolism = 0.002 * size_factor
    #     self.energy -= (energy_cost + metabolism)
    #     self.energy = max(0, self.energy)
        
    #     head_pos = self.creature.get_head_position()
    #     self.trajectory.append({
    #         'pos': head_pos.copy(),
    #         'angles': self.creature.angles[:self.creature.num_segments].copy(),
    #         'num_segments': self.creature.num_segments,
    #         'energy': self.energy
    #     })
        
    #     # V8 Enhanced Reward Function
    #     distance = self.get_distance(head_pos)
        
    #     # Velocity alignment reward
    #     vel = self.creature.head_vel
    #     speed = np.linalg.norm(vel)
    #     vel_dir = vel / (speed + 1e-6)
    #     alignment = np.dot(vel_dir, self.direction_vector)
        
    #     # Reward is mainly moving fast in the right direction
    #     reward = (alignment * speed * 0.5) + (distance * 0.05) - (energy_cost * 0.005)
        
    #     self.step_count += 1
    #     done = self.step_count >= self.max_steps or self.energy <= 0
        
    #     return self.get_state(), float(-reward), done, {
    #         'distance': distance,
    #         'segment_changed': segment_changed,
    #         'num_segments': self.creature.num_segments,
    #         'speed': speed
    #     }

# ==========================================================
# 5. Training
# ==========================================================
class TrainingCheckpoint:
    
    def __init__(self, episode, model_state, trajectory, max_distance, use_nn, num_segments):
        self.episode = episode
        self.model_state = copy.deepcopy(model_state)
        self.trajectory  = copy.deepcopy(trajectory)
        self.max_distance = max_distance
        self.use_nn = use_nn
        self.num_segments = num_segments
        
def train_with_checkpoints(episodes=200, checkpoint_interval=20,
                          target_angle_deg=0.0, simulation_duration=10.0,
                          use_neural_network=True, num_segments=5, fixed_segments=True):
    env = AquaticEnv(
        num_segments=num_segments,
        target_angle_deg=target_angle_deg,
        simulation_duration=simulation_duration,
        fixed_segments=fixed_segments
    )
    
#     if use_neural_network:
#         model = QNetwork(env.get_state_dim(), env.get_action_dim())
#         optimizer = optim.Adam(model.parameters(), lr=0.001)
#         print("Using NEURAL NETWORK for Q-estimation")
#     else:
#         model = TabularQ(env.get_state_dim(), env.get_action_dim())
#         optimizer = None
#         print("Using TABULAR Q-learning")
    
    checkpoints = []
    best_distance = -float('inf')
    best_trajectory = []
    
    print("=" * 60)
    print("TRAINING AQUATIC CREATURE V8")
    print("=" * 60)
    
#     for ep in range(episodes + 1):
#         state = env.reset()
#         epsilon = max(0.05, 1.0 - (ep / episodes) * 0.95)
#         max_distance = 0
        
#         for step in range(env.max_steps):
#             if random.random() < epsilon:
#                 action = random.randint(0, env.get_action_dim() - 1)
#             else:
#                 if use_neural_network:
#                     with torch.no_grad():
#                         state_tensor = torch.from_numpy(state)
#                         q_values = model(state_tensor)
#                         action = torch.argmin(q_values).item()
#                 else:
#                     q_values = model.get_q_values(state)
#                     action = int(np.argmin(q_values))
            
#             next_state, cost, done, info = env.step(action)
#             max_distance = max(max_distance, info['distance'])
            
#             if use_neural_network:
#                 with torch.no_grad():
#                     q_next = model(torch.from_numpy(next_state))
#                     target = cost + 0.95 * torch.min(q_next).item()
                
#                 current_q = model(torch.from_numpy(state))[action]
#                 loss = nn.MSELoss()(current_q, torch.tensor(target, dtype=torch.float32))
                
#                 optimizer.zero_grad()
#                 loss.backward()
#                 # V8: Gradient clipping for stability
#                 nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#                 optimizer.step()
#             else:
#                 q_next = model.get_q_values(next_state)
#                 target = cost + 0.95 * np.min(q_next)
#                 model.update(state, action, target)
            
#             state = next_state
#             if done:
#                 break
        
#         if max_distance > best_distance:
#             best_distance = max_distance
#             best_trajectory = copy.deepcopy(env.trajectory)
        
#         if ep % checkpoint_interval == 0:
#             if use_neural_network:
#                 model_state = model.state_dict()
#             else:
#                 model_state = model.get_state_dict()
            
#             checkpoint = TrainingCheckpoint(
#                 episode=ep,
#                 model_state=model_state,
#                 trajectory=copy.deepcopy(best_trajectory),
#                 max_distance=best_distance,
#                 use_nn=use_neural_network,
#                 num_segments=num_segments
#             )
#             checkpoints.append(checkpoint)
#             print(f"Ep {ep:3d} | Max dist: {max_distance:6.1f}m | Best: {best_distance:6.1f}m")
    
#     print("-" * 60)
#     print(f"Done! {len(checkpoints)} checkpoints saved.")
    
#     return model, checkpoints

# ==========================================================
# 6. Main Execution
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
    filename = os.path.join(result_dir, f"aquatic_V8_{method}_{seg_info}_{timestamp}.pkl")
    
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

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   AQUATIC CREATURE TRAINER - V8")
    print("=" * 60)
    
    try:
        episodes = int(input("Episodes [300]: ").strip() or 300)
        duration = float(input("Episode duration (s) [30]: ").strip() or 30)
        angle = float(input("Target direction (°) [0]: ").strip() or 0)
        
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
    model, checkpoints = train_with_checkpoints(
        episodes=episodes,
        checkpoint_interval=20,
        target_angle_deg=angle,
        simulation_duration=duration,
        use_neural_network=use_nn,
        num_segments=num_segments,
        fixed_segments=fixed_segments
    )
    
    saved_file = save_simulation(checkpoints, use_nn, angle, duration, num_segments, fixed_segments)
    print(f"\nSimulation saved to: {saved_file}")
    print("Run 'aquatic_evolution_run_V8.py' to visualize results.")
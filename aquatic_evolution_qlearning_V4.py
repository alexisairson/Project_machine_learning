import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pygame
import time
import copy
import math

# ==========================================================
# 1. Neural Network (Q-Network) - Stronger Architecture
# ==========================================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        # Initialize weights for better learning
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)


# ==========================================================
# 2. Tabular Q-Learning (No Neural Network)
# ==========================================================
class TabularQ:
    """Simple tabular Q-learning with state discretization"""
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
        return copy.deepcopy(self.q_table)
    
    def load_state_dict(self, state_dict):
        self.q_table = copy.deepcopy(state_dict)


# ==========================================================
# 3. Aquatic Creature
# ==========================================================
class AquaticCreature:
    def __init__(self, num_segments=5, max_segments=5, min_segments=2):
        self.num_segments = num_segments
        self.max_segments = max_segments
        self.min_segments = min_segments
        
        self.segment_length = 15.0
        self.angles = np.zeros(max_segments, dtype=np.float64)
        self.target_angles = np.zeros(max_segments, dtype=np.float64)
        
        self.head_pos = np.array([0.0, 0.0], dtype=np.float64)
        self.head_vel = np.array([0.0, 0.0], dtype=np.float64)
        
        self.water_drag = 0.08
        self.thrust_coefficient = 2.0
        self.angle_speed = 0.3
        self.max_angle = 1.2
        
    def reset(self, x=0.0, y=0.0):
        self.angles = np.zeros(self.max_segments, dtype=np.float64)
        self.target_angles = np.zeros(self.max_segments, dtype=np.float64)
        self.head_pos = np.array([x, y], dtype=np.float64)
        self.head_vel = np.array([0.0, 0.0], dtype=np.float64)
    
    def get_joint_positions(self):
        positions = [self.head_pos.copy()]
        cum_angle = 0.0
        for i in range(self.num_segments):
            cum_angle += self.angles[i]
            prev = positions[-1]
            new_x = prev[0] - self.segment_length * math.cos(cum_angle)
            new_y = prev[1] - self.segment_length * math.sin(cum_angle)
            positions.append(np.array([new_x, new_y]))
        return positions
    
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
        """Energy cost multiplier based on body size"""
        return 1.0 + (self.num_segments - 2) * 0.25
    
    def update(self, dt):
        old_angles = self.angles.copy()
        
        for i in range(self.num_segments):
            diff = self.target_angles[i] - self.angles[i]
            self.angles[i] += diff * self.angle_speed
        
        thrust = np.array([0.0, 0.0])
        
        for i in range(self.num_segments):
            angle_change = self.angles[i] - old_angles[i]
            
            if abs(angle_change) > 0.001:
                cum_angle = sum(self.angles[:i+1])
                seg_dir = np.array([math.cos(cum_angle), math.sin(cum_angle)])
                perp_dir = np.array([-seg_dir[1], seg_dir[0]])
                
                position_factor = (i + 1) / self.num_segments
                thrust_mag = abs(angle_change) * self.thrust_coefficient * position_factor
                
                thrust += thrust_mag * perp_dir * np.sign(angle_change)
                thrust[0] += thrust_mag * 0.5 * seg_dir[0]
        
        self.head_vel += thrust * dt
        
        speed = np.linalg.norm(self.head_vel)
        if speed > 0.001:
            drag = -self.water_drag * self.head_vel * speed
            self.head_vel += drag * dt
        
        self.head_pos += self.head_vel * dt
    
    def get_head_position(self):
        return self.head_pos.copy()
    
    def get_state(self, max_segments):
        """Get state vector with padding for variable segments"""
        angles_sin = np.sin(self.angles[:self.num_segments])
        angles_cos = np.cos(self.angles[:self.num_segments])
        angle_errors = (self.target_angles[:self.num_segments] - self.angles[:self.num_segments]) / self.max_angle
        
        # Pad to max_segments
        pad_size = max_segments - self.num_segments
        angles_sin = np.pad(angles_sin, (0, pad_size))
        angles_cos = np.pad(angles_cos, (0, pad_size))
        angle_errors = np.pad(angle_errors, (0, pad_size))
        
        return np.concatenate([
            angles_sin,
            angles_cos,
            angle_errors,
            self.head_vel / 10.0,
            [self.num_segments / max_segments]  # Normalized segment count
        ]).astype(np.float32)


# ==========================================================
# 4. Environment
# ==========================================================
class AquaticEnv:
    def __init__(self, num_segments=5, max_segments=5, min_segments=2,
                 target_angle_deg=0.0, simulation_duration=10.0, dt=0.02):
        
        self.initial_segments = num_segments
        self.max_segments = max_segments
        self.min_segments = min_segments
        
        self.target_angle_deg = target_angle_deg
        self.target_angle_rad = math.radians(target_angle_deg)
        self.direction_vector = np.array([
            math.cos(self.target_angle_rad),
            math.sin(self.target_angle_rad)
        ])
        
        self.simulation_duration = simulation_duration
        self.dt = dt
        self.max_steps = int(simulation_duration / dt)
        
        self.base_energy = 100.0
        self.max_energy = 100.0
        self.energy = self.max_energy
        
        self.creature = AquaticCreature(num_segments, max_segments, min_segments)
        self.reset()
    
    def reset(self):
        self.creature = AquaticCreature(self.initial_segments, self.max_segments, self.min_segments)
        self.creature.reset(0.0, 0.0)
        
        # Energy scales with body size
        self.max_energy = self.base_energy
        self.energy = self.max_energy
        
        self.step_count = 0
        self.start_pos = self.creature.get_head_position().copy()
        self.trajectory = [{'pos': self.start_pos.copy(), 
                           'angles': self.creature.angles[:self.creature.num_segments].copy(),
                           'num_segments': self.creature.num_segments,
                           'energy': self.energy}]
        return self.get_state()
    
    def get_state_dim(self):
        # sin + cos + error for each segment + vel(2) + segment_count(1)
        return self.max_segments * 3 + 3
    
    def get_action_dim(self):
        # 3 per segment (increase, decrease, hold) + add_segment + remove_segment
        return self.max_segments * 3 + 2
    
    def get_state(self):
        return self.creature.get_state(self.max_segments)
    
    def get_distance(self, pos):
        delta = pos - self.start_pos
        return np.dot(delta, self.direction_vector)
    
    def step(self, action):
        segment_idx = action // 3
        action_type = action % 3
        
        size_factor = self.creature.get_body_size_factor()
        energy_cost = 0.0
        segment_changed = None
        
        add_action = self.max_segments * 3
        remove_action = self.max_segments * 3 + 1
        
        if action == add_action:
            if self.creature.add_segment():
                segment_changed = ("ADD", self.creature.num_segments)
                energy_cost = 3.0 * size_factor
                # Energy max also scales
                self.max_energy = self.base_energy * size_factor
                self.energy = min(self.energy, self.max_energy)
        elif action == remove_action:
            if self.creature.remove_segment():
                segment_changed = ("REMOVE", self.creature.num_segments)
                energy_cost = 1.5 * size_factor
                self.max_energy = self.base_energy * self.creature.get_body_size_factor()
                self.energy = min(self.energy, self.max_energy)
        elif segment_idx < self.creature.num_segments:
            current_target = self.creature.target_angles[segment_idx]
            if action_type == 0:
                self.creature.set_target_angle(segment_idx, current_target + 0.2)
                energy_cost = 0.05 * size_factor
            elif action_type == 1:
                self.creature.set_target_angle(segment_idx, current_target - 0.2)
                energy_cost = 0.05 * size_factor
        
        # Physics
        for _ in range(3):
            self.creature.update(self.dt / 3)
        
        # Energy: cost + metabolism (larger = more metabolism)
        metabolism = 0.005 * size_factor
        self.energy -= (energy_cost + metabolism)
        self.energy = max(0, self.energy)
        
        # Record trajectory
        head_pos = self.creature.get_head_position()
        self.trajectory.append({
            'pos': head_pos.copy(),
            'angles': self.creature.angles[:self.creature.num_segments].copy(),
            'num_segments': self.creature.num_segments,
            'energy': self.energy
        })
        
        distance = self.get_distance(head_pos)
        reward = distance * 0.1 - energy_cost * 0.01
        
        self.step_count += 1
        done = self.step_count >= self.max_steps or self.energy <= 0
        
        return self.get_state(), float(-reward), done, {
            'distance': distance,
            'segment_changed': segment_changed,
            'num_segments': self.creature.num_segments
        }


# ==========================================================
# 5. Training
# ==========================================================
class TrainingCheckpoint:
    def __init__(self, episode, model_state, trajectory, max_distance, use_nn):
        self.episode = episode
        self.model_state = copy.deepcopy(model_state)
        self.trajectory = copy.deepcopy(trajectory)
        self.max_distance = max_distance
        self.use_nn = use_nn


def train_with_checkpoints(episodes=200, checkpoint_interval=20,
                          target_angle_deg=0.0, simulation_duration=10.0,
                          use_neural_network=True):
    env = AquaticEnv(
        num_segments=5,
        target_angle_deg=target_angle_deg,
        simulation_duration=simulation_duration
    )
    
    if use_neural_network:
        model = QNetwork(env.get_state_dim(), env.get_action_dim())
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        print("Using NEURAL NETWORK for Q-estimation (stronger architecture)")
    else:
        model = TabularQ(env.get_state_dim(), env.get_action_dim())
        optimizer = None
        print("Using TABULAR Q-learning (no neural network)")
    
    checkpoints = []
    best_distance = -float('inf')
    best_trajectory = []
    
    print("=" * 60)
    print("TRAINING AQUATIC CREATURE")
    print("=" * 60)
    print(f"Episodes: {episodes}")
    print(f"Duration: {simulation_duration}s per episode")
    print(f"Target: {target_angle_deg}° (0°=right, 90°=up, 180°=left)")
    print(f"Method: {'Neural Network' if use_neural_network else 'Tabular Q-Learning'}")
    print(f"Segments: variable ({env.min_segments}-{env.max_segments})")
    print("-" * 60)
    
    for ep in range(episodes + 1):
        state = env.reset()
        epsilon = max(0.05, 1.0 - (ep / episodes) * 0.95)
        max_distance = 0
        
        for step in range(env.max_steps):
            if random.random() < epsilon:
                action = random.randint(0, env.get_action_dim() - 1)
            else:
                if use_neural_network:
                    with torch.no_grad():
                        state_tensor = torch.from_numpy(state)
                        q_values = model(state_tensor)
                        action = torch.argmin(q_values).item()
                else:
                    q_values = model.get_q_values(state)
                    action = int(np.argmin(q_values))
            
            next_state, cost, done, info = env.step(action)
            max_distance = max(max_distance, info['distance'])
            
            if use_neural_network:
                with torch.no_grad():
                    q_next = model(torch.from_numpy(next_state))
                    target = cost + 0.95 * torch.min(q_next).item()
                
                current_q = model(torch.from_numpy(state))[action]
                loss = nn.MSELoss()(current_q, torch.tensor(target, dtype=torch.float32))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                q_next = model.get_q_values(next_state)
                target = cost + 0.95 * np.min(q_next)
                model.update(state, action, target)
            
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
                use_nn=use_neural_network
            )
            checkpoints.append(checkpoint)
            print(f"Ep {ep:3d} | Max dist: {max_distance:6.1f}m | Best: {best_distance:6.1f}m | Segs: {info['num_segments']}")
    
    print("-" * 60)
    print(f"Done! {len(checkpoints)} checkpoints saved.")
    
    return model, checkpoints


# ==========================================================
# 6. UI Components
# ==========================================================
class UIIntSlider:
    def __init__(self, x, y, w, h, min_v, max_v, val, label=""):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_v, self.max_v = int(min_v), int(max_v)
        self.value = int(val)
        self.label = label
        self.dragging = False
        self.font = pygame.font.SysFont("Arial", 14)
    
    def get_knob_x(self):
        if self.max_v <= self.min_v:
            return self.rect.x
        r = (self.value - self.min_v) / (self.max_v - self.min_v)
        return int(self.rect.x + r * self.rect.width)
    
    def _set_from_x(self, x):
        r = max(0, min(1, (x - self.rect.x) / self.rect.width))
        self.value = int(self.min_v + r * (self.max_v - self.min_v))
    
    def handle_event(self, event, pos):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(pos) or abs(pos[0] - self.get_knob_x()) < 12:
                self.dragging = True
                self._set_from_x(pos[0])
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._set_from_x(pos[0])
    
    def draw(self, screen, font):
        screen.blit(font.render(self.label, True, (200, 200, 200)), (self.rect.x, self.rect.y - 18))
        pygame.draw.rect(screen, (60, 60, 80), 
                        pygame.Rect(self.rect.x, self.rect.centery - 2, self.rect.width, 4), border_radius=2)
        kx = self.get_knob_x()
        pygame.draw.rect(screen, (100, 180, 255),
                        pygame.Rect(self.rect.x, self.rect.centery - 2, kx - self.rect.x, 4), border_radius=2)
        pygame.draw.circle(screen, (120, 200, 255), (kx, self.rect.centery), 10)
        pygame.draw.circle(screen, (255, 255, 255), (kx, self.rect.centery), 7)
        screen.blit(self.font.render(str(self.value), True, (200, 200, 200)), 
                   (self.rect.x + self.rect.width + 8, self.rect.y - 2))


class UIFloatSlider:
    def __init__(self, x, y, w, h, min_v, max_v, val, label="", log=False):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_v = max(min_v, 0.001) if log else min_v
        self.max_v = max_v
        self.value = max(val, self.min_v) if log else val
        self.label = label
        self.log = log
        self.dragging = False
        self.font = pygame.font.SysFont("Arial", 14)
    
    def get_knob_x(self):
        if self.log:
            r = (math.log10(max(self.value, self.min_v)) - math.log10(self.min_v)) / \
                max(math.log10(self.max_v) - math.log10(self.min_v), 0.001)
        else:
            r = (self.value - self.min_v) / max(self.max_v - self.min_v, 0.001)
        return int(self.rect.x + r * self.rect.width)
    
    def _set_from_x(self, x):
        r = max(0, min(1, (x - self.rect.x) / self.rect.width))
        if self.log:
            self.value = 10 ** (math.log10(self.min_v) + r * (math.log10(self.max_v) - math.log10(self.min_v)))
        else:
            self.value = self.min_v + r * (self.max_v - self.min_v)
    
    def handle_event(self, event, pos):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(pos) or abs(pos[0] - self.get_knob_x()) < 12:
                self.dragging = True
                self._set_from_x(pos[0])
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._set_from_x(pos[0])
    
    def draw(self, screen, font):
        screen.blit(font.render(self.label, True, (200, 200, 200)), (self.rect.x, self.rect.y - 18))
        pygame.draw.rect(screen, (60, 60, 80), 
                        pygame.Rect(self.rect.x, self.rect.centery - 2, self.rect.width, 4), border_radius=2)
        kx = self.get_knob_x()
        pygame.draw.rect(screen, (100, 180, 255),
                        pygame.Rect(self.rect.x, self.rect.centery - 2, kx - self.rect.x, 4), border_radius=2)
        pygame.draw.circle(screen, (120, 200, 255), (kx, self.rect.centery), 10)
        pygame.draw.circle(screen, (255, 255, 255), (kx, self.rect.centery), 7)
        val_text = f"{self.value:.1f}" if self.value < 10 else f"{self.value:.0f}"
        screen.blit(self.font.render(val_text, True, (200, 200, 200)), 
                   (self.rect.x + self.rect.width + 8, self.rect.y - 2))


class UIButton:
    def __init__(self, x, y, w, h, text, color=(60, 60, 80)):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.hover = False
    
    def draw(self, screen, font):
        c = (80, 80, 100) if self.hover else self.color
        pygame.draw.rect(screen, c, self.rect, border_radius=8)
        pygame.draw.rect(screen, (150, 150, 170), self.rect, 2, border_radius=8)
        s = font.render(self.text, True, (255, 255, 255))
        screen.blit(s, s.get_rect(center=self.rect.center))
    
    def check_hover(self, pos):
        self.hover = self.rect.collidepoint(pos)
    
    def clicked(self, pos):
        return self.rect.collidepoint(pos)


# ==========================================================
# 7. Visualization with Zoom
# ==========================================================
def visualize(checkpoints, speed=1.0, target_angle=0.0, duration=10.0):
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Aquatic Evolution - Use mouse wheel to zoom")
    
    font = pygame.font.SysFont("Arial", 18, bold=True)
    small = pygame.font.SysFont("Arial", 14)
    tiny = pygame.font.SysFont("Arial", 12)
    
    clock = pygame.time.Clock()
    
    target_rad = math.radians(target_angle)
    direction = np.array([math.cos(target_rad), math.sin(target_rad)])
    
    # UI
    ep_slider = UIIntSlider(50, 530, 120, 18, 0, max(0, len(checkpoints)-1), 0, "Episode")
    sp_slider = UIFloatSlider(250, 530, 120, 18, 0.01, 100.0, speed, "Speed", log=True)
    
    play_btn = UIButton(440, 522, 60, 32, "PLAY", (40, 120, 80))
    reset_btn = UIButton(510, 522, 60, 32, "RESET", (100, 80, 40))
    
    ep_idx = 0
    playing = False
    playback_pos = 0
    manual_zoom = 15.0  # User-controlled zoom
    
    running = True
    last_upd = time.time()
    
    while running:
        now = time.time()
        mouse = pygame.mouse.get_pos()
        
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            
            # Mouse wheel zoom
            if e.type == pygame.MOUSEWHEEL:
                zoom_factor = 1.2 if e.y > 0 else 0.8
                manual_zoom *= zoom_factor
                manual_zoom = max(1.0, min(100.0, manual_zoom))
            
            ep_slider.handle_event(e, mouse)
            sp_slider.handle_event(e, mouse)
            
            if e.type == pygame.MOUSEBUTTONDOWN:
                if play_btn.clicked(mouse):
                    playing = not playing
                    play_btn.text = "PAUSE" if playing else "PLAY"
                    play_btn.color = (120, 80, 40) if playing else (40, 120, 80)
                
                if reset_btn.clicked(mouse):
                    playback_pos = 0
                
                new_ep = int(ep_slider.value)
                if new_ep != ep_idx:
                    ep_idx = new_ep
                    playback_pos = 0
        
        play_btn.check_hover(mouse)
        reset_btn.check_hover(mouse)
        
        # Get current checkpoint
        if checkpoints and ep_idx < len(checkpoints):
            ck = checkpoints[ep_idx]
            trajectory = ck.trajectory
            max_dist = ck.max_distance
            use_nn = ck.use_nn
        else:
            trajectory = [{'pos': np.array([0.0, 0.0]), 'angles': np.zeros(5), 'num_segments': 5, 'energy': 100}]
            max_dist = 0
            use_nn = True
        
        # Playback update
        spd = sp_slider.value
        interval = 0.02 / max(spd, 0.01)
        
        if playing and trajectory and (now - last_upd) >= interval:
            last_upd = now
            playback_pos = min(playback_pos + 1, len(trajectory) - 1)
        
        # Current state
        if trajectory and playback_pos < len(trajectory):
            curr = trajectory[playback_pos]
            head = curr['pos'].copy()
            curr_energy = curr['energy']
            curr_segments = curr['num_segments']
        else:
            head = np.array([0.0, 0.0])
            curr_energy = 100
            curr_segments = 5
        start = np.array([0.0, 0.0])
        
        # Auto-zoom or use manual zoom
        zoom = manual_zoom
        
        # Camera centered between head and start
        cam_x = (head[0] + 0) / 2
        cam_y = (head[1] + 0) / 2
        
        # ========== DRAWING ==========
        screen.fill((8, 20, 40))
        
        # Target direction line (GREEN)
        line_len = 800
        line_start = start - direction * line_len
        line_end = start + direction * line_len
        
        p1 = (int((line_start[0]-cam_x)*zoom+400), int((line_start[1]-cam_y)*zoom+280))
        p2 = (int((line_end[0]-cam_x)*zoom+400), int((line_end[1]-cam_y)*zoom+280))
        
        pygame.draw.line(screen, (40, 80, 60), p1, p2, 4)
        pygame.draw.line(screen, (60, 150, 100), p1, p2, 2)
        
        # Arrow
        arrow = start + direction * 40
        ax = int((arrow[0]-cam_x)*zoom+400)
        ay = int((arrow[1]-cam_y)*zoom+280)
        perp = np.array([-direction[1], direction[0]])
        sz = int(10 * zoom / 15)
        pts = [
            (ax, ay),
            (ax - int(direction[0]*sz*2) + int(perp[0]*sz), ay - int(direction[1]*sz*2) + int(perp[1]*sz)),
            (ax - int(direction[0]*sz*2) - int(perp[0]*sz), ay - int(direction[1]*sz*2) - int(perp[1]*sz))
        ]
        pygame.draw.polygon(screen, (100, 220, 150), pts)
        
        # Distance circles
        for dist in [25, 50, 100, 200]:
            cx = int((0-cam_x)*zoom+400)
            cy = int((0-cam_y)*zoom+280)
            r = int(dist * zoom)
            if 10 < r < 1500:
                surf = pygame.Surface((800, 560), pygame.SRCALPHA)
                pygame.draw.circle(surf, (255, 80, 80, max(15, 45-r//25)), (cx, cy), r, 1)
                screen.blit(surf, (0, 0))
                if r > 25:
                    screen.blit(tiny.render(f"{dist}m", True, (150, 70, 70)), (cx+r-18, cy-6))
        
        # RED CROSS at start
        cx = int((0-cam_x)*zoom+400)
        cy = int((0-cam_y)*zoom+280)
        sz = int(10 * zoom / 15)
        pygame.draw.line(screen, (255, 50, 50), (cx-sz, cy), (cx+sz, cy), 4)
        pygame.draw.line(screen, (255, 50, 50), (cx, cy-sz), (cx, cy+sz), 4)
        screen.blit(tiny.render("START", True, (255, 100, 100)), (cx+sz+5, cy-8))
        
        # RED TRAJECTORY
        if trajectory and playback_pos > 0:
            pts = []
            for i in range(0, playback_pos+1, max(1, len(trajectory)//300)):
                p = trajectory[i]['pos']
                px = int((p[0]-cam_x)*zoom+400)
                py = int((p[1]-cam_y)*zoom+280)
                pts.append((px, py))
            
            if len(pts) > 1:
                for i in range(1, len(pts)):
                    progress = i / len(pts)
                    c = (int(100+155*progress), 30, 30)
                    pygame.draw.line(screen, c, pts[i-1], pts[i], max(2, int(3*zoom/15)))
        
        # CREATURE
        if trajectory and playback_pos < len(trajectory):
            curr = trajectory[playback_pos]
            angles = curr['angles']
            pos = curr['pos']
            num_seg = curr['num_segments']
            
            joints = [pos.copy()]
            cum_angle = 0.0
            seg_len = 15.0
            for i in range(num_seg):
                if i < len(angles):
                    cum_angle += angles[i]
                prev = joints[-1]
                new_x = prev[0] - seg_len * math.cos(cum_angle)
                new_y = prev[1] - seg_len * math.sin(cum_angle)
                joints.append(np.array([new_x, new_y]))
            
            for i in range(len(joints) - 1):
                p1 = (int((joints[i][0]-cam_x)*zoom+400), int((joints[i][1]-cam_y)*zoom+280))
                p2 = (int((joints[i+1][0]-cam_x)*zoom+400), int((joints[i+1][1]-cam_y)*zoom+280))
                th = max(3, int(10 * zoom / 15))
                
                if i < len(angles):
                    c = int(abs(angles[i]) / 1.2 * 150)
                    seg_color = (c, 200-c//2, 255-c//2)
                else:
                    seg_color = (0, 200, 255)
                
                pygame.draw.line(screen, (0, 60, 100), p1, p2, th+2)
                pygame.draw.line(screen, seg_color, p1, p2, th)
                pygame.draw.circle(screen, (150, 255, 255), p1, th//2+2)
            
            # Eyes
            if len(joints) > 1:
                head_dir = joints[0] - joints[1]
                if np.linalg.norm(head_dir) > 0.01:
                    head_dir = head_dir / np.linalg.norm(head_dir)
                else:
                    head_dir = np.array([1.0, 0.0])
                
                perp = np.array([-head_dir[1], head_dir[0]])
                hs = (int((joints[0][0]-cam_x)*zoom+400), int((joints[0][1]-cam_y)*zoom+280))
                off = head_dir * int(5*zoom/15)
                spread = perp * int(4*zoom/15)
                er = max(2, int(4*zoom/15))
                pr = max(1, int(2*zoom/15))
                
                for sgn in [-1, 1]:
                    ex = int(hs[0] + off[0] + spread[0]*sgn)
                    ey = int(hs[1] + off[1] + spread[1]*sgn)
                    pygame.draw.circle(screen, (255, 255, 255), (ex, ey), er)
                    pygame.draw.circle(screen, (20, 20, 40), (ex, ey), pr)
        
        # ========== UI PANEL ==========
        pygame.draw.rect(screen, (15, 25, 45), pygame.Rect(0, 500, 800, 100))
        pygame.draw.line(screen, (50, 70, 100), (0, 500), (800, 500), 2)
        
        ep_slider.draw(screen, small)
        sp_slider.draw(screen, small)
        play_btn.draw(screen, font)
        reset_btn.draw(screen, font)
        
        # Info panel
        info_s = pygame.Surface((180, 115), pygame.SRCALPHA)
        pygame.draw.rect(info_s, (15, 25, 45, 220), info_s.get_rect(), border_radius=8)
        screen.blit(info_s, (10, 10))
        
        # Distance
        delta = head - start
        dist = np.dot(delta, direction)
        screen.blit(font.render(f"DIST: {dist:.1f}m", True, (255, 255, 255)), (20, 15))
        screen.blit(font.render(f"SEGS: {curr_segments}", True, (0, 220, 255)), (20, 38))
        
        # Method
        method = "NN" if use_nn else "Tabular"
        screen.blit(small.render(f"Method: {method}", True, (200, 200, 200)), (20, 58))
        
        # Energy bar
        max_energy = 100 * (1 + (curr_segments - 2) * 0.25)  # Approximate max energy
        e_pct = curr_energy / max(100, max_energy)
        e_color = (100, 255, 100) if e_pct > 0.5 else (255, 200, 50) if e_pct > 0.25 else (255, 80, 80)
        pygame.draw.rect(screen, (40, 40, 60), (20, 78, 140, 12), border_radius=3)
        pygame.draw.rect(screen, e_color, (20, 78, int(140*min(1, e_pct)), 12), border_radius=3)
        screen.blit(tiny.render(f"Energy: {int(curr_energy)}%", True, (255, 255, 255)), (25, 78))
        
        # Progress
        progress = playback_pos / max(1, len(trajectory)-1) if trajectory else 0
        t_curr = progress * duration
        screen.blit(tiny.render(f"Time: {t_curr:.1f}s / {duration:.0f}s", True, (180, 180, 180)), (20, 95))
        
        # Checkpoint info
        if checkpoints and ep_idx < len(checkpoints):
            ck = checkpoints[ep_idx]
            info2 = pygame.Surface((160, 55), pygame.SRCALPHA)
            pygame.draw.rect(info2, (15, 25, 45, 220), info2.get_rect(), border_radius=8)
            screen.blit(info2, (630, 10))
            screen.blit(font.render(f"Ep: {ck.episode}", True, (100, 255, 150)), (640, 15))
            screen.blit(small.render(f"Best: {ck.max_distance:.1f}m", True, (200, 200, 200)), (640, 40))
        
        # Zoom indicator
        screen.blit(tiny.render(f"Zoom: {zoom:.1f}x (scroll wheel)", True, (150, 150, 150)), (600, 575))
        
        if not playing:
            screen.blit(tiny.render("Press PLAY | Scroll to zoom", True, (100, 120, 160)), (300, 495))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()


# ==========================================================
# 8. Main
# ==========================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   AQUATIC CREATURE - Q-LEARNING")
    print("=" * 60)
    print("\nFeatures:")
    print("  - Variable number of body segments (2-10)")
    print("  - Energy cost scales with body size")
    print("  - Add/remove segments as actions")
    print("  - Zoom with mouse wheel during visualization")
    print("-" * 60)
    
    try:
        episodes = int(input("Episodes [200]: ").strip() or 200)
        duration = float(input("Episode duration (s) [10]: ").strip() or 10)
        angle = float(input("Target direction (°, 0=right, 90=up) [0]: ").strip() or 0)
        speed = float(input("Playback speed (0.01-100) [1]: ").strip() or 1)
        speed = max(0.01, min(100.0, speed))
        
        nn_choice = input("Use Neural Network? (y/n) [y]: ").strip().lower()
        use_nn = nn_choice != 'n'
    except:
        episodes, duration, angle, speed, use_nn = 200, 10.0, 0.0, 1.0, True
    
    print()
    model, checkpoints = train_with_checkpoints(
        episodes=episodes,
        checkpoint_interval=20,
        target_angle_deg=angle,
        simulation_duration=duration,
        use_neural_network=use_nn
    )
    
    print("\n" + "-" * 60)
    print("VISUALIZATION")
    print("-" * 60)
    print("Scroll wheel: Zoom in/out")
    print("Green line: Target direction")
    print("Red line: Head trajectory")
    print("Red cross: Starting point")
    print("-" * 60 + "\n")
    
    visualize(checkpoints, speed, angle, duration)

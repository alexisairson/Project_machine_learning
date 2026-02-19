import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pygame
import time
from collections import deque
import copy
import math

# ==========================================================
# 1. Neural Network (Q-Network)
# ==========================================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.network(x)


# ==========================================================
# 2. Aquatic Creature
# ==========================================================
class AquaticCreature:
    def __init__(self, num_segments=3):
        self.num_segments = num_segments
        self.angles = np.zeros(num_segments, dtype=np.float64)
        self.angular_vel = np.zeros(num_segments, dtype=np.float64)
        self.pos = np.array([0.0, 0.0], dtype=np.float64)
        self.vel = np.array([0.0, 0.0], dtype=np.float64)
        self.segment_length = 12.0
        self.water_drag = 0.15
        self.max_angle = 1.2
        
    def reset(self, x=0.0, y=0.0):
        self.angles = np.zeros(self.num_segments, dtype=np.float64)
        self.angular_vel = np.zeros(self.num_segments, dtype=np.float64)
        self.pos = np.array([x, y], dtype=np.float64)
        self.vel = np.array([0.0, 0.0], dtype=np.float64)
    
    def get_joint_positions(self):
        positions = [self.pos.copy()]
        cum_angle = 0.0
        for i in range(self.num_segments):
            cum_angle += self.angles[i]
            prev = positions[-1]
            new_x = prev[0] - self.segment_length * math.cos(cum_angle)
            new_y = prev[1] - self.segment_length * math.sin(cum_angle)
            positions.append(np.array([new_x, new_y]))
        return positions
    
    def update(self, dt):
        thrust = np.array([0.0, 0.0])
        
        for i in range(self.num_segments):
            angular_speed = abs(self.angular_vel[i])
            if angular_speed > 0.01:
                cum_angle = sum(self.angles[:i+1])
                segment_dir = np.array([math.cos(cum_angle), math.sin(cum_angle)])
                perp_dir = np.array([-segment_dir[1], segment_dir[0]])
                position_factor = (i + 1) / self.num_segments
                thrust_mag = angular_speed * 0.8 * position_factor
                thrust += thrust_mag * perp_dir * np.sign(self.angular_vel[i])
                forward_thrust = angular_speed * 0.3 * position_factor
                thrust[0] += forward_thrust
        
        self.vel += thrust * dt
        speed = np.linalg.norm(self.vel)
        if speed > 0.001:
            drag = -self.water_drag * self.vel * speed
            self.vel += drag * dt
        
        self.pos += self.vel * dt
        self.angular_vel *= 0.85
    
    def set_angle(self, segment_idx, target_angle):
        if 0 <= segment_idx < self.num_segments:
            diff = target_angle - self.angles[segment_idx]
            self.angular_vel[segment_idx] = diff * 0.5
            self.angles[segment_idx] = np.clip(target_angle, -self.max_angle, self.max_angle)
    
    def get_head_position(self):
        return self.pos.copy()
    
    def get_state(self):
        return np.concatenate([
            np.sin(self.angles),
            np.cos(self.angles),
            self.angular_vel / 0.5,
            self.vel / 10.0
        ]).astype(np.float32)


# ==========================================================
# 3. Environment
# ==========================================================
class AquaticEnv:
    def __init__(self, num_segments=3, target_angle_deg=0.0, 
                 simulation_duration=10.0, dt=0.02):
        self.num_segments = num_segments
        self.target_angle_deg = target_angle_deg
        self.target_angle_rad = math.radians(target_angle_deg)
        self.direction_vector = np.array([
            math.cos(self.target_angle_rad),
            math.sin(self.target_angle_rad)
        ])
        self.simulation_duration = simulation_duration
        self.dt = dt
        self.max_steps = int(simulation_duration / dt)
        self.max_energy = 100.0
        self.energy = self.max_energy
        self.creature = AquaticCreature(num_segments)
        self.reset()
    
    def reset(self):
        self.creature.reset()
        self.energy = self.max_energy
        self.time = 0.0
        self.step_count = 0
        self.start_pos = self.creature.get_head_position().copy()
        self.trajectory = []  # Store trajectory
        return self.get_state()
    
    def get_state_dim(self):
        # sin(5) + cos(5) + angular_vel(5) + vel(2) = 17
        return self.num_segments * 3 + 2
    
    def get_action_dim(self):
        return self.num_segments * 3  # swing_up, swing_down, hold per segment
    
    def get_state(self):
        return self.creature.get_state()
    
    def get_distance(self, pos):
        delta = pos - self.start_pos
        return np.dot(delta, self.direction_vector)
    
    def step(self, action):
        segment_idx = action // 3
        action_type = action % 3
        energy_cost = 0.0
        
        if segment_idx < self.num_segments:
            current_angle = self.creature.angles[segment_idx]
            if action_type == 0:
                self.creature.set_angle(segment_idx, current_angle + 0.15)
                energy_cost = 0.05
            elif action_type == 1:
                self.creature.set_angle(segment_idx, current_angle - 0.15)
                energy_cost = 0.05
        
        for _ in range(3):
            self.creature.update(self.dt / 3)
        
        self.energy -= energy_cost
        self.energy = max(0, min(self.max_energy, self.energy + 0.01))
        
        head_pos = self.creature.get_head_position()
        distance = self.get_distance(head_pos)
        
        # Store trajectory
        self.trajectory.append({
            'pos': head_pos.copy(),
            'angles': self.creature.angles.copy(),
            'vel': self.creature.vel.copy()
        })
        
        reward = distance * 0.5 - energy_cost * 0.1
        
        self.step_count += 1
        self.time += self.dt
        done = self.step_count >= self.max_steps or self.energy <= 0
        
        return self.get_state(), float(-reward), done, {'distance': distance}


# ==========================================================
# 4. Training with Trajectory Storage
# ==========================================================
class TrainingCheckpoint:
    def __init__(self, episode, model_state, trajectory, max_distance):
        self.episode = episode
        self.model_state = copy.deepcopy(model_state)
        self.trajectory = copy.deepcopy(trajectory)  # Store best trajectory
        self.max_distance = max_distance


def train_with_checkpoints(episodes=200, checkpoint_interval=20,
                          target_angle_deg=0.0, simulation_duration=10.0):
    env = AquaticEnv(
        num_segments=3,
        target_angle_deg=target_angle_deg,
        simulation_duration=simulation_duration
    )
    
    model = QNetwork(env.get_state_dim(), env.get_action_dim())
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    
    checkpoints = []
    best_distance = -float('inf')
    best_trajectory = []
    
    print("=" * 60)
    print("TRAINING AQUATIC CREATURE")
    print("=" * 60)
    print(f"Episodes: {episodes}")
    print(f"Duration: {simulation_duration}s per episode")
    print(f"Target: {target_angle_deg}° (0°=right, 90°=up, 180°=left)")
    print(f"Segments: 5 (fixed)")
    print("-" * 60)
    
    for ep in range(episodes + 1):
        state = env.reset()
        epsilon = max(0.05, 1.0 - (ep / episodes) * 0.95)
        max_distance = 0
        
        for step in range(env.max_steps):
            if random.random() < epsilon:
                action = random.randint(0, env.get_action_dim() - 1)
            else:
                with torch.no_grad():
                    q_values = model(torch.from_numpy(state))
                    action = torch.argmin(q_values).item()
            
            next_state, cost, done, info = env.step(action)
            max_distance = max(max_distance, info['distance'])
            
            with torch.no_grad():
                q_next = model(torch.from_numpy(next_state))
                target = cost + 0.95 * torch.min(q_next).item()
            
            current_q = model(torch.from_numpy(state))[action]
            loss = nn.MSELoss()(current_q, torch.tensor(target, dtype=torch.float32))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state
            if done:
                break
        
        # Track best trajectory
        if max_distance > best_distance:
            best_distance = max_distance
            best_trajectory = copy.deepcopy(env.trajectory)
        
        if ep % checkpoint_interval == 0:
            checkpoint = TrainingCheckpoint(
                episode=ep,
                model_state=model.state_dict(),
                trajectory=copy.deepcopy(best_trajectory),
                max_distance=best_distance
            )
            checkpoints.append(checkpoint)
            print(f"Ep {ep:3d} | Max dist: {max_distance:6.1f}m | Best: {best_distance:6.1f}m")
    
    print("-" * 60)
    print(f"Done! {len(checkpoints)} checkpoints saved.")
    
    return model, checkpoints


# ==========================================================
# 5. UI Components
# ==========================================================
class UIIntInputBox:
    """Input box for integer values only"""
    def __init__(self, x, y, w, h, val=0):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = str(int(val))
        self.active = False
        self.font = pygame.font.SysFont("Arial", 14)
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                self.active = False
                return True
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.unicode.isdigit():
                self.text = self.text[:5] + event.unicode
        return False
    
    def get_value(self):
        try:
            return int(self.text) if self.text else 0
        except:
            return 0
    
    def set_value(self, v):
        self.text = str(int(v))
    
    def draw(self, screen):
        c = (100, 150, 255) if self.active else (80, 80, 100)
        pygame.draw.rect(screen, (40, 40, 60), self.rect, border_radius=4)
        pygame.draw.rect(screen, c, self.rect, 2, border_radius=4)
        s = self.font.render(self.text, True, (255, 255, 255))
        screen.blit(s, s.get_rect(center=self.rect.center))


class UIIntSlider:
    """Slider for integer values only"""
    def __init__(self, x, y, w, h, min_v, max_v, val, label=""):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_v = int(min_v)
        self.max_v = int(max_v)
        self.value = int(val)
        self.label = label
        self.dragging = False
        self.input = UIIntInputBox(x + w + 8, y - 5, 40, 25, int(val))
    
    def get_knob_x(self):
        if self.max_v <= self.min_v:
            return self.rect.x
        r = (self.value - self.min_v) / (self.max_v - self.min_v)
        return int(self.rect.x + r * self.rect.width)
    
    def _set_from_x(self, x):
        r = max(0, min(1, (x - self.rect.x) / self.rect.width))
        self.value = int(self.min_v + r * (self.max_v - self.min_v))
        self.input.set_value(self.value)
    
    def handle_event(self, event, pos):
        if event.type == pygame.MOUSEBUTTONDOWN:
            kx = self.get_knob_x()
            if abs(pos[0] - kx) < 12 and abs(pos[1] - self.rect.centery) < 15:
                self.dragging = True
            elif self.rect.collidepoint(pos):
                self._set_from_x(pos[0])
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._set_from_x(pos[0])
        if self.input.handle_event(event):
            self.value = max(self.min_v, min(self.max_v, self.input.get_value()))
            self.input.set_value(self.value)
    
    def draw(self, screen, font):
        screen.blit(font.render(self.label, True, (200, 200, 200)), (self.rect.x, self.rect.y - 18))
        pygame.draw.rect(screen, (60, 60, 80), 
                        pygame.Rect(self.rect.x, self.rect.centery - 2, self.rect.width, 4), border_radius=2)
        kx = self.get_knob_x()
        pygame.draw.rect(screen, (100, 180, 255),
                        pygame.Rect(self.rect.x, self.rect.centery - 2, kx - self.rect.x, 4), border_radius=2)
        pygame.draw.circle(screen, (120, 200, 255), (kx, self.rect.centery), 10)
        pygame.draw.circle(screen, (255, 255, 255), (kx, self.rect.centery), 7)
        self.input.draw(screen)


class UIFloatSlider:
    """Slider for float values with input box"""
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
                (math.log10(self.max_v) - math.log10(self.min_v))
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
            kx = self.get_knob_x()
            if abs(pos[0] - kx) < 12 and abs(pos[1] - self.rect.centery) < 15:
                self.dragging = True
            elif self.rect.collidepoint(pos):
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
        # Show value
        val_text = f"{self.value:.1f}" if self.value < 10 else f"{self.value:.0f}"
        screen.blit(font.render(val_text, True, (200, 200, 200)), (self.rect.x + self.rect.width + 8, self.rect.y - 2))


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
# 6. Visualization
# ==========================================================
def visualize(model, checkpoints, speed=1.0, target_angle=0.0, duration=10.0):
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Aquatic Evolution - Best Trajectories")
    
    font = pygame.font.SysFont("Arial", 18, bold=True)
    small = pygame.font.SysFont("Arial", 14)
    tiny = pygame.font.SysFont("Arial", 12)
    
    clock = pygame.time.Clock()
    
    target_rad = math.radians(target_angle)
    direction = np.array([math.cos(target_rad), math.sin(target_rad)])
    
    # UI
    ep_slider = UIIntSlider(50, 530, 130, 18, 0, max(0, len(checkpoints)-1), 0, "Episode")
    sp_slider = UIFloatSlider(280, 530, 130, 18, 0.01, 100.0, speed, "Speed", log=True)
    
    play_btn = UIButton(490, 522, 65, 32, "PLAY", (40, 120, 80))
    reset_btn = UIButton(565, 522, 65, 32, "RESET", (100, 80, 40))
    
    # State
    ep_idx = 0
    playing = False
    playback_pos = 0  # Current position in trajectory playback
    
    running = True
    last_upd = time.time()
    
    while running:
        now = time.time()
        mouse = pygame.mouse.get_pos()
        
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            
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
        else:
            trajectory = []
            max_dist = 0
        
        # Playback update
        spd = sp_slider.value
        interval = 0.03 / max(spd, 0.01)
        
        if playing and trajectory and (now - last_upd) >= interval:
            last_upd = now
            playback_pos = min(playback_pos + 1, len(trajectory) - 1)
        
        # Camera
        if trajectory and playback_pos < len(trajectory):
            curr = trajectory[playback_pos]
            head = curr['pos']
        else:
            head = np.array([0.0, 0.0])
        start = np.array([0.0, 0.0])
        
        # Calculate zoom to fit full trajectory
        if trajectory:
            all_x = [p['pos'][0] for p in trajectory[:playback_pos+1]] + [0]
            all_y = [p['pos'][1] for p in trajectory[:playback_pos+1]] + [0]
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            
            margin = 100
            w = max(max_x - min_x, 20) + margin * 2
            h = max(max_y - min_y, 20) + margin * 2
            
            zoom = min(800 / w, 560 / h, 50.0)
            zoom = max(zoom, 3.0)
        else:
            zoom = 15.0
        
        cam_x = (head[0] + start[0]) / 2
        cam_y = (head[1] + start[1]) / 2
        
        # Draw
        screen.fill((8, 20, 40))
        
        # Direction line
        line_len = 500
        p1 = start - direction * line_len
        p2 = start + direction * line_len
        pygame.draw.line(screen, (50, 60, 80),
                        (int((p1[0]-cam_x)*zoom+400), int((p1[1]-cam_y)*zoom+280)),
                        (int((p2[0]-cam_x)*zoom+400), int((p2[1]-cam_y)*zoom+280)), 1)
        
        # Arrow
        arrow = start + direction * 25
        ax = int((arrow[0]-cam_x)*zoom+400)
        ay = int((arrow[1]-cam_y)*zoom+280)
        perp = np.array([-direction[1], direction[0]])
        sz = int(6 * zoom / 15)
        pts = [(ax, ay),
               (ax - int(direction[0]*sz*2) + int(perp[0]*sz), ay - int(direction[1]*sz*2) + int(perp[1]*sz)),
               (ax - int(direction[0]*sz*2) - int(perp[0]*sz), ay - int(direction[1]*sz*2) - int(perp[1]*sz))]
        pygame.draw.polygon(screen, (80, 120, 180), pts)
        
        # Circles
        for dist in [25, 50, 100, 200]:
            cx = int((start[0]-cam_x)*zoom+400)
            cy = int((start[1]-cam_y)*zoom+280)
            r = int(dist * zoom)
            if 10 < r < 800:
                surf = pygame.Surface((800, 560), pygame.SRCALPHA)
                pygame.draw.circle(surf, (255, 80, 80, max(15, 50-r//30)), (cx, cy), r, 1)
                screen.blit(surf, (0, 0))
                if r > 30:
                    screen.blit(tiny.render(f"{dist}m", True, (150, 70, 70)), (cx+r-18, cy-6))
        
        # Cross at start
        cx = int((start[0]-cam_x)*zoom+400)
        cy = int((start[1]-cam_y)*zoom+280)
        sz = int(8 * zoom / 15)
        pygame.draw.line(screen, (255, 50, 50), (cx-sz, cy), (cx+sz, cy), 3)
        pygame.draw.line(screen, (255, 50, 50), (cx, cy-sz), (cx, cy+sz), 3)
        
        # Trajectory (red line showing path)
        if trajectory and playback_pos > 0:
            pts = [(int((trajectory[i]['pos'][0]-cam_x)*zoom+400), 
                   int((trajectory[i]['pos'][1]-cam_y)*zoom+280)) 
                  for i in range(0, playback_pos+1, max(1, len(trajectory)//500))]
            for i in range(1, len(pts)):
                progress = i / len(pts)
                c = (int(80+175*progress), 30, 30)
                pygame.draw.line(screen, c, pts[i-1], pts[i], max(1, int(2*zoom/15)))
        
        # Creature at current playback position
        if trajectory and playback_pos < len(trajectory):
            curr = trajectory[playback_pos]
            angles = curr['angles']
            pos = curr['pos']
            
            # Calculate joint positions from angles
            joints = [pos.copy()]
            cum_angle = 0.0
            seg_len = 12.0
            for angle in angles:
                cum_angle += angle
                prev = joints[-1]
                new_x = prev[0] - seg_len * math.cos(cum_angle)
                new_y = prev[1] - seg_len * math.sin(cum_angle)
                joints.append(np.array([new_x, new_y]))
            
            # Draw segments
            for i in range(len(joints) - 1):
                p1 = (int((joints[i][0]-cam_x)*zoom+400), int((joints[i][1]-cam_y)*zoom+280))
                p2 = (int((joints[i+1][0]-cam_x)*zoom+400), int((joints[i+1][1]-cam_y)*zoom+280))
                th = max(3, int(8 * zoom / 15))
                pygame.draw.line(screen, (0, 60, 100), p1, p2, th+2)
                pygame.draw.line(screen, (0, 200, 255), p1, p2, th)
                pygame.draw.circle(screen, (150, 255, 255), p1, th//2+1)
            
            # Eyes
            if len(joints) > 1:
                head_dir = joints[0] - joints[1]
                if np.linalg.norm(head_dir) > 0.01:
                    head_dir = head_dir / np.linalg.norm(head_dir)
                else:
                    head_dir = np.array([1.0, 0.0])
                
                perp = np.array([-head_dir[1], head_dir[0]])
                hs = (int((joints[0][0]-cam_x)*zoom+400), int((joints[0][1]-cam_y)*zoom+280))
                off = head_dir * int(4*zoom/15)
                spread = perp * int(3*zoom/15)
                er = max(2, int(3*zoom/15))
                pr = max(1, int(1.5*zoom/15))
                
                for sgn in [-1, 1]:
                    ex = int(hs[0] + off[0] + spread[0]*sgn)
                    ey = int(hs[1] + off[1] + spread[1]*sgn)
                    pygame.draw.circle(screen, (255, 255, 255), (ex, ey), er)
                    pygame.draw.circle(screen, (20, 20, 40), (ex, ey), pr)
        
        # UI Panel
        pygame.draw.rect(screen, (15, 25, 45), pygame.Rect(0, 500, 800, 100))
        pygame.draw.line(screen, (50, 70, 100), (0, 500), (800, 500), 2)
        
        ep_slider.draw(screen, small)
        sp_slider.draw(screen, small)
        play_btn.draw(screen, font)
        reset_btn.draw(screen, font)
        
        # Info panel
        info_s = pygame.Surface((180, 85), pygame.SRCALPHA)
        pygame.draw.rect(info_s, (15, 25, 45, 220), info_s.get_rect(), border_radius=8)
        screen.blit(info_s, (10, 10))
        
        # Distance
        if trajectory and playback_pos < len(trajectory):
            delta = trajectory[playback_pos]['pos'] - start
            dist = np.dot(delta, direction)
        else:
            dist = 0
        screen.blit(font.render(f"DIST: {dist:.1f}m", True, (255, 255, 255)), (20, 15))
        
        # Progress bar
        progress = playback_pos / max(1, len(trajectory)-1) if trajectory else 0
        pygame.draw.rect(screen, (40, 40, 60), (20, 40, 140, 12), border_radius=3)
        pygame.draw.rect(screen, (100, 200, 255), (20, 40, int(140*progress), 12), border_radius=3)
        screen.blit(tiny.render(f"Progress: {int(progress*100)}%", True, (255, 255, 255)), (25, 40))
        
        # Time
        t_curr = progress * duration
        screen.blit(tiny.render(f"Time: {t_curr:.1f}s / {duration:.0f}s", True, (180, 180, 180)), (20, 58))
        
        # Checkpoint info
        if checkpoints and ep_idx < len(checkpoints):
            ck = checkpoints[ep_idx]
            info2 = pygame.Surface((160, 50), pygame.SRCALPHA)
            pygame.draw.rect(info2, (15, 25, 45, 220), info2.get_rect(), border_radius=8)
            screen.blit(info2, (630, 10))
            screen.blit(font.render(f"Ep: {ck.episode}", True, (100, 255, 150)), (640, 15))
            screen.blit(small.render(f"Best: {ck.max_distance:.1f}m", True, (200, 200, 200)), (640, 38))
        
        if not playing:
            screen.blit(tiny.render("Select episode/speed, press PLAY to watch trajectory", True, (100, 120, 160)), (220, 495))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()


# ==========================================================
# 7. Main
# ==========================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   AQUATIC CREATURE - Q-LEARNING")
    print("=" * 60)
    print("\nCreature: 5 segments (fixed)")
    print("Movement: Segment oscillation creates thrust")
    print("Visualization: Shows best trajectory from training")
    print("-" * 60)
    
    try:
        episodes = int(input("Episodes [200]: ").strip() or 200)
        duration = float(input("Episode duration (s) [10]: ").strip() or 10)
        angle = float(input("Target direction (°, 0=right, 90=up) [0]: ").strip() or 0)
        speed = float(input("Playback speed (0.01-100) [1]: ").strip() or 1)
        speed = max(0.01, min(100.0, speed))
    except:
        episodes, duration, angle, speed = 200, 10.0, 0.0, 1.0
    
    print()
    model, checkpoints = train_with_checkpoints(
        episodes=episodes,
        checkpoint_interval=20,
        target_angle_deg=angle,
        simulation_duration=duration
    )
    
    print("\n" + "-" * 60)
    print("VISUALIZATION - Shows best trajectory from training")
    print("-" * 60)
    print("Episode slider: Select checkpoint (integer values)")
    print("Speed slider: Playback speed (0.01-100)")
    print("PLAY: Watch the trajectory")
    print("RESET: Restart from beginning")
    print("-" * 60 + "\n")
    
    visualize(model, checkpoints, speed, angle, duration)

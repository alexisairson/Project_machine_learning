import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pygame
import time
from collections import deque
import copy

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
# 2. Environment with Dynamic Segments
# ==========================================================
class AquaticEnv:
    def __init__(self, num_segments=5, max_segments=10, min_segments=2):
        self.initial_num_segments = num_segments
        self.num_segments = num_segments
        self.max_segments = max_segments
        self.min_segments = min_segments
        self.reset()

    def reset(self):
        self.num_segments = self.initial_num_segments
        self.angles = np.zeros(self.num_segments, dtype=np.float32)
        self.pos = np.array([0.0, 0.0], dtype=np.float32)
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.time = 0
        self.energy = 100.0
        self.max_energy = 100.0
        self.segment_changed = None  # Track what changed
        return self.get_state()

    def get_state_dim(self):
        """State: angles (sin/cos) + velocity magnitude + energy + num_segments normalized"""
        return self.max_segments * 2 + 3

    def get_action_dim(self):
        """Actions: 2 per segment (move +/-) + add_segment + remove_segment"""
        return self.max_segments * 2 + 2

    def get_state(self):
        """Create fixed-size state vector"""
        # Angles encoded as sin/cos, padded for max_segments
        angle_sin = np.zeros(self.max_segments, dtype=np.float32)
        angle_cos = np.zeros(self.max_segments, dtype=np.float32)
        for i in range(self.num_segments):
            angle_sin[i] = np.sin(self.angles[i])
            angle_cos[i] = np.cos(self.angles[i])
        
        state = np.concatenate([
            angle_sin,
            angle_cos,
            [np.linalg.norm(self.velocity) / 10.0],  # Normalized velocity
            [self.energy / self.max_energy],  # Normalized energy
            [self.num_segments / self.max_segments]  # Normalized segment count
        ]).astype(np.float32)
        return state

    def step(self, action):
        """Execute action and return new state, cost, done, info"""
        old_angles = self.angles.copy()
        old_segments = self.num_segments
        self.segment_changed = None
        
        energy_cost = 0.0
        add_segment_action = self.max_segments * 2
        remove_segment_action = self.max_segments * 2 + 1
        
        # Interpret action
        if action < self.max_segments * 2:
            # Move segment action
            seg_idx = action // 2
            direction = 1 if action % 2 == 0 else -1
            
            if seg_idx < self.num_segments:
                self.angles[seg_idx] += direction * 0.3
                self.angles[seg_idx] = np.clip(self.angles[seg_idx], -1.0, 1.0)
                energy_cost = 0.3
        
        elif action == add_segment_action:
            # Add segment action
            if self.num_segments < self.max_segments:
                self.num_segments += 1
                self.angles = np.append(self.angles, 0.0)
                self.segment_changed = ("ADD", self.num_segments - 1)
                energy_cost = 2.0  # Higher cost for structural change
        
        elif action == remove_segment_action:
            # Remove segment action
            if self.num_segments > self.min_segments:
                self.num_segments -= 1
                self.angles = self.angles[:-1]
                self.segment_changed = ("REMOVE", self.num_segments)
                energy_cost = 1.0
        
        # Physics update
        thrust_mag = np.sum(np.abs(self.angles - old_angles)) if old_segments == self.num_segments else 0
        direction = np.mean(self.angles) if len(self.angles) > 0 else 0
        
        thrust = np.array([np.cos(direction), np.sin(direction)]) * thrust_mag * 0.5
        self.velocity = self.velocity * 0.92 + thrust
        self.pos += self.velocity
        
        # Energy update
        self.energy -= energy_cost
        self.energy = max(0, min(self.max_energy, self.energy + 0.05))  # Small recovery
        
        # Cost function: balance distance, energy, and complexity
        cost = 0.1 * thrust_mag - self.velocity[0] * 0.5 + energy_cost * 0.1
        
        # Bonus for efficient swimming
        if self.velocity[0] > 0.1:
            cost -= 0.1  # Reward forward movement
        
        self.time += 1
        done = self.time > 400 or self.energy <= 0
        
        info = {
            'thrust': thrust_mag,
            'segment_changed': self.segment_changed,
            'num_segments': self.num_segments
        }
        
        return self.get_state(), float(cost), done, info


# ==========================================================
# 3. Training with Checkpoints
# ==========================================================
class TrainingCheckpoint:
    """Stores a training checkpoint for later visualization"""
    def __init__(self, episode, model_state, trajectory, final_distance, num_segments, energy):
        self.episode = episode
        self.model_state = copy.deepcopy(model_state)
        self.trajectory = list(trajectory)  # Copy trajectory
        self.final_distance = final_distance
        self.num_segments = num_segments
        self.energy = energy


def train_with_checkpoints(episodes=200, checkpoint_interval=20):
    """Train the model and save checkpoints every N episodes"""
    env = AquaticEnv(num_segments=5)
    model = QNetwork(env.get_state_dim(), env.get_action_dim())
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    checkpoints = []
    best_distance = -float('inf')
    
    print("=" * 60)
    print("TRAINING AQUATIC CREATURE WITH Q-LEARNING")
    print("=" * 60)
    print(f"Episodes: {episodes}, Checkpoint interval: {checkpoint_interval}")
    print("-" * 60)
    
    for ep in range(episodes + 1):
        state = env.reset()
        epsilon = max(0.05, 1.0 - (ep / episodes) * 0.95)
        trajectory = []
        
        for step in range(400):
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, env.get_action_dim() - 1)
            else:
                with torch.no_grad():
                    q_values = model(torch.from_numpy(state))
                    action = torch.argmin(q_values).item()
            
            next_state, cost, done, info = env.step(action)
            
            # Store position for trajectory
            trajectory.append(env.pos.copy())
            
            # Q-learning update (cost minimization)
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
        
        final_distance = env.pos[0]
        if final_distance > best_distance:
            best_distance = final_distance
        
        # Save checkpoint every checkpoint_interval episodes
        if ep % checkpoint_interval == 0:
            checkpoint = TrainingCheckpoint(
                episode=ep,
                model_state=model.state_dict(),
                trajectory=trajectory,
                final_distance=final_distance,
                num_segments=env.num_segments,
                energy=env.energy
            )
            checkpoints.append(checkpoint)
            print(f"Episode {ep:3d} | Distance: {final_distance:7.2f}m | Segments: {env.num_segments} | Best: {best_distance:7.2f}m")
    
    print("-" * 60)
    print(f"Training complete! {len(checkpoints)} checkpoints saved.")
    print(f"Best distance achieved: {best_distance:.2f}m")
    
    return model, checkpoints


# ==========================================================
# 4. Visualization with UI Controls
# ==========================================================
class UIButton:
    """Simple UI button"""
    def __init__(self, x, y, width, height, text, color=(60, 60, 80)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = (80, 80, 100)
        self.is_hovered = False
    
    def draw(self, screen, font):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        pygame.draw.rect(screen, (150, 150, 170), self.rect, 2, border_radius=8)
        
        text_surf = font.render(self.text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)
    
    def check_hover(self, pos):
        self.is_hovered = self.rect.collidepoint(pos)
        return self.is_hovered
    
    def check_click(self, pos):
        return self.rect.collidepoint(pos)


class UISlider:
    """Simple UI slider for episode selection"""
    def __init__(self, x, y, width, height, min_val, max_val, initial_val=0):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.dragging = False
        self.knob_radius = 12
    
    def get_knob_x(self):
        ratio = (self.value - self.min_val) / max(1, self.max_val - self.min_val)
        return int(self.rect.x + ratio * self.rect.width)
    
    def draw(self, screen, font):
        # Track
        track_rect = pygame.Rect(self.rect.x, self.rect.centery - 3, self.rect.width, 6)
        pygame.draw.rect(screen, (60, 60, 80), track_rect, border_radius=3)
        
        # Filled portion
        knob_x = self.get_knob_x()
        filled_rect = pygame.Rect(self.rect.x, self.rect.centery - 3, knob_x - self.rect.x, 6)
        pygame.draw.rect(screen, (100, 150, 255), filled_rect, border_radius=3)
        
        # Knob
        pygame.draw.circle(screen, (150, 200, 255), (knob_x, self.rect.centery), self.knob_radius)
        pygame.draw.circle(screen, (255, 255, 255), (knob_x, self.rect.centery), self.knob_radius - 3)
        
        # Value label
        label = font.render(f"Episode: {int(self.value)}", True, (255, 255, 255))
        screen.blit(label, (self.rect.x, self.rect.y - 25))
    
    def handle_event(self, event, pos):
        if event.type == pygame.MOUSEBUTTONDOWN:
            knob_x = self.get_knob_x()
            knob_rect = pygame.Rect(knob_x - self.knob_radius, self.rect.centery - self.knob_radius,
                                   self.knob_radius * 2, self.knob_radius * 2)
            if knob_rect.collidepoint(pos) or self.rect.collidepoint(pos):
                self.dragging = True
                self._update_value(pos[0])
        
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._update_value(pos[0])
    
    def _update_value(self, x):
        ratio = (x - self.rect.x) / self.rect.width
        ratio = max(0, min(1, ratio))
        self.value = self.min_val + ratio * (self.max_val - self.min_val)


class NotificationFrame:
    """Display notifications for segment changes"""
    def __init__(self):
        self.messages = []
        self.duration = 4.0
    
    def add_message(self, text, color=(255, 255, 100)):
        self.messages.append((text, time.time(), color))
    
    def update(self):
        current_time = time.time()
        self.messages = [(m, t, c) for m, t, c in self.messages if current_time - t < self.duration]
    
    def draw(self, screen, font, y_start=120):
        y_offset = y_start
        for message, timestamp, color in self.messages:
            alpha = max(0, min(255, int(255 * (1 - (time.time() - timestamp) / self.duration))))
            
            text_surface = font.render(message, True, color)
            padding = 10
            bg_rect = text_surface.get_rect()
            bg_rect.topleft = (20, y_offset)
            bg_rect.inflate_ip(padding * 2, padding * 2)
            
            bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
            pygame.draw.rect(bg_surface, (30, 30, 50, int(alpha * 0.9)), bg_surface.get_rect(), border_radius=6)
            pygame.draw.rect(bg_surface, (*color[:3], int(alpha * 0.6)), bg_surface.get_rect(), width=2, border_radius=6)
            screen.blit(bg_surface, bg_rect)
            
            text_rect = text_surface.get_rect()
            text_rect.center = bg_rect.center
            screen.blit(text_surface, text_rect)
            
            y_offset += 45


class Bubble:
    """Simple bubble effect"""
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.vx = random.uniform(-0.3, 0.3)
        self.vy = random.uniform(-0.8, -0.2)
        self.life = 200
    
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 4
    
    def draw(self, screen, cam_x, cam_y, zoom):
        alpha = max(0, min(255, int(self.life)))
        radius = max(1, int(3 * zoom / 15))
        
        s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        pygame.draw.circle(s, (180, 220, 255, alpha), (radius, radius), radius)
        
        pos = (int((self.x - cam_x) * zoom + 400), int((self.y - cam_y) * zoom + 300))
        screen.blit(s, (pos[0]-radius, pos[1]-radius))


def visualize_with_ui(model, checkpoints, playback_speed=0.1):
    """
    Visualization with UI controls:
    - Episode slider to select checkpoint
    - Play/pause button
    - Creature centered with full trajectory visible
    - Very slow playback
    """
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Aquatic Evolution - Q-Learning Visualization")
    
    # Fonts
    font = pygame.font.SysFont("Arial", 22, bold=True)
    small_font = pygame.font.SysFont("Arial", 16)
    title_font = pygame.font.SysFont("Arial", 24, bold=True)
    
    clock = pygame.time.Clock()
    
    # UI Elements
    if checkpoints:
        slider = UISlider(200, 550, 400, 30, 0, len(checkpoints)-1, 0)
    else:
        slider = UISlider(200, 550, 400, 30, 0, 1, 0)
    
    play_button = UIButton(650, 540, 100, 40, "PLAY", (40, 120, 80))
    
    notifications = NotificationFrame()
    
    # State
    is_playing = False
    current_checkpoint_idx = 0
    env = AquaticEnv(num_segments=5)
    
    # Load initial checkpoint
    if checkpoints:
        ck = checkpoints[0]
        model.load_state_dict(ck.model_state)
        env.initial_num_segments = 5
        env.reset()
    
    state = env.reset()
    trajectory = []
    bubbles = []
    zoom = 10.0
    
    # Timing
    last_update_time = time.time()
    update_interval = 1.0 / playback_speed  # Seconds between simulation steps
    
    running = True
    
    while running:
        current_time = time.time()
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Handle slider
            slider.handle_event(event, mouse_pos)
            
            # Handle play button
            if event.type == pygame.MOUSEBUTTONDOWN:
                if play_button.check_click(mouse_pos):
                    is_playing = not is_playing
                    play_button.text = "PAUSE" if is_playing else "PLAY"
                    play_button.color = (120, 80, 40) if is_playing else (40, 120, 80)
                
                # Check if slider changed
                new_idx = int(slider.value)
                if new_idx != current_checkpoint_idx and checkpoints:
                    current_checkpoint_idx = new_idx
                    # Load checkpoint
                    ck = checkpoints[new_idx]
                    model.load_state_dict(ck.model_state)
                    # Reset environment with checkpoint's segment count
                    env.initial_num_segments = 5  # Reset to initial
                    state = env.reset()
                    trajectory = []
                    bubbles = []
                    notifications.add_message(f"Loaded Episode {ck.episode}", (100, 200, 255))
        
        # Update hover states
        play_button.check_hover(mouse_pos)
        
        # Check if slider value changed
        new_idx = int(slider.value)
        if new_idx != current_checkpoint_idx and checkpoints:
            current_checkpoint_idx = new_idx
            ck = checkpoints[new_idx]
            model.load_state_dict(ck.model_state)
            env.initial_num_segments = 5
            state = env.reset()
            trajectory = []
            bubbles = []
            notifications.add_message(f"Loaded Episode {ck.episode}", (100, 200, 255))
        
        # Simulation update (only when playing and enough time has passed)
        if is_playing and (current_time - last_update_time) >= update_interval:
            last_update_time = current_time
            
            with torch.no_grad():
                action = torch.argmin(model(torch.from_numpy(state))).item()
            
            state, _, done, info = env.step(action)
            
            # Store trajectory point
            trajectory.append(env.pos.copy())
            
            # Create bubbles on movement
            if info.get('thrust', 0) > 0.03:
                px_x, px_y = env.pos[0] * 10, env.pos[1] * 10
                bubbles.append(Bubble(px_x, px_y))
            
            # Handle segment changes
            if info.get('segment_changed'):
                event_type, seg_idx = info['segment_changed']
                if event_type == "ADD":
                    notifications.add_message(f"MEMBER #{seg_idx+1} ADDED!", (100, 255, 100))
                else:
                    notifications.add_message(f"MEMBER #{seg_idx+1} DISCARDED!", (255, 100, 100))
            
            if done:
                state = env.reset()
                trajectory = []
                notifications.add_message("Episode Complete - Resetting", (200, 200, 200))
        
        # Update bubbles
        for b in bubbles[:]:
            b.update()
            if b.life <= 0:
                bubbles.remove(b)
        
        # Update notifications
        notifications.update()
        
        # Calculate camera and zoom
        px_x, px_y = env.pos[0] * 10, env.pos[1] * 10
        
        # Calculate zoom to show full trajectory
        if trajectory:
            traj_x = [p[0] * 10 for p in trajectory]
            traj_y = [p[1] * 10 for p in trajectory]
            min_x, max_x = min(traj_x + [0]), max(traj_x + [0])  # Include start point
            min_y, max_y = min(traj_y), max(traj_y)
            
            # Add margin
            margin = 150
            width_needed = (max_x - min_x) + margin * 2
            height_needed = (max_y - min_y) + margin * 2
            
            # Calculate zoom to fit trajectory
            zoom_x = 800 / max(width_needed, 100)
            zoom_y = 600 / max(height_needed, 100)
            target_zoom = min(zoom_x, zoom_y, 20.0)
            target_zoom = max(target_zoom, 2.0)  # Minimum zoom
        else:
            target_zoom = 15.0
        
        zoom += (target_zoom - zoom) * 0.05
        
        # Camera centered on creature
        cam_x = px_x
        cam_y = px_y
        
        # Calculate offset to center the trajectory view
        if trajectory:
            # Center between creature and start point (weighted towards showing both)
            start_x = 0
            mid_x = (px_x + start_x) / 2
            # Adjust camera to center the view
            cam_x = mid_x
        
        # Drawing
        screen.fill((8, 25, 50))  # Deep underwater blue
        
        # Draw grid lines for reference
        grid_spacing = int(50 * zoom)
        if grid_spacing > 20:
            for i in range(-20, 20):
                gx = int((i * 50 - cam_x) * zoom + 400)
                if 0 < gx < 800:
                    pygame.draw.line(screen, (20, 40, 70), (gx, 0), (gx, 600), 1)
                gy = int((i * 50 - cam_y) * zoom + 300)
                if 0 < gy < 600:
                    pygame.draw.line(screen, (20, 40, 70), (0, gy), (800, gy), 1)
        
        # Draw start line (red)
        line_x = int((0 - cam_x) * zoom + 400)
        if -10 < line_x < 810:
            pygame.draw.line(screen, (255, 60, 60), (line_x, 0), (line_x, 600), 4)
            # Label
            start_label = small_font.render("START", True, (255, 100, 100))
            screen.blit(start_label, (line_x - 25, 560))
        
        # Draw trajectory (red line)
        if len(trajectory) > 1:
            points = []
            for p in trajectory:
                tx, ty = p[0] * 10, p[1] * 10
                screen_x = int((tx - cam_x) * zoom + 400)
                screen_y = int((ty - cam_y) * zoom + 300)
                points.append((screen_x, screen_y))
            
            # Draw trajectory with gradient
            for i in range(1, len(points)):
                # Color gradient: older = darker
                progress = i / len(points)
                r = int(150 + 105 * progress)
                g = int(30 + 20 * progress)
                b = int(30 + 20 * progress)
                thickness = max(1, int(3 * zoom / 15))
                pygame.draw.line(screen, (r, g, b), points[i-1], points[i], thickness)
        
        # Draw bubbles
        for b in bubbles:
            b.draw(screen, cam_x, cam_y, zoom)
        
        # Draw creature
        curr_x, curr_y = px_x, px_y
        cum_angle = 0
        segment_positions = [(curr_x, curr_y)]
        
        for i in range(env.num_segments):
            seg_len = 15
            next_x = curr_x + np.cos(cum_angle) * seg_len
            next_y = curr_y + np.sin(cum_angle) * seg_len
            segment_positions.append((next_x, next_y))
            
            p1 = (int((curr_x - cam_x) * zoom + 400), int((curr_y - cam_y) * zoom + 300))
            p2 = (int((next_x - cam_x) * zoom + 400), int((next_y - cam_y) * zoom + 300))
            
            thickness = int(max(3, 10 * zoom / 15))
            
            # Draw segment with glow effect
            pygame.draw.line(screen, (0, 150, 200), p1, p2, thickness + 2)
            pygame.draw.line(screen, (0, 220, 255), p1, p2, thickness)
            
            # Draw joint
            pygame.draw.circle(screen, (100, 230, 255), p1, thickness // 2 + 2)
            pygame.draw.circle(screen, (200, 250, 255), p1, thickness // 2)
            
            curr_x, curr_y = next_x, next_y
            cum_angle += env.angles[i]
        
        # Draw head (yellow marker)
        head_pos = segment_positions[-1]
        head_screen = (int((head_pos[0] - cam_x) * zoom + 400), 
                       int((head_pos[1] - cam_y) * zoom + 300))
        pygame.draw.circle(screen, (255, 220, 50), head_screen, int(6 * zoom / 15) + 3)
        pygame.draw.circle(screen, (255, 250, 150), head_screen, int(4 * zoom / 15) + 1)
        
        # Draw UI panel background
        panel_rect = pygame.Rect(0, 520, 800, 80)
        pygame.draw.rect(screen, (20, 30, 50, 230), panel_rect)
        pygame.draw.line(screen, (60, 80, 120), (0, 520), (800, 520), 2)
        
        # Draw slider and button
        slider.draw(screen, small_font)
        play_button.draw(screen, font)
        
        # Draw info panel (top left)
        info_panel = pygame.Surface((200, 120), pygame.SRCALPHA)
        pygame.draw.rect(info_panel, (20, 30, 50, 200), info_panel.get_rect(), border_radius=10)
        screen.blit(info_panel, (10, 10))
        
        screen.blit(font.render(f"DISTANCE: {env.pos[0]:.2f}m", True, (255, 255, 255)), (20, 15))
        screen.blit(font.render(f"SEGMENTS: {env.num_segments}", True, (0, 220, 255)), (20, 42))
        
        # Energy bar
        energy_pct = env.energy / env.max_energy
        energy_color = (100, 255, 100) if energy_pct > 0.5 else (255, 200, 50) if energy_pct > 0.25 else (255, 80, 80)
        pygame.draw.rect(screen, (40, 40, 60), (20, 75, 150, 18), border_radius=4)
        pygame.draw.rect(screen, energy_color, (20, 75, int(150 * energy_pct), 18), border_radius=4)
        pygame.draw.rect(screen, (150, 150, 150), (20, 75, 150, 18), 2, border_radius=4)
        screen.blit(small_font.render(f"ENERGY: {int(env.energy)}%", True, (255, 255, 255)), (25, 76))
        
        # Current checkpoint info (top right)
        if checkpoints and current_checkpoint_idx < len(checkpoints):
            ck = checkpoints[current_checkpoint_idx]
            info_panel2 = pygame.Surface((200, 70), pygame.SRCALPHA)
            pygame.draw.rect(info_panel2, (20, 30, 50, 200), info_panel2.get_rect(), border_radius=10)
            screen.blit(info_panel2, (590, 10))
            
            screen.blit(font.render(f"EPISODE: {ck.episode}", True, (100, 255, 150)), (600, 15))
            screen.blit(small_font.render(f"Best: {ck.final_distance:.2f}m", True, (200, 200, 200)), (600, 45))
        
        # Draw notifications
        notifications.draw(screen, font, y_start=140)
        
        # Instructions (when not playing)
        if not is_playing:
            inst_text = small_font.render("Select episode with slider, press PLAY to start", True, (150, 150, 180))
            screen.blit(inst_text, (200, 500))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()


# ==========================================================
# 5. Main Entry Point
# ==========================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   AQUATIC EVOLUTION - Q-LEARNING SIMULATION")
    print("=" * 60)
    print("\nThis simulation trains an aquatic creature to swim")
    print("using Q-learning (cost minimization).")
    print("\nActions include:")
    print("  - Moving body segments (+/- angles)")
    print("  - Adding new body segments")
    print("  - Removing body segments")
    print("\nThe creature learns optimal swimming through")
    print("reinforcement learning over multiple episodes.")
    print("-" * 60)
    
    # Get user preferences
    try:
        episodes_input = input("\nNumber of training episodes [default=200]: ").strip()
        episodes = int(episodes_input) if episodes_input else 200
        
        speed_input = input("Playback speed (0.01-1.0, lower=slower) [default=0.08]: ").strip()
        playback_speed = float(speed_input) if speed_input else 0.08
        playback_speed = max(0.01, min(1.0, playback_speed))
        
    except ValueError:
        episodes = 200
        playback_speed = 0.08
    
    print(f"\nTraining for {episodes} episodes...")
    print(f"Playback speed: {playback_speed}")
    print("-" * 60 + "\n")
    
    # Train and save checkpoints
    model, checkpoints = train_with_checkpoints(episodes=episodes, checkpoint_interval=20)
    
    # Launch visualization
    print("\n" + "=" * 60)
    print("LAUNCHING VISUALIZATION")
    print("=" * 60)
    print("\nControls:")
    print("  - Use the SLIDER to select different training episodes")
    print("  - Press PLAY to watch the creature swim")
    print("  - Press PAUSE to stop")
    print("  - Close window to exit")
    print("-" * 60 + "\n")
    
    visualize_with_ui(model, checkpoints, playback_speed=playback_speed)

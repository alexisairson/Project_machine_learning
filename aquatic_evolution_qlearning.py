import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pygame
import time
from collections import deque

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
# 2. Environment
# ==========================================================
class AquaticEnv:
    def __init__(self, num_segments=5):
        self.initial_num_segments = num_segments
        self.num_segments = num_segments
        self.state_dim = (num_segments * 2) + 2 
        self.action_dim = num_segments * 2 + 1  # +1 for "do nothing"
        self.reset()

    def reset(self):
        self.num_segments = self.initial_num_segments
        self.angles = np.zeros(self.num_segments, dtype=np.float32)
        self.pos = np.array([0.0, 0.0], dtype=np.float32)
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.time = 0
        self.energy = 100.0  # Starting energy
        self.max_energy = 100.0
        self.segment_change_event = None  # Track segment changes
        return self._get_obs()

    def _get_obs(self):
        clock = np.sin(self.time * 0.3)
        # Pad angles if needed for variable segment count
        angle_sin = np.sin(self.angles)
        angle_cos = np.cos(self.angles)
        return np.concatenate([angle_sin, angle_cos, 
                               [np.linalg.norm(self.velocity)], [clock]]).astype(np.float32)

    def step(self, action):
        old_angles = self.angles.copy()
        self.segment_change_event = None
        
        # Action interpretation
        energy_cost = 0.0
        if action < self.num_segments:
            self.angles[action] += 0.3
            energy_cost = 0.5
        elif action < 2 * self.num_segments:
            self.angles[action - self.num_segments] -= 0.3
            energy_cost = 0.5
        # else: action == 2 * num_segments means "do nothing"
        
        self.angles = np.clip(self.angles, -1.0, 1.0)
        thrust_mag = np.sum(np.abs(self.angles - old_angles))
        direction = np.mean(self.angles)
        
        # Physics: thrust based on angle changes
        thrust = np.array([np.cos(direction), np.sin(direction)]) * thrust_mag * 0.5
        self.velocity = self.velocity * 0.92 + thrust 
        self.pos += self.velocity
        
        # Energy management
        self.energy -= energy_cost
        self.energy = max(0, self.energy)
        # Small energy regeneration (like breathing/recovery)
        self.energy = min(self.max_energy, self.energy + 0.02)
        
        # Cost: balance between energy efficiency and forward progress
        cost = 0.1 * thrust_mag - self.velocity[0] + energy_cost * 0.1
        self.time += 1
        
        done = self.time > 200 or self.energy <= 0
        return self._get_obs(), float(cost), done, thrust_mag

    def add_segment(self):
        """Add a new segment to the creature"""
        if self.num_segments < 10:  # Max 10 segments
            self.num_segments += 1
            self.angles = np.append(self.angles, 0.0)
            self.action_dim = self.num_segments * 2 + 1
            self.segment_change_event = ("ADD", self.num_segments - 1)
            return True
        return False

    def remove_segment(self):
        """Remove a segment from the creature"""
        if self.num_segments > 2:  # Min 2 segments
            self.num_segments -= 1
            self.angles = self.angles[:-1]
            self.action_dim = self.num_segments * 2 + 1
            self.segment_change_event = ("REMOVE", self.num_segments)
            return True
        return False

# ==========================================================
# 3. Visuals (Pygame)
# ==========================================================
class Bubble:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.vx = random.uniform(-0.5, 0.5)
        self.vy = random.uniform(-1.0, -0.2)
        self.life = 255
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 5
    def draw(self, screen, cam_x, cam_y, zoom):
        alpha = max(0, min(255, int(self.life)))
        radius = max(1, int(2 * zoom / 15))
        
        s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        pygame.draw.circle(s, (200, 230, 255, alpha), (radius, radius), radius)
        
        pos = (int((self.x - cam_x) * zoom + 400), int((self.y - cam_y) * zoom + 300))
        screen.blit(s, (pos[0]-radius, pos[1]-radius))


class NotificationFrame:
    """Display notification for segment changes"""
    def __init__(self):
        self.messages = []  # List of (message, timestamp)
        self.duration = 3.0  # seconds
    
    def add_message(self, text, color=(255, 255, 100)):
        self.messages.append((text, time.time(), color))
    
    def update(self):
        current_time = time.time()
        self.messages = [(m, t, c) for m, t, c in self.messages if current_time - t < self.duration]
    
    def draw(self, screen, font):
        y_offset = 150
        for message, timestamp, color in self.messages:
            alpha = max(0, min(255, int(255 * (1 - (time.time() - timestamp) / self.duration))))
            
            # Create text surface
            text_surface = font.render(message, True, color)
            
            # Create semi-transparent background
            padding = 10
            bg_rect = text_surface.get_rect()
            bg_rect.topleft = (20, y_offset)
            bg_rect.inflate_ip(padding * 2, padding * 2)
            
            # Draw background
            bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
            pygame.draw.rect(bg_surface, (40, 40, 60, int(alpha * 0.8)), bg_surface.get_rect(), border_radius=5)
            pygame.draw.rect(bg_surface, (*color[:3], int(alpha * 0.5)), bg_surface.get_rect(), width=2, border_radius=5)
            screen.blit(bg_surface, bg_rect)
            
            # Draw text
            text_rect = text_surface.get_rect()
            text_rect.center = bg_rect.center
            screen.blit(text_surface, text_rect)
            
            y_offset += 40


def run_visual_simulation(model, generation_num, slow_factor=0.3):
    """
    Run visual simulation with improved features:
    - Slower simulation speed
    - Red trajectory line
    - Starting point always visible
    - Dynamic dezoom
    - Energy display
    - Segment change notifications
    """
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Aquatic Evolution - Q-Learning")
    font = pygame.font.SysFont("Arial", 22, bold=True)
    small_font = pygame.font.SysFont("Arial", 16)
    notification_font = pygame.font.SysFont("Arial", 18, bold=True)
    
    clock = pygame.time.Clock()
    env = AquaticEnv()
    state = env.reset()
    
    bubbles = []
    trajectory = deque(maxlen=500)  # Store trajectory points (red line)
    zoom = 15.0
    min_zoom = 3.0
    
    notifications = NotificationFrame()
    
    # Simulation timing
    sim_speed = slow_factor  # Slow factor (lower = slower)
    frame_count = 0
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Add segment manually for demonstration
                    if env.add_segment():
                        notifications.add_message("SEGMENT ADDED!", (100, 255, 100))
                elif event.key == pygame.K_BACKSPACE:
                    # Remove segment manually for demonstration
                    if env.remove_segment():
                        notifications.add_message("SEGMENT REMOVED!", (255, 100, 100))
                elif event.key == pygame.K_r:
                    # Reset simulation
                    state = env.reset()
                    trajectory.clear()
                    bubbles.clear()
                    notifications.add_message("SIMULATION RESET", (100, 200, 255))

        # Slow down simulation by only updating every N frames
        frame_count += 1
        update_simulation = (frame_count % max(1, int(1/sim_speed))) == 0
        
        if update_simulation:
            with torch.no_grad():
                # Handle variable action space
                action = torch.argmin(model(torch.from_numpy(state))[:env.action_dim]).item()
            
            state, _, done, thrust_power = env.step(action)
            
            # Check for segment changes from environment
            if env.segment_change_event:
                event_type, seg_idx = env.segment_change_event
                if event_type == "ADD":
                    notifications.add_message(f"MEMBER #{seg_idx+1} ADDED!", (100, 255, 100))
                else:
                    notifications.add_message(f"MEMBER #{seg_idx+1} DISCARDED!", (255, 100, 100))
            
            if done: 
                state = env.reset()
                trajectory.clear()
                bubbles = []
                notifications.add_message("NEW GENERATION STARTING...", (200, 200, 200))

        # Physics to Pixels
        px_x, px_y = env.pos[0] * 10, env.pos[1] * 10
        
        # Store trajectory point (every few frames to avoid too many points)
        if update_simulation:
            trajectory.append((px_x, px_y))

        if update_simulation and thrust_power > 0.05:
            bubbles.append(Bubble(px_x, px_y))

        # Camera Logic: Keep starting point visible
        # The camera should show both the creature and the starting point
        start_x, start_y = 0, 0
        
        # Calculate required zoom to keep both visible
        dist_from_start = np.sqrt((px_x - start_x)**2 + (px_y - start_y)**2)
        
        # Dynamic zoom: zoom out when creature moves far from start
        if dist_from_start > 0:
            # Calculate zoom to fit both creature and start point
            required_view_width = dist_from_start + 100  # Add margin
            target_zoom = min(15.0, max(min_zoom, 300 / required_view_width))
        else:
            target_zoom = 15.0
        
        zoom += (target_zoom - zoom) * 0.03  # Smooth zoom transition
        
        # Camera position: center between creature and start point, but bias towards creature
        cam_x = px_x * 0.7  # Bias towards creature
        cam_y = px_y * 0.5 + 300 / zoom  # Slight offset for better view
        
        # Additional dezoom if creature is approaching window edges
        screen_margin = 100
        creature_screen_x = (px_x - cam_x) * zoom + 400
        creature_screen_y = (px_y - cam_y) * zoom + 300
        
        if creature_screen_x < screen_margin or creature_screen_x > 800 - screen_margin:
            zoom = max(min_zoom, zoom * 0.98)
        if creature_screen_y < screen_margin or creature_screen_y > 600 - screen_margin:
            zoom = max(min_zoom, zoom * 0.98)

        # Draw
        screen.fill((10, 30, 60))  # Deep Underwater Blue

        # Draw Red Starting Line (Distance = 0)
        line_x = int((0 - cam_x) * zoom + 400)
        if -100 < line_x < 900:
            pygame.draw.line(screen, (255, 50, 50), (line_x, 0), (line_x, 600), 3)
            # Label for start line
            start_label = small_font.render("START", True, (255, 100, 100))
            screen.blit(start_label, (line_x - 25, 10))

        # Draw Trajectory (Red Line)
        if len(trajectory) > 1:
            points = [(int((tx - cam_x) * zoom + 400), int((ty - cam_y) * zoom + 300)) 
                      for tx, ty in trajectory]
            # Filter points that are on screen
            visible_points = [(x, y) for x, y in points if -50 < x < 850 and -50 < y < 650]
            if len(visible_points) > 1:
                # Draw with gradient (older = more transparent)
                for i in range(1, len(visible_points)):
                    alpha = int(255 * (i / len(visible_points)))
                    color = (255, 50 + alpha//3, 50)
                    pygame.draw.line(screen, color, visible_points[i-1], visible_points[i], 
                                   max(1, int(2 * zoom / 15)))

        # Draw Bubbles
        for b in bubbles[:]:
            b.update()
            if b.life <= 0:
                bubbles.remove(b)
            else:
                b.draw(screen, cam_x, cam_y, zoom)

        # Draw Creature Body
        curr_x, curr_y = px_x, px_y
        cum_angle = 0
        for i in range(env.num_segments):
            seg_len = 15 
            next_x = curr_x + np.cos(cum_angle) * seg_len
            next_y = curr_y + np.sin(cum_angle) * seg_len
            
            p1 = (int((curr_x - cam_x) * zoom + 400), int((curr_y - cam_y) * zoom + 300))
            p2 = (int((next_x - cam_x) * zoom + 400), int((next_y - cam_y) * zoom + 300))
            
            # Draw segment (Cyan Blue)
            thickness = int(max(2, 8 * zoom / 15))
            pygame.draw.line(screen, (0, 200, 255), p1, p2, thickness)
            # Draw joint (Bright Cyan)
            pygame.draw.circle(screen, (150, 255, 255), p1, thickness // 2 + 1)
            
            curr_x, curr_y = next_x, next_y
            cum_angle += env.angles[i]
        
        # Draw head marker
        pygame.draw.circle(screen, (255, 200, 0), p1, thickness + 2)

        # Draw Energy Bar
        energy_bar_width = 200
        energy_bar_height = 20
        energy_x = 20
        energy_y = 80
        
        # Background
        pygame.draw.rect(screen, (40, 40, 60), (energy_x, energy_y, energy_bar_width, energy_bar_height), border_radius=5)
        # Energy fill
        energy_fill = int((env.energy / env.max_energy) * energy_bar_width)
        energy_color = (100, 255, 100) if env.energy > 50 else (255, 200, 50) if env.energy > 25 else (255, 80, 80)
        pygame.draw.rect(screen, energy_color, (energy_x, energy_y, energy_fill, energy_bar_height), border_radius=5)
        # Border
        pygame.draw.rect(screen, (200, 200, 200), (energy_x, energy_y, energy_bar_width, energy_bar_height), 2, border_radius=5)
        # Label
        energy_label = font.render(f"ENERGY: {int(env.energy)}%", True, (255, 255, 255))
        screen.blit(energy_label, (energy_x + energy_bar_width + 10, energy_y - 2))

        # UI
        txt_col = (255, 255, 255)
        screen.blit(font.render(f"DISTANCE: {env.pos[0]:.2f}m", True, txt_col), (20, 20))
        screen.blit(font.render(f"GENERATION: {generation_num}", True, (0, 255, 100)), (20, 50))
        
        # Member count
        member_text = f"SEGMENTS: {env.num_segments}"
        screen.blit(font.render(member_text, True, (0, 200, 255)), (550, 20))
        
        # Controls hint
        controls = small_font.render("SPACE: Add segment | BACKSPACE: Remove | R: Reset", True, (150, 150, 150))
        screen.blit(controls, (20, 570))

        # Update and draw notifications
        notifications.update()
        notifications.draw(screen, notification_font)

        pygame.display.flip()
        clock.tick(60)  # 60 FPS for smooth animation (but simulation updates slower)

    pygame.quit()

# ==========================================================
# 4. Training (DQN with Cost Minimization)
# ==========================================================
def train(episodes=200):
    env = AquaticEnv()
    model = QNetwork(env.state_dim, env.action_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training AI (Cost Minimization Strategy)...")
    print("=" * 50)
    
    best_distance = -float('inf')
    
    for e in range(episodes + 1):
        state = env.reset()
        epsilon = max(0.01, 1.0 - (e / episodes))
        total_distance = 0
        
        for step in range(200):
            # Epsilon-greedy: choose random action or min(Q)
            if random.random() < epsilon:
                action = random.randint(0, env.action_dim - 1)
            else:
                with torch.no_grad():
                    q_values = model(torch.from_numpy(state))
                    action = torch.argmin(q_values[:env.action_dim]).item()

            next_state, cost, done, _ = env.step(action)
            total_distance = env.pos[0]
            
            with torch.no_grad():
                # Bellman Target for Costs: target = cost + gamma * min(Q_next)
                q_next = model(torch.from_numpy(next_state))
                target = cost + 0.95 * torch.min(q_next[:env.action_dim]).item()
            
            current_pred = model(torch.from_numpy(state))[action]
            loss = nn.MSELoss()(current_pred, torch.tensor(target, dtype=torch.float32))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state = next_state
            if done: break
        
        if total_distance > best_distance:
            best_distance = total_distance
        
        if e % 50 == 0:
            print(f"Gen {e:3d} | Distance: {total_distance:6.2f}m | Best: {best_distance:6.2f}m | Epsilon: {epsilon:.3f}")
    
    print("=" * 50)
    print(f"Training complete! Best distance achieved: {best_distance:.2f}m")
    return model, episodes

# ==========================================================
# 5. Evolution with Q-Learning Integration
# ==========================================================
class EvolvingCreature:
    """A creature that can evolve its structure using Q-learning"""
    def __init__(self, num_segments=5):
        self.env = AquaticEnv(num_segments)
        self.model = QNetwork(self.env.state_dim, self.env.action_dim)
        self.fitness = 0
        self.energy_efficiency = 0
    
    def evaluate(self, steps=200):
        """Evaluate creature fitness"""
        state = self.env.reset()
        total_distance = 0
        total_energy_used = 0
        
        for _ in range(steps):
            with torch.no_grad():
                action = torch.argmin(self.model(torch.from_numpy(state))[:self.env.action_dim]).item()
            
            state, cost, done, thrust = self.env.step(action)
            total_distance = self.env.pos[0]
            total_energy_used += thrust * 0.5
            if done: break
        
        self.fitness = total_distance - total_energy_used * 0.01
        self.energy_efficiency = total_distance / max(1, total_energy_used)
        return self.fitness
    
    def mutate_structure(self):
        """Mutate the creature's structure (add/remove segments)"""
        mutation_type = random.choice(['add', 'remove', 'none'])
        
        if mutation_type == 'add' and self.env.num_segments < 10:
            self.env.add_segment()
            # Expand model to handle new action space
            old_model = self.model
            self.model = QNetwork(self.env.state_dim, self.env.action_dim)
            # Copy old weights where possible
            with torch.no_grad():
                for i, param in enumerate(old_model.parameters()):
                    if i < len(list(self.model.parameters())):
                        self.model.state_dict()
            return "ADD"
        elif mutation_type == 'remove' and self.env.num_segments > 2:
            self.env.remove_segment()
            # Shrink model
            self.model = QNetwork(self.env.state_dim, self.env.action_dim)
            return "REMOVE"
        return "NONE"


def run_evolution_simulation(generations=50, population_size=20, slow_factor=0.3):
    """
    Run evolutionary simulation with Q-learning:
    - Population of creatures
    - Selection based on fitness
    - Structure mutation (add/remove segments)
    - Visualization of best creature
    """
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Aquatic Evolution - Q-Learning with Structure Evolution")
    font = pygame.font.SysFont("Arial", 22, bold=True)
    small_font = pygame.font.SysFont("Arial", 16)
    notification_font = pygame.font.SysFont("Arial", 18, bold=True)
    
    clock = pygame.time.Clock()
    notifications = NotificationFrame()
    
    # Initialize population
    population = [EvolvingCreature(num_segments=random.randint(3, 7)) for _ in range(population_size)]
    
    generation = 0
    running = True
    
    while running and generation < generations:
        # Evaluate all creatures
        fitness_scores = []
        for creature in population:
            fitness = creature.evaluate(steps=100)
            fitness_scores.append((fitness, creature))
        
        # Sort by fitness (higher is better)
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        best_creature = fitness_scores[0][1]
        
        # Visualize best creature
        env = best_creature.env
        state = env.reset()
        trajectory = deque(maxlen=500)
        bubbles = []
        zoom = 15.0
        
        # Run visualization for this generation's best
        for step in range(150):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
            
            if not running:
                break
                
            # Update simulation
            with torch.no_grad():
                action = torch.argmin(best_creature.model(torch.from_numpy(state))[:env.action_dim]).item()
            
            state, _, done, thrust_power = env.step(action)
            
            if env.segment_change_event:
                event_type, seg_idx = env.segment_change_event
                if event_type == "ADD":
                    notifications.add_message(f"MEMBER #{seg_idx+1} ADDED!", (100, 255, 100))
                else:
                    notifications.add_message(f"MEMBER #{seg_idx+1} DISCARDED!", (255, 100, 100))
            
            if done:
                state = env.reset()
                trajectory.clear()
            
            px_x, px_y = env.pos[0] * 10, env.pos[1] * 10
            trajectory.append((px_x, px_y))
            
            if thrust_power > 0.05:
                bubbles.append(Bubble(px_x, px_y))
            
            # Camera
            dist_from_start = abs(px_x)
            target_zoom = max(3.0, min(15.0, 300 / (dist_from_start + 50)))
            zoom += (target_zoom - zoom) * 0.03
            cam_x = px_x * 0.7
            cam_y = px_y * 0.5
            
            # Draw
            screen.fill((10, 30, 60))
            
            # Start line
            line_x = int((0 - cam_x) * zoom + 400)
            pygame.draw.line(screen, (255, 50, 50), (line_x, 0), (line_x, 600), 3)
            start_label = small_font.render("START", True, (255, 100, 100))
            screen.blit(start_label, (line_x - 25, 10))
            
            # Trajectory
            if len(trajectory) > 1:
                points = [(int((tx - cam_x) * zoom + 400), int((ty - cam_y) * zoom + 300)) 
                          for tx, ty in trajectory]
                for i in range(1, len(points)):
                    pygame.draw.line(screen, (255, 50, 50), points[i-1], points[i], 
                                   max(1, int(2 * zoom / 15)))
            
            # Bubbles
            for b in bubbles[:]:
                b.update()
                if b.life <= 0:
                    bubbles.remove(b)
                else:
                    b.draw(screen, cam_x, cam_y, zoom)
            
            # Creature
            curr_x, curr_y = px_x, px_y
            cum_angle = 0
            for i in range(env.num_segments):
                seg_len = 15
                next_x = curr_x + np.cos(cum_angle) * seg_len
                next_y = curr_y + np.sin(cum_angle) * seg_len
                
                p1 = (int((curr_x - cam_x) * zoom + 400), int((curr_y - cam_y) * zoom + 300))
                p2 = (int((next_x - cam_x) * zoom + 400), int((next_y - cam_y) * zoom + 300))
                
                thickness = int(max(2, 8 * zoom / 15))
                pygame.draw.line(screen, (0, 200, 255), p1, p2, thickness)
                pygame.draw.circle(screen, (150, 255, 255), p1, thickness // 2 + 1)
                
                curr_x, curr_y = next_x, next_y
                cum_angle += env.angles[i]
            
            pygame.draw.circle(screen, (255, 200, 0), p1, thickness + 2)
            
            # Energy bar
            energy_bar_width = 200
            energy_x, energy_y = 20, 80
            pygame.draw.rect(screen, (40, 40, 60), (energy_x, energy_y, energy_bar_width, 20), border_radius=5)
            energy_fill = int((env.energy / env.max_energy) * energy_bar_width)
            energy_color = (100, 255, 100) if env.energy > 50 else (255, 200, 50)
            pygame.draw.rect(screen, energy_color, (energy_x, energy_y, energy_fill, 20), border_radius=5)
            pygame.draw.rect(screen, (200, 200, 200), (energy_x, energy_y, energy_bar_width, 20), 2, border_radius=5)
            energy_label = font.render(f"ENERGY: {int(env.energy)}%", True, (255, 255, 255))
            screen.blit(energy_label, (energy_x + energy_bar_width + 10, energy_y - 2))
            
            # UI
            screen.blit(font.render(f"DISTANCE: {env.pos[0]:.2f}m", True, (255, 255, 255)), (20, 20))
            screen.blit(font.render(f"GENERATION: {generation}", True, (0, 255, 100)), (20, 50))
            screen.blit(font.render(f"SEGMENTS: {env.num_segments}", True, (0, 200, 255)), (550, 20))
            screen.blit(font.render(f"BEST FITNESS: {fitness_scores[0][0]:.2f}", True, (255, 200, 100)), (550, 50))
            
            controls = small_font.render("Evolution mode - Best creature shown", True, (150, 150, 150))
            screen.blit(controls, (20, 570))
            
            notifications.update()
            notifications.draw(screen, notification_font)
            
            pygame.display.flip()
            clock.tick(60)
        
        # Selection and reproduction
        top_half = [c for _, c in fitness_scores[:population_size // 2]]
        new_population = top_half.copy()
        
        while len(new_population) < population_size:
            parent = random.choice(top_half)
            child = EvolvingCreature(num_segments=parent.env.num_segments)
            # Copy parent's model weights
            child.model.load_state_dict(parent.model.state_dict())
            # Mutate structure
            mutation = child.mutate_structure()
            if mutation == "ADD":
                notifications.add_message("EVOLUTION: New segment added!", (100, 255, 150))
            elif mutation == "REMOVE":
                notifications.add_message("EVOLUTION: Segment removed!", (255, 150, 100))
            new_population.append(child)
        
        population = new_population
        generation += 1
        notifications.add_message(f"GENERATION {generation} COMPLETE", (100, 200, 255))
    
    pygame.quit()
    return best_creature


if __name__ == "__main__":
    print("=" * 60)
    print("AQUATIC EVOLUTION - Q-LEARNING SIMULATION")
    print("=" * 60)
    print("\nSelect mode:")
    print("1. Train and visualize (standard)")
    print("2. Evolution simulation (with structure mutations)")
    print()
    
    try:
        choice = input("Enter choice (1 or 2, default=1): ").strip()
        if choice == "2":
            slow = input("Slow factor (0.1-1.0, default=0.3): ").strip()
            slow_factor = float(slow) if slow else 0.3
            slow_factor = max(0.1, min(1.0, slow_factor))
            run_evolution_simulation(slow_factor=slow_factor)
        else:
            slow = input("Slow factor (0.1-1.0, default=0.3): ").strip()
            slow_factor = float(slow) if slow else 0.3
            slow_factor = max(0.1, min(1.0, slow_factor))
            
            trained_model, total_gen = train(200)
            print("\nLaunching visualization...")
            print("Controls:")
            print("  SPACE: Add segment")
            print("  BACKSPACE: Remove segment")
            print("  R: Reset simulation")
            print()
            run_visual_simulation(trained_model, total_gen, slow_factor=slow_factor)
    except KeyboardInterrupt:
        print("\nSimulation terminated by user.")

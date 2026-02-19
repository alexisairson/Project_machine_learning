import pygame
import numpy as np
import math
import pickle
import os
import time
import copy  # Added this

# ==========================================================
# 0. Data Structure (Must be defined to load pickle files)
# ==========================================================
class TrainingCheckpoint:
    """Matches the definition in the training script"""
    def __init__(self, episode, model_state, trajectory, max_distance, use_nn, num_segments):
        self.episode = episode
        self.model_state = copy.deepcopy(model_state)
        self.trajectory = copy.deepcopy(trajectory)
        self.max_distance = max_distance
        self.use_nn = use_nn
        self.num_segments = num_segments

# ==========================================================
# 1. UI Components (Fixed Input Handling)
# ==========================================================
class UIInputBox:
    """Input box for text/numeric input - V8 Fixed"""
    def __init__(self, x, y, w, h, initial_value="1.0"):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = str(initial_value)
        self.active = False
        self.font = pygame.font.SysFont("Arial", 14)
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
            return False
        
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                self.active = False
                return True # Confirm input
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                # V8 FIX: Accept digits and dots more reliably
                if event.unicode.isdigit() or event.unicode == '.':
                    if len(self.text) < 8:
                        self.text += event.unicode
            return False
        return False
    
    def get_value(self):
        try:
            return float(self.text) if self.text else 1.0
        except:
            return 1.0
    
    def set_value(self, v):
        # Update text to match slider if needed
        self.text = f"{v:.2f}" if v < 10 else f"{v:.1f}"
    
    def draw(self, screen):
        color = (100, 150, 255) if self.active else (80, 80, 100)
        pygame.draw.rect(screen, (40, 40, 60), self.rect, border_radius=4)
        pygame.draw.rect(screen, color, self.rect, 2, border_radius=4)
        
        text_surface = self.font.render(self.text, True, (255, 255, 255))
        # Center text
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

class UIEpisodeSlider:
    def __init__(self, x, y, w, h, checkpoints, label=""):
        self.rect = pygame.Rect(x, y, w, h)
        self.checkpoints = checkpoints
        self.label = label
        self.dragging = False
        self.font = pygame.font.SysFont("Arial", 14)
        self.idx = 0
    
    def get_current_episode(self):
        if self.checkpoints and self.idx < len(self.checkpoints):
            return self.checkpoints[self.idx].episode
        return 0
    
    def get_knob_x(self):
        if len(self.checkpoints) <= 1:
            return self.rect.x
        ratio = self.idx / (len(self.checkpoints) - 1)
        return int(self.rect.x + ratio * self.rect.width)
    
    def _set_from_x(self, x):
        ratio = max(0, min(1, (x - self.rect.x) / self.rect.width))
        self.idx = int(ratio * (len(self.checkpoints) - 1))
        self.idx = max(0, min(len(self.checkpoints) - 1, self.idx))
    
    def handle_event(self, event, pos):
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Click on track or near knob
            if self.rect.collidepoint(pos) or abs(pos[0] - self.get_knob_x()) < 12:
                self.dragging = True
                self._set_from_x(pos[0])
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._set_from_x(pos[0])
    
    def draw(self, screen, font):
        screen.blit(font.render(self.label, True, (200, 200, 200)), (self.rect.x, self.rect.y - 18))
        # Track
        pygame.draw.rect(screen, (60, 60, 80), 
                        pygame.Rect(self.rect.x, self.rect.centery - 2, self.rect.width, 4), border_radius=2)
        # Filled part
        kx = self.get_knob_x()
        pygame.draw.rect(screen, (100, 180, 255),
                        pygame.Rect(self.rect.x, self.rect.centery - 2, kx - self.rect.x, 4), border_radius=2)
        # Knob
        pygame.draw.circle(screen, (120, 200, 255), (kx, self.rect.centery), 10)
        pygame.draw.circle(screen, (255, 255, 255), (kx, self.rect.centery), 7)
        
        # Label
        ep = self.get_current_episode()
        screen.blit(self.font.render(f"Ep {ep}", True, (200, 200, 200)), 
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
            # Handle zero/negative safety
            safe_val = max(self.value, self.min_v)
            ratio = (math.log10(safe_val) - math.log10(self.min_v)) / \
                    max(math.log10(self.max_v) - math.log10(self.min_v), 0.001)
        else:
            ratio = (self.value - self.min_v) / max(self.max_v - self.min_v, 0.001)
        return int(self.rect.x + ratio * self.rect.width)
    
    def _set_from_x(self, x):
        ratio = max(0, min(1, (x - self.rect.x) / self.rect.width))
        if self.log:
            self.value = 10 ** (math.log10(self.min_v) + ratio * (math.log10(self.max_v) - math.log10(self.min_v)))
        else:
            self.value = self.min_v + ratio * (self.max_v - self.min_v)
    
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
# 2. Visualization
# ==========================================================
def visualize(checkpoints, target_angle=0.0, duration=10.0):
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Aquatic Evolution V8 - Visualizer")
    
    font = pygame.font.SysFont("Arial", 18, bold=True)
    small = pygame.font.SysFont("Arial", 14)
    tiny = pygame.font.SysFont("Arial", 12)
    
    clock = pygame.time.Clock()
    
    target_rad = math.radians(target_angle)
    direction = np.array([math.cos(target_rad), math.sin(target_rad)])
    
    # UI Elements
    ep_slider = UIEpisodeSlider(50, 530, 120, 18, checkpoints, "Episode")
    speed_slider = UIFloatSlider(250, 530, 100, 18, 0.01, 100.0, 1.0, "Speed", log=True)
    # V8 Fix: Input box width increased slightly for better visibility
    speed_input = UIInputBox(360, 523, 50, 25, "1.0")
    
    play_btn = UIButton(430, 522, 60, 32, "PLAY", (40, 120, 80))
    reset_btn = UIButton(500, 522, 60, 32, "RESET", (100, 80, 40))
    
    ep_idx = 0
    playing = False
    playback_pos = 0
    manual_zoom = 15.0
    speed = 1.0
    
    running = True
    last_upd = time.time()
    
    while running:
        now = time.time()
        mouse = pygame.mouse.get_pos()
        
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            
            if e.type == pygame.MOUSEWHEEL:
                zoom_factor = 1.2 if e.y > 0 else 0.8
                manual_zoom *= zoom_factor
                manual_zoom = max(1.0, min(100.0, manual_zoom))
            
            # Handle UI
            ep_slider.handle_event(e, mouse)
            speed_slider.handle_event(e, mouse)
            
            # V8 Fix: Handle input box separately
            if speed_input.handle_event(e):
                # User pressed Enter in speed box
                speed = speed_input.get_value()
                speed_slider.value = max(0.01, min(100.0, speed))
            
            if e.type == pygame.MOUSEBUTTONDOWN:
                if play_btn.clicked(mouse):
                    playing = not playing
                    play_btn.text = "PAUSE" if playing else "PLAY"
                    play_btn.color = (120, 80, 40) if playing else (40, 120, 80)
                
                if reset_btn.clicked(mouse):
                    playback_pos = 0
            
            # Sync slider to input box if dragging slider
            if speed_slider.dragging:
                speed_input.set_value(speed_slider.value)
            
            # Reset playback if episode changed manually
            new_idx = ep_slider.idx
            if new_idx != ep_idx:
                ep_idx = new_idx
                playback_pos = 0
        
        # Update speed from slider if not using input
        if not speed_input.active:
            speed = speed_slider.value
            # Only update text if not currently typing
            # This keeps the input box showing the slider value
        
        play_btn.check_hover(mouse)
        reset_btn.check_hover(mouse)
        
        # Get current checkpoint data
        if checkpoints and ep_idx < len(checkpoints):
            ck = checkpoints[ep_idx]
            trajectory = ck.trajectory
            max_dist = ck.max_distance
            use_nn = ck.use_nn
            curr_num_segments = ck.num_segments
        else:
            trajectory = [{'pos': np.array([0.0, 0.0]), 'angles': np.zeros(5), 'num_segments': 5, 'energy': 100}]
            max_dist = 0
            use_nn = True
            curr_num_segments = 5
        
        # Playback logic
        interval = 0.02 / max(speed, 0.01)
        if playing and trajectory and (now - last_upd) >= interval:
            last_upd = now
            playback_pos = min(playback_pos + 1, len(trajectory) - 1)
        
        # Current state data
        if trajectory and playback_pos < len(trajectory):
            curr = trajectory[playback_pos]
            head = curr['pos'].copy()
            curr_energy = curr['energy']
            curr_segments = curr['num_segments']
        else:
            head = np.array([0.0, 0.0])
            curr_energy = 100
            curr_segments = curr_num_segments
        start = np.array([0.0, 0.0])
        
        zoom = manual_zoom
        cam_x = (head[0] + 0) / 2
        cam_y = (head[1] + 0) / 2
        
        # ========== DRAWING ==========
        screen.fill((8, 20, 40))
        
        # Grid axis
        axis_y_screen = int((0 - cam_y) * zoom + 280)
        pygame.draw.line(screen, (60, 60, 80), (0, axis_y_screen), (800, axis_y_screen), 1)
        
        # Target direction line
        line_start = start - direction * 2000
        line_end = start + direction * 2000
        p1 = (int((line_start[0]-cam_x)*zoom+400), int((line_start[1]-cam_y)*zoom+280))
        p2 = (int((line_end[0]-cam_x)*zoom+400), int((line_end[1]-cam_y)*zoom+280))
        pygame.draw.line(screen, (30, 100, 50), p1, p2, 5)
        pygame.draw.line(screen, (50, 200, 100), p1, p2, 2)
        
        # Arrow
        arrow = start + direction * 40
        ax = int((arrow[0]-cam_x)*zoom+400)
        ay = int((arrow[1]-cam_y)*zoom+280)
        perp = np.array([-direction[1], direction[0]])
        sz = int(12 * zoom / 15)
        pts = [
            (ax, ay),
            (ax - int(direction[0]*sz*2) + int(perp[0]*sz), ay - int(direction[1]*sz*2) + int(perp[1]*sz)),
            (ax - int(direction[0]*sz*2) - int(perp[0]*sz), ay - int(direction[1]*sz*2) - int(perp[1]*sz))
        ]
        pygame.draw.polygon(screen, (80, 220, 120), pts)
        
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
        
        # Start cross
        cx = int((0-cam_x)*zoom+400)
        cy = int((0-cam_y)*zoom+280)
        sz = int(10 * zoom / 15)
        pygame.draw.line(screen, (255, 50, 50), (cx-sz, cy), (cx+sz, cy), 4)
        pygame.draw.line(screen, (255, 50, 50), (cx, cy-sz), (cx, cy+sz), 4)
        screen.blit(tiny.render("START", True, (255, 100, 100)), (cx+sz+5, cy-8))
        
        # Trajectory
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
        
        # Creature
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
        speed_slider.draw(screen, small)
        speed_input.draw(screen)
        play_btn.draw(screen, font)
        reset_btn.draw(screen, font)
        
        # Info panel
        info_s = pygame.Surface((180, 115), pygame.SRCALPHA)
        pygame.draw.rect(info_s, (15, 25, 45, 220), info_s.get_rect(), border_radius=8)
        screen.blit(info_s, (10, 10))
        
        delta = head - start
        dist = np.dot(delta, direction)
        screen.blit(font.render(f"DIST: {dist:.1f}m", True, (255, 255, 255)), (20, 15))
        screen.blit(font.render(f"SEGS: {curr_segments}", True, (0, 220, 255)), (20, 38))
        
        method = "NN" if use_nn else "Tabular"
        screen.blit(small.render(f"Method: {method}", True, (200, 200, 200)), (20, 58))
        
        max_energy = 100 * (1 + (curr_segments - 2) * 0.25)
        e_pct = curr_energy / max(100, max_energy)
        e_color = (100, 255, 100) if e_pct > 0.5 else (255, 200, 50) if e_pct > 0.25 else (255, 80, 80)
        pygame.draw.rect(screen, (40, 40, 60), (20, 78, 140, 12), border_radius=3)
        pygame.draw.rect(screen, e_color, (20, 78, int(140*min(1, e_pct)), 12), border_radius=3)
        screen.blit(tiny.render(f"Energy: {int(curr_energy)}%", True, (255, 255, 255)), (25, 78))
        
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
        
        screen.blit(tiny.render(f"Zoom: {zoom:.1f}x (scroll)", True, (150, 150, 150)), (600, 575))
        
        if not playing:
            screen.blit(tiny.render("Press PLAY | Scroll to zoom", True, (100, 120, 160)), (300, 495))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

# ==========================================================
# 3. Loader
# ==========================================================
def list_saved_simulations():
    result_dir = os.path.join(os.getcwd(), "Result")
    if not os.path.exists(result_dir):
        return []
    files = [os.path.join(result_dir, f) for f in os.listdir(result_dir) 
             if f.startswith('aquatic_') and f.endswith('.pkl')]
    return sorted(files, reverse=True)

def load_simulation(filename):
    with open(filename, 'rb') as f:
        save_data = pickle.load(f)
    
    return (save_data['checkpoints'], save_data['use_nn'], 
            save_data['target_angle'], save_data['duration'],
            save_data.get('num_segments', 5), save_data.get('fixed_segments', True))

# ==========================================================
# 4. Main
# ==========================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   AQUATIC CREATURE VISUALIZER - V8")
    print("=" * 60)
    
    saved_files = list_saved_simulations()
    
    if not saved_files:
        print("No saved simulations found in 'Result/' folder.")
        print("Please run 'aquatic_evolution_train_V8.py' first.")
    else:
        print(f"Found {len(saved_files)} saved simulation(s):")
        for i, f in enumerate(saved_files[:10]):
            print(f"  {i+1}. {os.path.basename(f)}")
        
        try:
            choice = input("\nSelect number to load [1]: ").strip()
            idx = int(choice) - 1 if choice else 0
            if 0 <= idx < len(saved_files):
                print(f"\nLoading {os.path.basename(saved_files[idx])}...")
                checkpoints, use_nn, angle, duration, num_seg, fixed = load_simulation(saved_files[idx])
                print(f"Loaded: {len(checkpoints)} checkpoints")
                print(f"Target: {angle}°, Duration: {duration}s")
                visualize(checkpoints, angle, duration)
            else:
                print("Invalid selection.")
        except Exception as e:
            print(f"Error: {e}")
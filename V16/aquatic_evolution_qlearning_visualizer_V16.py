import pygame
import numpy as np
import math
import pickle
import os
import time

# ==========================================================
# 0. Data Structure
# ==========================================================
class TrainingCheckpoint:
    def __init__(self, episode, model_state, trajectory, max_distance, use_nn, num_segments):
        self.episode = episode
        self.model_state = model_state
        self.trajectory = trajectory
        self.max_distance = max_distance
        self.use_nn = use_nn
        self.num_segments = num_segments


# ==========================================================
# 1. Configuration Parser
# ==========================================================
def _parse_value(value_str):
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
    return _load_config_dict(filepath)


# ==========================================================
# 2. UI Components
# ==========================================================
class UIInputBox:
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
                return True
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
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
        if v < 1:
            self.text = f"{v:.2f}"
        elif v < 10:
            self.text = f"{v:.1f}"
        else:
            self.text = f"{int(v)}"
    
    def draw(self, screen):
        color = (100, 150, 255) if self.active else (80, 80, 100)
        pygame.draw.rect(screen, (40, 40, 60), self.rect, border_radius=4)
        pygame.draw.rect(screen, color, self.rect, 2, border_radius=4)
        
        text_surface = self.font.render(self.text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)


class UIProgressSlider:
    """YouTube-style progress slider for simulation timeline."""
    
    def __init__(self, x, y, w, h, duration, label=""):
        self.rect = pygame.Rect(x, y, w, h)
        self.duration = duration
        self.current_time = 0.0
        self.label = label
        self.dragging = False
        self.font = pygame.font.SysFont("Arial", 14)
        self.small_font = pygame.font.SysFont("Arial", 12)
    
    def get_knob_x(self):
        if self.duration <= 0:
            return self.rect.x
        ratio = self.current_time / self.duration
        return int(self.rect.x + ratio * self.rect.width)
    
    def _set_from_x(self, x):
        ratio = max(0, min(1, (x - self.rect.x) / self.rect.width))
        self.current_time = ratio * self.duration
    
    def set_time(self, t):
        self.current_time = max(0, min(self.duration, t))
    
    def get_frame_index(self, total_frames, dt):
        """Convert current time to frame index."""
        if total_frames <= 0:
            return 0
        frame = int(self.current_time / dt)
        return min(frame, total_frames - 1)
    
    def handle_event(self, event, pos):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(pos) or abs(pos[0] - self.get_knob_x()) < 12:
                self.dragging = True
                self._set_from_x(pos[0])
                return True  # Indicate that time was changed
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._set_from_x(pos[0])
            return True
        return False
    
    def draw(self, screen, font):
        # Label
        screen.blit(font.render(self.label, True, (200, 200, 200)), (self.rect.x, self.rect.y - 18))
        
        # Track background
        pygame.draw.rect(screen, (60, 60, 80), 
                        pygame.Rect(self.rect.x, self.rect.centery - 3, self.rect.width, 6), border_radius=3)
        
        # Progress fill
        kx = self.get_knob_x()
        pygame.draw.rect(screen, (0, 180, 180),  # Teal color
                        pygame.Rect(self.rect.x, self.rect.centery - 3, kx - self.rect.x, 6), border_radius=3)
        
        # Knob
        pygame.draw.circle(screen, (0, 200, 200), (kx, self.rect.centery), 8)
        pygame.draw.circle(screen, (255, 255, 255), (kx, self.rect.centery), 5)
        
        # Time display
        current_str = f"{self.current_time:.1f}s"
        total_str = f"{self.duration:.0f}s"
        time_text = f"{current_str} / {total_str}"
        screen.blit(self.small_font.render(time_text, True, (200, 200, 200)), 
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
    def __init__(self, x, y, w, h, text, color=(60, 60, 80), toggle=False):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.hover = False
        self.toggle = toggle
        self.active = False
    
    def draw(self, screen, font):
        if self.toggle and self.active:
            c = (self.color[0] + 40, self.color[1] + 40, self.color[2] + 40)
            c = tuple(min(255, v) for v in c)
        elif self.hover:
            c = (80, 80, 100)
        else:
            c = self.color
        pygame.draw.rect(screen, c, self.rect, border_radius=8)
        border_color = (200, 200, 100) if (self.toggle and self.active) else (150, 150, 170)
        pygame.draw.rect(screen, border_color, self.rect, 2, border_radius=8)
        s = font.render(self.text, True, (255, 255, 255))
        screen.blit(s, s.get_rect(center=self.rect.center))
    
    def check_hover(self, pos):
        self.hover = self.rect.collidepoint(pos)
    
    def clicked(self, pos):
        if self.rect.collidepoint(pos):
            if self.toggle:
                self.active = not self.active
            return True
        return False


# ==========================================================
# 3. Visualization
# ==========================================================
def visualize(checkpoints, target_angle=0.0, duration=10.0):
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Aquatic Evolution V17 - Visualizer")
    
    font = pygame.font.SysFont("Arial", 18, bold=True)
    small = pygame.font.SysFont("Arial", 14)
    tiny = pygame.font.SysFont("Arial", 12)
    
    clock = pygame.time.Clock()
    
    target_rad = math.radians(target_angle)
    # Direction in WORLD coordinates (Y-up, CCW angles from X-axis)
    # The coordinate conversion (cam_y - world_y) handles the Y flip for drawing
    direction = np.array([math.cos(target_rad), math.sin(target_rad)])
    
    creature_config = parse_creature_config("Creature.txt")
    base_energy = creature_config.get('base_energy', 100.0)
    size_penalty_factor = creature_config.get('size_penalty_factor', 0.25)
    
    # Progress slider (YouTube-style timeline)
    progress_slider = UIProgressSlider(50, 500, 550, 18, duration, "Timeline")
    
    # Speed slider - now with range 0.1x to 10x for proper fast forward
    speed_slider = UIFloatSlider(620, 565, 80, 18, 0.1, 10.0, 1.0, "Speed", log=True)
    speed_input = UIInputBox(710, 558, 40, 25, "1.0")
    
    # Control buttons
    play_btn = UIButton(50, 558, 60, 32, "PLAY", (40, 120, 80))
    reset_btn = UIButton(120, 558, 60, 32, "RESET", (100, 80, 40))
    
    # Vector display toggle buttons
    thrust_btn = UIButton(200, 558, 70, 32, "THRUST", (80, 60, 120), toggle=True)
    drag_btn = UIButton(280, 558, 70, 32, "DRAG", (120, 60, 80), toggle=True)
    
    # Episode navigation buttons
    prev_ep_btn = UIButton(370, 558, 50, 32, "< Ep", (60, 60, 80))
    next_ep_btn = UIButton(430, 558, 50, 32, "Ep >", (60, 60, 80))
    
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
        
        # Get current checkpoint data
        if checkpoints and ep_idx < len(checkpoints):
            ck = checkpoints[ep_idx]
            trajectory = ck.trajectory
            max_dist = ck.max_distance
            use_nn = ck.use_nn
            curr_num_segments = ck.num_segments
            current_episode = ck.episode
        else:
            trajectory = [{'pos': np.array([0.0, 0.0]), 'segment_angles': np.zeros(5), 
                          'num_segments': 5, 'energy': 100.0, 'base_orientation': 0.0}]
            max_dist = 0
            use_nn = True
            curr_num_segments = 5
            current_episode = 0
        
        # Calculate dt from trajectory
        if trajectory and len(trajectory) > 1:
            sim_dt = duration / len(trajectory)
        else:
            sim_dt = 0.01
        
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            
            if e.type == pygame.MOUSEWHEEL:
                zoom_factor = 1.2 if e.y > 0 else 0.8
                manual_zoom *= zoom_factor
                manual_zoom = max(1.0, min(100.0, manual_zoom))
            
            # Handle progress slider
            if progress_slider.handle_event(e, mouse):
                # User dragged the progress slider - update playback position
                playback_pos = progress_slider.get_frame_index(len(trajectory), sim_dt)
            
            # Handle speed slider
            speed_slider.handle_event(e, mouse)
            
            if speed_input.handle_event(e):
                speed = speed_input.get_value()
                speed_slider.value = max(0.1, min(10.0, speed))
            
            if e.type == pygame.MOUSEBUTTONDOWN:
                if play_btn.clicked(mouse):
                    playing = not playing
                    play_btn.text = "PAUSE" if playing else "PLAY"
                    play_btn.color = (120, 80, 40) if playing else (40, 120, 80)
                
                if reset_btn.clicked(mouse):
                    playback_pos = 0
                    progress_slider.set_time(0)
                
                # Handle vector toggle buttons
                thrust_btn.clicked(mouse)
                drag_btn.clicked(mouse)
                
                # Handle episode navigation buttons
                if prev_ep_btn.clicked(mouse):
                    if ep_idx > 0:
                        ep_idx -= 1
                        playback_pos = 0
                        progress_slider.set_time(0)
                
                if next_ep_btn.clicked(mouse):
                    if ep_idx < len(checkpoints) - 1:
                        ep_idx += 1
                        playback_pos = 0
                        progress_slider.set_time(0)
            
            if speed_slider.dragging:
                speed_input.set_value(speed_slider.value)
        
        if not speed_input.active:
            speed = speed_slider.value
        
        # Update button hover states
        play_btn.check_hover(mouse)
        reset_btn.check_hover(mouse)
        thrust_btn.check_hover(mouse)
        drag_btn.check_hover(mouse)
        prev_ep_btn.check_hover(mouse)
        next_ep_btn.check_hover(mouse)
        
        # Playback with variable speed
        if trajectory:
            base_interval = 0.02  # Base interval at 1x speed
            
            if playing:
                # Calculate how many frames to advance based on speed
                frames_per_update = max(1, int(speed))
                time_per_frame = base_interval / max(speed, 0.1)
                
                if (now - last_upd) >= time_per_frame:
                    last_upd = now
                    playback_pos = min(playback_pos + frames_per_update, len(trajectory) - 1)
                    
                    # Update progress slider
                    current_time = playback_pos * sim_dt
                    progress_slider.set_time(current_time)
        
        # Get current frame data
        if trajectory and playback_pos < len(trajectory):
            curr = trajectory[playback_pos]
            head = np.array(curr['pos']) if not isinstance(curr['pos'], np.ndarray) else curr['pos'].copy()
            curr_energy = curr.get('energy', 100.0)
            curr_segments = curr.get('num_segments', curr_num_segments)
            base_orientation = curr.get('base_orientation', 0.0)
            segment_angles = curr.get('segment_angles', np.zeros(curr_segments))
            if not isinstance(segment_angles, np.ndarray):
                segment_angles = np.array(segment_angles)
        else:
            head = np.array([0.0, 0.0])
            curr_energy = 100.0
            curr_segments = curr_num_segments
            base_orientation = 0.0
            segment_angles = np.zeros(curr_segments)
        start = np.array([0.0, 0.0])
        
        zoom = manual_zoom
        cam_x = (head[0] + 0) / 2
        cam_y = (head[1] + 0) / 2
        
        screen.fill((8, 20, 40))
        
        # ========================================
        # Draw X-axis (GREY horizontal line)
        # ========================================
        x_axis_y_screen = int((cam_y - 0) * zoom + 280)  # Y=0 in world coords (flipped for pygame)
        pygame.draw.line(screen, (100, 100, 100), (0, x_axis_y_screen), (800, x_axis_y_screen), 2)
        
        # X-axis label
        screen.blit(tiny.render("X-axis", True, (120, 120, 120)), (750, x_axis_y_screen + 5))
        
        # ========================================
        # Draw target direction (TEAL line with arrow)
        # ========================================
        # Draw the direction line extending from origin
        line_length = 2000  # Long line to span the screen
        line_start = start - direction * line_length
        line_end = start + direction * line_length
        p1 = (int((line_start[0]-cam_x)*zoom+400), int((cam_y-line_start[1])*zoom+280))
        p2 = (int((line_end[0]-cam_x)*zoom+400), int((cam_y-line_end[1])*zoom+280))
        
        # Teal color for target direction
        teal_color = (0, 180, 180)
        pygame.draw.line(screen, (0, 100, 100), p1, p2, 4)  # Darker outline
        pygame.draw.line(screen, teal_color, p1, p2, 2)  # Teal main line
        
        # Draw arrow at the target direction (from origin pointing in direction)
        arrow_pos = start + direction * 50  # Position arrow near origin (world coords)
        ax = int((arrow_pos[0]-cam_x)*zoom+400)
        ay = int((cam_y-arrow_pos[1])*zoom+280)
        
        # Convert direction to screen space (flip Y) for arrow drawing
        dir_screen = np.array([direction[0], -direction[1]])
        perp_screen = np.array([-dir_screen[1], dir_screen[0]])
        sz = int(15 * zoom / 15)
        
        # Arrow pointing in target direction (screen coords)
        pts = [
            (ax + int(dir_screen[0]*sz*1.5), ay + int(dir_screen[1]*sz*1.5)),  # Tip
            (ax - int(dir_screen[0]*sz*0.5) + int(perp_screen[0]*sz), ay - int(dir_screen[1]*sz*0.5) + int(perp_screen[1]*sz)),
            (ax - int(dir_screen[0]*sz*0.5) - int(perp_screen[0]*sz), ay - int(dir_screen[1]*sz*0.5) - int(perp_screen[1]*sz))
        ]
        pygame.draw.polygon(screen, teal_color, pts)
        pygame.draw.polygon(screen, (0, 220, 220), pts, 2)  # Brighter outline
        
        # Label for target direction
        label_pos = start + direction * 80
        label_x = int((label_pos[0]-cam_x)*zoom+400)
        label_y = int((cam_y-label_pos[1])*zoom+280)
        screen.blit(tiny.render(f"Target: {target_angle:.0f}°", True, teal_color), (label_x + 10, label_y - 8))
        
        # ========================================
        # Draw start position marker
        # ========================================
        cx = int((0-cam_x)*zoom+400)
        cy = int((cam_y-0)*zoom+280)
        sz = int(10 * zoom / 15)
        pygame.draw.line(screen, (255, 50, 50), (cx-sz, cy), (cx+sz, cy), 4)
        pygame.draw.line(screen, (255, 50, 50), (cx, cy-sz), (cx, cy+sz), 4)
        screen.blit(tiny.render("START", True, (255, 100, 100)), (cx+sz+5, cy-8))
        
        # ========================================
        # Draw trajectory
        # ========================================
        if trajectory and playback_pos > 0:
            pts = []
            for i in range(0, playback_pos+1, max(1, len(trajectory)//300)):
                p = trajectory[i]['pos']
                if not isinstance(p, np.ndarray):
                    p = np.array(p)
                px = int((p[0]-cam_x)*zoom+400)
                py = int((cam_y-p[1])*zoom+280)
                pts.append((px, py))
            
            if len(pts) > 1:
                for i in range(1, len(pts)):
                    progress = i / len(pts)
                    c = (int(100+155*progress), 30, 30)
                    pygame.draw.line(screen, c, pts[i-1], pts[i], max(2, int(3*zoom/15)))
        
        # ========================================
        # Draw creature
        # ========================================
        if trajectory and playback_pos < len(trajectory):
            curr = trajectory[playback_pos]
            seg_angles = curr.get('segment_angles', np.zeros(curr_segments))
            if not isinstance(seg_angles, np.ndarray):
                seg_angles = np.array(seg_angles)
            pos = curr['pos']
            if not isinstance(pos, np.ndarray):
                pos = np.array(pos)
            num_seg = curr.get('num_segments', curr_segments)
            base_orient = curr.get('base_orientation', 0.0)
            
            # Build joint positions
            joints = [pos.copy()]
            seg_len = 15.0
            
            for i in range(num_seg):
                seg_world_angle = base_orient + seg_angles[i]
                prev = joints[-1]
                new_x = prev[0] - seg_len * math.cos(seg_world_angle)
                new_y = prev[1] - seg_len * math.sin(seg_world_angle)
                joints.append(np.array([new_x, new_y]))
            
            # Draw segments
            for i in range(len(joints) - 1):
                p1 = (int((joints[i][0]-cam_x)*zoom+400), int((cam_y-joints[i][1])*zoom+280))
                p2 = (int((joints[i+1][0]-cam_x)*zoom+400), int((cam_y-joints[i+1][1])*zoom+280))
                th = max(3, int(10 * zoom / 15))
                
                if i < len(seg_angles):
                    angle_normalized = min(abs(seg_angles[i]) / math.pi, 1.0)
                    c = int(angle_normalized * 200)
                    seg_color = (c, 200-c//2, 255-c//2)
                else:
                    seg_color = (0, 200, 255)
                
                pygame.draw.line(screen, (0, 60, 100), p1, p2, th+2)
                pygame.draw.line(screen, seg_color, p1, p2, th)
                pygame.draw.circle(screen, (150, 255, 255), p1, th//2+2)
            
            # Eyes
            if len(joints) > 1:
                head_angle = base_orient + (seg_angles[0] if len(seg_angles) > 0 else 0)
                head_dir = np.array([math.cos(head_angle), math.sin(head_angle)])
                
                # Convert to screen space (Y flipped)
                off = np.array([head_dir[0], -head_dir[1]]) * int(5*zoom/15)
                perp_screen = np.array([head_dir[1], head_dir[0]]) * int(4*zoom/15)
                hs = (int((joints[0][0]-cam_x)*zoom+400), int((cam_y-joints[0][1])*zoom+280))
                er = max(2, int(4*zoom/15))
                pr = max(1, int(2*zoom/15))
                
                for sgn in [-1, 1]:
                    ex = int(hs[0] + off[0] + perp_screen[0]*sgn)
                    ey = int(hs[1] + off[1] + perp_screen[1]*sgn)
                    pygame.draw.circle(screen, (255, 255, 255), (ex, ey), er)
                    pygame.draw.circle(screen, (20, 20, 40), (ex, ey), pr)
            
            # Draw thrust and drag vectors if enabled
            thrust_vectors = curr.get('thrust_vectors', [])
            drag_vectors = curr.get('drag_vectors', [])
            total_thrust = curr.get('thrust', [0.0, 0.0])
            total_drag = curr.get('drag_force', [0.0, 0.0])
            
            if not isinstance(total_thrust, np.ndarray):
                total_thrust = np.array(total_thrust) if total_thrust else np.zeros(2)
            if not isinstance(total_drag, np.ndarray):
                total_drag = np.array(total_drag) if total_drag else np.zeros(2)
            
            VECTOR_SCALE_FACTOR = 10
            vector_scale = VECTOR_SCALE_FACTOR * zoom / 15
            
            # Draw thrust vectors (CYAN)
            if thrust_btn.active and thrust_vectors:
                for i, tv in enumerate(thrust_vectors):
                    if i < len(joints):
                        joint_pos = joints[i]
                        jx = int((joint_pos[0] - cam_x) * zoom + 400)
                        jy = int((cam_y - joint_pos[1]) * zoom + 280)
                        
                        tv_arr = np.array(tv) if not isinstance(tv, np.ndarray) else tv
                        tv_mag = np.linalg.norm(tv_arr)
                        
                        if tv_mag > 0.01:
                            end_x = jx + int(tv_arr[0] * vector_scale)
                            end_y = jy - int(tv_arr[1] * vector_scale)  # Negate Y for pygame coords
                            pygame.draw.line(screen, (0, 255, 255), (jx, jy), (end_x, end_y), 3)
                            
                            if tv_mag > 0.1:
                                vec_norm = tv_arr / tv_mag
                                # Screen space perpendicular (Y flipped): (vy, vx)
                                perp = np.array([vec_norm[1], vec_norm[0]])
                                arrow_size = 6
                                arrow_x = end_x - int(vec_norm[0] * arrow_size) + int(perp[0] * arrow_size * 0.5)
                                arrow_y = end_y + int(vec_norm[1] * arrow_size) + int(perp[1] * arrow_size * 0.5)
                                arrow_x2 = end_x - int(vec_norm[0] * arrow_size) - int(perp[0] * arrow_size * 0.5)
                                arrow_y2 = end_y + int(vec_norm[1] * arrow_size) - int(perp[1] * arrow_size * 0.5)
                                pygame.draw.polygon(screen, (0, 255, 255), [(end_x, end_y), (arrow_x, arrow_y), (arrow_x2, arrow_y2)])
            
            # Total thrust on head
            if thrust_btn.active and np.linalg.norm(total_thrust) > 0.01:
                thrust_mag = np.linalg.norm(total_thrust)
                head_pos = joints[0]
                hx = int((head_pos[0] - cam_x) * zoom + 400)
                hy = int((cam_y - head_pos[1]) * zoom + 280)
                
                end_x = hx + int(total_thrust[0] * vector_scale * 2)
                end_y = hy - int(total_thrust[1] * vector_scale * 2)  # Negate Y for pygame coords
                
                pygame.draw.line(screen, (0, 200, 255), (hx, hy), (end_x, end_y), 4)
                
                if thrust_mag > 0.1:
                    vec_norm = total_thrust / thrust_mag
                    # Screen space perpendicular (Y flipped): (vy, vx)
                    perp = np.array([vec_norm[1], vec_norm[0]])
                    arrow_size = 10
                    arrow_x = end_x - int(vec_norm[0] * arrow_size) + int(perp[0] * arrow_size * 0.5)
                    arrow_y = end_y + int(vec_norm[1] * arrow_size) + int(perp[1] * arrow_size * 0.5)
                    arrow_x2 = end_x - int(vec_norm[0] * arrow_size) - int(perp[0] * arrow_size * 0.5)
                    arrow_y2 = end_y + int(vec_norm[1] * arrow_size) - int(perp[1] * arrow_size * 0.5)
                    pygame.draw.polygon(screen, (0, 200, 255), [(end_x, end_y), (arrow_x, arrow_y), (arrow_x2, arrow_y2)])
            
            # Draw drag vectors (ORANGE)
            if drag_btn.active and drag_vectors:
                for i, dv in enumerate(drag_vectors):
                    if i < len(joints):
                        joint_pos = joints[i]
                        jx = int((joint_pos[0] - cam_x) * zoom + 400)
                        jy = int((cam_y - joint_pos[1]) * zoom + 280)
                        
                        dv_arr = np.array(dv) if not isinstance(dv, np.ndarray) else dv
                        dv_mag = np.linalg.norm(dv_arr)
                        
                        if dv_mag > 0.01:
                            end_x = jx + int(dv_arr[0] * vector_scale)
                            end_y = jy - int(dv_arr[1] * vector_scale)  # Negate Y for pygame coords
                            pygame.draw.line(screen, (255, 165, 0), (jx, jy), (end_x, end_y), 3)
                            
                            if dv_mag > 0.1:
                                vec_norm = dv_arr / dv_mag
                                # Screen space perpendicular (Y flipped): (vy, vx)
                                perp = np.array([vec_norm[1], vec_norm[0]])
                                arrow_size = 6
                                arrow_x = end_x - int(vec_norm[0] * arrow_size) + int(perp[0] * arrow_size * 0.5)
                                arrow_y = end_y + int(vec_norm[1] * arrow_size) + int(perp[1] * arrow_size * 0.5)
                                arrow_x2 = end_x - int(vec_norm[0] * arrow_size) - int(perp[0] * arrow_size * 0.5)
                                arrow_y2 = end_y + int(vec_norm[1] * arrow_size) - int(perp[1] * arrow_size * 0.5)
                                pygame.draw.polygon(screen, (255, 165, 0), [(end_x, end_y), (arrow_x, arrow_y), (arrow_x2, arrow_y2)])
            
            # Total drag on head
            if drag_btn.active and np.linalg.norm(total_drag) > 0.01:
                drag_mag = np.linalg.norm(total_drag)
                head_pos = joints[0]
                hx = int((head_pos[0] - cam_x) * zoom + 400)
                hy = int((cam_y - head_pos[1]) * zoom + 280)
                
                end_x = hx + int(total_drag[0] * vector_scale * 2)
                end_y = hy - int(total_drag[1] * vector_scale * 2)  # Negate Y for pygame coords
                
                pygame.draw.line(screen, (255, 100, 50), (hx, hy), (end_x, end_y), 4)
                
                if drag_mag > 0.1:
                    vec_norm = total_drag / drag_mag
                    # Screen space perpendicular (Y flipped): (vy, vx)
                    perp = np.array([vec_norm[1], vec_norm[0]])
                    arrow_size = 10
                    arrow_x = end_x - int(vec_norm[0] * arrow_size) + int(perp[0] * arrow_size * 0.5)
                    arrow_y = end_y + int(vec_norm[1] * arrow_size) + int(perp[1] * arrow_size * 0.5)
                    arrow_x2 = end_x - int(vec_norm[0] * arrow_size) - int(perp[0] * arrow_size * 0.5)
                    arrow_y2 = end_y + int(vec_norm[1] * arrow_size) - int(perp[1] * arrow_size * 0.5)
                    pygame.draw.polygon(screen, (255, 100, 50), [(end_x, end_y), (arrow_x, arrow_y), (arrow_x2, arrow_y2)])
        
        # ========================================
        # UI Panel (bottom)
        # ========================================
        pygame.draw.rect(screen, (15, 25, 45), pygame.Rect(0, 530, 800, 70))
        pygame.draw.line(screen, (50, 70, 100), (0, 530), (800, 530), 2)
        
        # Progress slider (YouTube-style timeline)
        progress_slider.draw(screen, small)
        
        # Speed slider
        speed_slider.draw(screen, small)
        speed_input.draw(screen)
        
        # Buttons
        play_btn.draw(screen, font)
        reset_btn.draw(screen, font)
        thrust_btn.draw(screen, small)
        drag_btn.draw(screen, small)
        prev_ep_btn.draw(screen, small)
        next_ep_btn.draw(screen, small)
        
        # ========================================
        # Status indicators (above timeline)
        # ========================================
        # Episode indicator
        ep_color = (100, 255, 150)
        screen.blit(tiny.render(f"Episode: {current_episode}", True, ep_color), (50, 515))
        
        # Total episodes indicator
        total_episodes = checkpoints[-1].episode if checkpoints else 0
        screen.blit(tiny.render(f"/ {total_episodes}", True, (150, 150, 150)), (140, 515))
        
        # Speed indicator
        speed_color = (100, 255, 100) if speed >= 1 else (255, 200, 100)
        screen.blit(tiny.render(f"Speed: {speed:.1f}x", True, speed_color), (200, 515))
        
        # Zoom indicator
        screen.blit(tiny.render(f"Zoom: {zoom:.1f}x (scroll)", True, (150, 150, 150)), (300, 515))
        
        # ========================================
        # Info panel (top left)
        # ========================================
        info_s = pygame.Surface((180, 115), pygame.SRCALPHA)
        pygame.draw.rect(info_s, (15, 25, 45, 220), info_s.get_rect(), border_radius=8)
        screen.blit(info_s, (10, 10))
        
        delta = head - start
        dist = np.dot(delta, direction)
        screen.blit(font.render(f"DIST: {dist:.1f}m", True, (255, 255, 255)), (20, 15))
        screen.blit(font.render(f"SEGS: {curr_segments}", True, (0, 220, 255)), (20, 38))
        
        method = "NN (Efficient)" if use_nn else "Tabular"
        screen.blit(small.render(f"Method: {method}", True, (200, 200, 200)), (20, 58))
        
        max_energy = base_energy * (1.0 + (curr_segments - 2) * size_penalty_factor)
        e_pct = curr_energy / max(max_energy, 1.0)
        e_color = (100, 255, 100) if e_pct > 0.5 else (255, 200, 50) if e_pct > 0.25 else (255, 80, 80)
        pygame.draw.rect(screen, (40, 40, 60), (20, 78, 140, 12), border_radius=3)
        pygame.draw.rect(screen, e_color, (20, 78, int(140*min(1, e_pct)), 12), border_radius=3)
        screen.blit(tiny.render(f"Energy: {int(curr_energy):.0f}/{int(max_energy):.0f}", True, (255, 255, 255)), (25, 78))
        
        progress = playback_pos / max(1, len(trajectory)-1) if trajectory else 0
        t_curr = progress * duration
        screen.blit(tiny.render(f"Time: {t_curr:.1f}s / {duration:.0f}s", True, (180, 180, 180)), (20, 95))
        
        # Checkpoint info (top right)
        if checkpoints and ep_idx < len(checkpoints):
            ck = checkpoints[ep_idx]
            info2 = pygame.Surface((160, 55), pygame.SRCALPHA)
            pygame.draw.rect(info2, (15, 25, 45, 220), info2.get_rect(), border_radius=8)
            screen.blit(info2, (630, 10))
            screen.blit(font.render(f"Ep: {ck.episode}", True, (100, 255, 150)), (640, 15))
            screen.blit(small.render(f"Best: {ck.max_distance:.1f}m", True, (200, 200, 200)), (640, 40))
        
        # Vector legend
        if thrust_btn.active or drag_btn.active:
            legend_y = 130
            legend_height = 50 if (thrust_btn.active and drag_btn.active) else 30
            legend_s = pygame.Surface((140, legend_height), pygame.SRCALPHA)
            pygame.draw.rect(legend_s, (15, 25, 45, 220), legend_s.get_rect(), border_radius=8)
            screen.blit(legend_s, (10, legend_y))
            
            if thrust_btn.active:
                pygame.draw.line(screen, (0, 255, 255), (20, legend_y + 15), (50, legend_y + 15), 3)
                screen.blit(tiny.render("Thrust vectors", True, (0, 255, 255)), (55, legend_y + 10))
            
            if drag_btn.active:
                drag_legend_y = legend_y + (25 if thrust_btn.active else 0)
                pygame.draw.line(screen, (255, 165, 0), (20, drag_legend_y + 15), (50, drag_legend_y + 15), 3)
                screen.blit(tiny.render("Drag vectors", True, (255, 165, 0)), (55, drag_legend_y + 10))
        
        # Controls hint
        if not playing:
            screen.blit(tiny.render("PLAY/PAUSE | Scroll to zoom | Toggle THRUST/DRAG for vectors", True, (100, 120, 160)), (350, 515))
        
        screen.blit(tiny.render(f"Angles: {curr_segments} (per-segment)", True, (80, 100, 140)), (10, 595))
        
        # Version info
        screen.blit(tiny.render("V17 - Efficient NN + Load/Continue Training", True, (60, 80, 120)), (500, 595))
        
        # Legend for axes
        screen.blit(tiny.render("Grey: X-axis | Teal: Target direction", True, (80, 80, 80)), (560, 25))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()


# ==========================================================
# 4. Loader
# ==========================================================
def list_saved_simulations(script_dir):
    """List saved simulations from the Result folder in the script directory."""
    result_dir = os.path.join(script_dir, "Result")
    if not os.path.exists(result_dir):
        return []
    
    files = []
    for f in os.listdir(result_dir):
        if f.startswith('simulation_') and f.endswith('.pkl'):
            files.append(os.path.join(result_dir, f))
    
    return sorted(files, reverse=True)


def load_simulation(filename):
    """Load simulation from file and convert to TrainingCheckpoint objects."""
    with open(filename, 'rb') as f:
        save_data = pickle.load(f)
    
    checkpoints = []
    for cp_dict in save_data['checkpoints']:
        trajectory = []
        for frame in cp_dict['trajectory']:
            new_frame = {}
            for key, value in frame.items():
                if key == 'pos' and not isinstance(value, np.ndarray):
                    new_frame[key] = np.array(value)
                elif key == 'segment_angles' and not isinstance(value, np.ndarray):
                    new_frame[key] = np.array(value)
                elif key in ['thrust_vectors', 'drag_vectors'] and value:
                    new_frame[key] = [np.array(v) if not isinstance(v, np.ndarray) else v for v in value]
                elif key in ['thrust', 'drag_force'] and value:
                    new_frame[key] = np.array(value) if not isinstance(value, np.ndarray) else value
                else:
                    new_frame[key] = value
            trajectory.append(new_frame)
        
        checkpoint = TrainingCheckpoint(
            episode=cp_dict['episode'],
            model_state=cp_dict['model_state'],
            trajectory=trajectory,
            max_distance=cp_dict['max_distance'],
            use_nn=cp_dict['use_nn'],
            num_segments=cp_dict['num_segments']
        )
        checkpoints.append(checkpoint)
    
    return (checkpoints, save_data['use_nn'], 
            save_data['target_angle'], save_data['duration'],
            save_data.get('num_segments', 5), save_data.get('fixed_segments', True))


# ==========================================================
# 5. Main
# ==========================================================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\n" + "=" * 60)
    print("   AQUATIC CREATURE VISUALIZER - V17")
    print("   (YouTube-style controls + Episode navigation)")
    print("=" * 60)
    
    saved_files = list_saved_simulations(script_dir)
    
    if not saved_files:
        print("No saved simulations found in 'Result/' folder.")
        print("Please run 'aquatic_evolution_qlearning_V17.py' first.")
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
            import traceback
            traceback.print_exc()
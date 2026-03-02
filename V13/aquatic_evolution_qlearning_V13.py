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
#       PHYSICS MODEL CLASS
# ==========================================================

def parse_physics_model(filepath):

    DEFAULT = {"thrust_coefficient": 3.5, "angle_speed": 0.4, "drag_spread_factor": 2.0}
    
    config = {}
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                
                if "thrust_coefficient" in line and "=" in line:
                    config["thrust_coefficient"] = float(line.split("=")[1].strip())
                elif "angle_speed" in line and "=" in line:
                    config["angle_speed"] = float(line.split("=")[1].strip())
                elif "drag_spread_factor" in line and "=" in line:
                    config["drag_spread_factor"] = float(line.split("=")[1].strip())
            return config
        
    except FileNotFoundError:
        print("Configuration file 'PhysicsModel.txt' not found. Using default parameters.")
        return DEFAULT

class PhysicsModel:

    """
    Aquatic propulsion physics model.
    Handles thrust generation from segment rotation and drag calculation.
    """

    def __init__(self, thrust_coefficient=3.5, 
                 angle_speed=0.4, drag_spread_factor=2.0):
        
        self.angle_speed        = angle_speed
        self.thrust_coefficient = thrust_coefficient
        self.drag_spread_factor = drag_spread_factor

    def _apply_drag(self, creature, dt):
        """Calculate and apply drag based on segment orientations."""
        
        # This angle is the one of the velocity vector w.r.t. the x axis
        vel_angle = math.atan2(creature.head_vel[1], creature.head_vel[0])
        
        # We compute the sum of the perpendicular projections of all segments
        total_projected_area = 0.0
        for i in range(creature.num_segments):

            # This angle is the CCW orientation of the segment w.r.t. the x axis
            seg_world_angle = creature.base_orientation + creature.segment_angles[i]

            # We compute the section perpendicular to the velocity vector
            angle_to_velocity = seg_world_angle - vel_angle
            projected_area = abs(math.sin(angle_to_velocity)) * creature.segment_length

            # The tail segments are less affected by the drag
            total_projected_area += projected_area * ( (creature.num_segments - i) / creature.num_segments)
        
        # This step allows to have a drag coefficient that doesn't depend on the length of the segments
        normalized_area = total_projected_area / creature.segment_length
        current_drag_coeff = self.drag_spread_factor * normalized_area
        
        # The drag is proportional to coeff * speed^2 in the opposite direction of the velocity
        # Thus, drag = - coeff * speed^2 * normalized_velocity_vector
        # Thus, drag = - coeff * speed * velocity_vecotr
        drag = -current_drag_coeff * creature.head_vel * np.linalg.norm(creature.head_vel)

        # We apply the drag to slow down the progression of the creature
        creature.head_vel = creature.head_vel + drag * dt

    def _apply_thrust(self, creature, old_angles, dt):
        """Calculate and apply thrust based on segment orientations."""

        thrust = np.array([0.0, 0.0])
        
        for i in range(creature.num_segments):

            # Each angle change will generate some thrust
            angle_change = creature.segment_angles[i] - old_angles[i]
            if abs(angle_change) < 1e-8:
                continue
            
            # The swept area of fluid = (1/2) × L^2 × angular_variation
            # This multiplied by a thrust coeff provides the magnitude of the thrust due to this angle variation
            swept_area = 0.5 * abs(angle_change) * (creature.segment_length)**2
            thrust_magnitude = self.thrust_coefficient * swept_area
            
            # This angle is the CCW orientation of the segment w.r.t. the x axis
            seg_world_angle = creature.base_orientation + creature.segment_angles[i]
            
            # Thrust is perpendicular to the movement of the segment
            # Thus, we use the final position of the segment as reference
            if angle_change > 0:
                thrust_angle = seg_world_angle + math.pi / 2
            else:
                thrust_angle = seg_world_angle - math.pi / 2
            
            # We sum the thrust of each angular change
            thrust_dir = np.array([math.cos(thrust_angle), math.sin(thrust_angle)])
            thrust += thrust_magnitude * thrust_dir
        
        # We apply the thrust to modify the progression of the creature
        creature.head_vel = creature.head_vel + thrust * dt

    def update_physics(self, creature, dt):
        """Update creature's position and velocity based on physics."""

        # Store old angles for thrust calculation
        old_angles = creature.segment_angles.copy()
        
        # Update angles towards target
        for i in range(creature.num_segments):
            diff = creature.target_angles[i] - creature.segment_angles[i]
            creature.segment_angles[i] += diff * self.angle_speed
        
        # Apply thrust from angle changes (modifies the head velocity)
        self._apply_thrust(creature, old_angles, dt)
        
        # Apply drag (modifies the head velocity)
        self._apply_drag(creature, dt)
        
        # Update position
        creature.head_pos = creature.head_pos + creature.head_vel * dt

# ==========================================================
#       CREATURE CLASS
# ==========================================================

def parse_creature(filepath):

    DEFAULT = {"segment_length": 3.0, "size_penalty_factor": 0.25,
               "fixed_segments": True, "num_segments": 2,
               "min_segments": 2, "max_segments": 10,
               "target_angle_deg": 0, "max_angle_deg": 180.0,
               "use_nn": False,
               "base_energy": 100.0, "metabolism_rate": 0.002,
               "energy_cost_angle_move": 0.03,
               "energy_cost_add_segment": 3.0, "energy_cost_remove_segment": 1.5,
               "angle_step": 10}
    
    config = {}
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                
                if "segment_length" in line and "=" in line:
                    config["segment_length"] = float(line.split("=")[1].strip())
                elif "size_penalty_factor" in line and "=" in line:
                    config["size_penalty_factor"] = float(line.split("=")[1].strip())
                elif "fixed_segments" in line and "=" in line:
                    config["fixed_segments"] = True if "true" in line.split("=")[1].strip().lower() else False
                elif "num_segments" in line and "=" in line:
                    config["num_segments"] = int(line.split("=")[1].strip())
                elif "min_segments" in line and "=" in line:
                    config["min_segments"] = int(line.split("=")[1].strip())
                elif "max_segments" in line and "=" in line:
                    config["max_segments"] = int(line.split("=")[1].strip())
                elif "target_angle_deg" in line and "=" in line:
                    config["target_angle_deg"] = float(line.split("=")[1].strip())
                elif "max_angle_deg" in line and "=" in line:
                    config["max_angle_deg"] = float(line.split("=")[1].strip())
                elif "use_nn" in line and "=" in line:
                    config["use_nn"] = True if "true" in line.split("=")[1].strip().lower() else False
                elif "base_energy" in line and "=" in line:
                    config["base_energy"] = float(line.split("=")[1].strip())
                elif "metabolism_rate" in line and "=" in line:
                    config["metabolism_rate"] = float(line.split("=")[1].strip())
                elif "energy_cost_angle_move" in line and "=" in line:
                    config["energy_cost_angle_move"] = float(line.split("=")[1].strip())
                elif "energy_cost_add_segment" in line and "=" in line:
                    config["energy_cost_add_segment"] = float(line.split("=")[1].strip())
                elif "energy_cost_remove_segment" in line and "=" in line:
                    config["energy_cost_remove_segment"] = float(line.split("=")[1].strip())
                elif "angle_step" in line and "=" in line:
                    config["angle_step"] = float(line.split("=")[1].strip())
            return config
        
    except FileNotFoundError:
        print("Configuration file 'Creature.txt' not found. Using default parameters.")
        return DEFAULT

class Creature:

    """
    Aquatic creature with segment-based body.
    The creature has multiple body segments that can rotate independently.
    Energy is consumed for movement and metabolism.
    """

    def __init__(self,
                 segment_length=3.0, size_penalty_factor=0.25,
                 fixed_segments=True, num_segments=2,
                 min_segments=2, max_segments=10,
                 target_angle_deg=0, max_angle_deg=180.0,
                 use_nn=True,
                 base_energy=100.0, metabolism_rate=0.002,
                 energy_cost_angle_move=0.03,
                 energy_cost_add_segment=3.0, energy_cost_remove_segment=1.5,
                 angle_step=10):
        
        # Body configuration
        self.segment_length      = float(segment_length)       # Segment length
        self.size_penalty_factor = float(size_penalty_factor)  # Factor for energy consumption
        
        # Number of segment
        self.fixed_segments = bool(fixed_segments) # Can the creature modify the number of segments?
        self.num_segments   = int(num_segments)    # The current number of segments
        self.min_segments   = int(min_segments)    # The minimal number of segments
        self.max_segments   = int(max_segments)    # The maximal number of segments

        # Initialize the current number of segments if the number of segments is not fixed
        if not fixed_segments:
            self.num_segments = int(min_segments)
        
        # Movement configuration
        self.target_angle_deg = float(target_angle_deg)                     # The desired direction of progression in °
        self.base_orientation = math.radians(target_angle_deg)              # The equivalent in radians
        self.direction_vector = np.array([math.cos(self.base_orientation),
                                          math.sin(self.base_orientation)]) # Unitary direction vector for desired movement
        
        # Learning configuration
        self.use_nn = bool(use_nn)    # Can the creature learn by using a neural network?
        
        # Energy configuration
        self.base_energy = float(base_energy) # Base level of energy
        self.max_energy  = float(base_energy) # Maximal level of energy (will depend on the size)
        self.energy      = float(base_energy) # Current level of energy

        self.metabolism_rate            = float(metabolism_rate)               # Energy cost to stay alive
        self.energy_cost_angle_move     = float(energy_cost_angle_move)        # Energy cost to rotate a segment
        self.energy_cost_add_segment    = float(energy_cost_add_segment)       # Energy cost to add a segment
        self.energy_cost_remove_segment = float(energy_cost_remove_segment)    # Energy cost to remove a segment
        
        # Segment angles: orientation of each segment relative to base_orientation
        self.max_angle      = math.radians(float(max_angle_deg))             # The maximal angle of the segment w.r.t. base_orientation
        self.segment_angles = np.zeros(int(max_segments), dtype=np.float64)  # The current values of the angles
        self.target_angles  = np.zeros(int(max_segments), dtype=np.float64)  # The current values of the angles after chosen action

        # Angle delta for a movement
        self.angle_step_deg = angle_step                # Angle delta for a movement in °
        self.angle_step_rad = math.radians(angle_step)  # Equivalent in radians
        
        # Position and velocity
        self.head_pos = np.array([0.0, 0.0], dtype=np.float64)  # Position of the head of the creature
        self.head_vel = np.array([0.0, 0.0], dtype=np.float64)  # Velocity vector of the head
        
        # Starting position tracking
        self.start_pos = np.array([0.0, 0.0], dtype=np.float64)

    def reset_creature(self, x=0.0, y=0.0, orientation=None):
        """Reset the creature for a new episode."""

        # Reset orientation
        self.base_orientation = orientation if orientation is not None else self.base_orientation
        
        # Straight body: all segments aligned with base_orientation
        self.segment_angles[:self.num_segments] = 0.0
        self.target_angles[:self.num_segments]  = 0.0
        
        # Reset position and velocity
        self.head_pos = np.array([x, y], dtype=np.float64)
        self.head_vel = np.array([0.0, 0.0], dtype=np.float64)
        
        # Track starting position
        self.start_pos = np.array([x, y], dtype=np.float64)
        
        # Reset energy level
        self.energy = self.max_energy

        # The max energy level depends on the current size of the creature
        self.max_energy = self.base_energy * self.get_body_size_factor()

    def get_body_size_factor(self):
        """Calculate energy multiplier based on body size."""
        return 1.0 + (self.num_segments - 2) * self.size_penalty_factor
    
    def consume_energy_metabolism(self):
        """Consume energy for staying alive."""

        # Compute the energy cost for staying alive
        size_factor = self.get_body_size_factor()
        energy_cost = self.metabolism_rate * size_factor

        # Ensures that the creature still has enough energy to stay alive
        if self.energy - energy_cost < 0:
            self.energy = 0 # If not, it dies (energy to 0)
            return [False]

        # If it's the case, decrease the level of energy
        self.energy -= - energy_cost

        return [True, energy_cost]
    
    def consume_energy_angle_move(self, angle_delta):
        """Consume energy for changing segment orientation."""

        # Compute the energy cost to make the move
        size_factor = self.get_body_size_factor()
        energy_cost = self.energy_cost_angle_move * abs(angle_delta) * size_factor
        
        # Ensures that the creature still has enough energy to make the move
        if self.energy - energy_cost < 0:
            return [False]
        
        # If it's the case, decrease the energy level of the creature
        self.energy -= energy_cost

        return [True, energy_cost]
    
    def add_segment(self):
        """Add a new segment to the creature if possible."""

        # Compute the energy cost to add a segment
        size_factor = self.get_body_size_factor()
        energy_cost = self.energy_cost_add_segment * size_factor
        
        # Ensures that we don't exceed the maximal number of segments
        if self.num_segments >= self.max_segments:
            return [False]
        
        # And that the creature still has enough energy to add the segment
        if self.energy - energy_cost < 0:
            return [False]
        
        # If it's the case,
        self.energy       -= energy_cost # Decrease the energy level of the creature
        self.num_segments += 1           # Increase the number of segments

        # Update max energy based on new body size and ensure current energy does not exceed new max
        self.max_energy = self.base_energy * self.get_body_size_factor()
        self.energy     = min(self.energy, self.max_energy)
        
        # Initialize the orientation variables for the new segment
        self.segment_angles[self.num_segments - 1] = 0.0
        self.target_angles[self.num_segments - 1]  = 0.0
        
        return [True, energy_cost]
    
    def remove_segment(self):
        """Remove a segment from the creature if possible."""

        # Compute the energy cost to remove a segment
        size_factor = self.get_body_size_factor()
        energy_cost = self.energy_cost_remove_segment * size_factor
        
        # Ensures that we don't go below the minimal number of segments
        if self.num_segments <= self.min_segments:
            return [False]
        
        # And that the creature still has enough energy to remove the segment
        if self.energy - energy_cost < 0:
            return [False]
        
        # If it's the case,
        self.energy       -= energy_cost # Decrease the energy level of the creature
        self.num_segments -= 1           # Decrease the number of segments

        # Update max energy based on new body size and ensure current energy does not exceed new max
        self.max_energy = self.base_energy * self.get_body_size_factor()
        self.energy     = min(self.energy, self.max_energy)

        # Pops the last segment angles (not really necessary since we track num_segments, but for cleanliness)
        self.segment_angles = self.segment_angles[:self.num_segments]
        self.target_angles  = self.target_angles[:self.num_segments]

        return [True, energy_cost]
    
    def get_trajectory_frame(self):
        """Get current state as a dictionary for trajectory recording."""

        return {'pos': self.head_pos.copy(),                                        # The current position of the creature's head in 2D space.
                'segment_angles': self.segment_angles[:self.num_segments].copy(),   # The angles of the segments relative to the base orientation.
                'num_segments': self.num_segments,                                  # The current number of segments the creature has.
                'energy': self.energy,                                              # The current energy level of the creature.
                'base_orientation': self.base_orientation}                          # The base orientation of the creature.

# ==========================================================
#       LEARNING PROCESS CLASS
# ==========================================================

def parse_learning_process(filepath):

    DEFAULT = {"use_nn": False, "learning_rate_nn": 0.001, "learning_rate_tabular": 0.15, "gamma": 0.9,
               "coeff_dist": 0.05, "coeff_vel": 0.5, "coeff_energy": 0.005,
               "epsilon_min": 0.05}
    
    config = {}
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                
                if "use_nn" in line and "=" in line:
                    config["use_nn"] = True if "true" in line.split("=")[1].strip().lower() else False
                elif "learning_rate_nn" in line and "=" in line:
                    config["learning_rate_nn"] = float(line.split("=")[1].strip())
                elif "learning_rate_tabular" in line and "=" in line:
                    config["learning_rate_tabular"] = float(line.split("=")[1].strip())
                elif "gamma" in line and "=" in line:
                    config["gamma"] = float(line.split("=")[1].strip())
                elif "coeff_dist" in line and "=" in line:
                    config["coeff_dist"] = float(line.split("=")[1].strip())
                elif "coeff_vel" in line and "=" in line:
                    config["coeff_vel"] = float(line.split("=")[1].strip())
                elif "coeff_energy" in line and "=" in line:
                    config["coeff_energy"] = float(line.split("=")[1].strip())
                elif "epsilon_min" in line and "=" in line:
                    config["epsilon_min"] = float(line.split("=")[1].strip())
            return config
        
    except FileNotFoundError:
        print("Configuration file 'LearningProcess.txt' not found. Using default parameters.")
        return DEFAULT

class QTabular:
    """Tabular Q-learning with state discretization."""

    def __init__(self, 
                 state_dim, action_dim,
                 learning_rate=0.15):

        # Dimensions
        self.state_dim  = state_dim         # Current state is the orientation of the segments: should be creature.num_segments
        self.action_dim = action_dim        # Current actions are opening/closing/holding each angle (+add/remove a segment): should be 3 * creature.num_segments (+2)

        self.q_table = {}                   # Q-table that associates each state to a set of actions
        self.learning_rate = learning_rate  # learning rate in the Q-learning update rule

    def get_q_values(self, state):
        """Retrieve the q-values as a table."""

        state = tuple(state)

        # Add the state in the table if not already in
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_dim)
        
        # Return the table
        return self.q_table[state]
    
    def update_q_values(self, state, action, target):
        """Update the q-values using the Q-learning update rule."""

        state = tuple(state)

        # Add the state in the table if not already in
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_dim)

        # Q(state, action) = Q(state, action) + learning_rate * (target - Q(state, action))
        self.q_table[state][action] = self.q_table[state][action] + self.learning_rate * (target - self.q_table[state][action])

class QNetwork(nn.Module):
    """Neural network for Q-value approximation."""
    
    def __init__(self, state_dim, action_dim):

        super(QNetwork, self).__init__()
        
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
        
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def get_q_values(self, state):
        """Retrieve the q-values as a tensor."""

        with torch.no_grad():
            state_tensor = torch.from_numpy(state) if isinstance(state, np.ndarray) else state
            return self.network(state_tensor).numpy()
        
    def update_q_values(self, state, action, target):
        """Update the q-values using ADAM optimizer."""

        with torch.no_grad():

            # Computes (target - Q(state, action))
            current_q = self.model(torch.from_numpy(state))[action]
            loss = nn.MSELoss()(current_q, torch.tensor(target, dtype=torch.float32))
            
            # Backward propagation
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Optimizer doing its job based on the difference
            self.optimizer.step()
            
            # Return this difference
            return loss.item()

class LearningProcess:

    """
    Q-learning implementation supporting both neural network and tabular approaches.
    """

    def __init__(self,
                 state_dim, action_dim,
                 use_nn=False, 
                 learning_rate_nn=0.001, learning_rate_tabular = 0.15,
                 gamma = 0.95,
                 coeff_dist=0.05, coeff_vel=0.5, coeff_energy=0.005,
                 epsilon_min=0.05):
        
        # Dimensions
        self.state_dim  = state_dim         # Current state is the orientation of the segments: should be creature.num_segments
        self.action_dim = action_dim        # Current actions are opening/closing/holding each angle (+add/remove a segment): should be 3 * creature.num_segments (+2)

        # Whether to use or not a neural network
        self.use_neural_network = use_nn

        # If yes, initializes the Neural Network and use ADAM as optimizer
        if use_nn:
            self.model     = QNetwork(state_dim, action_dim)
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate_nn)

        # Otherwise, initializes the Tabular model
        else:
            self.model     = QTabular(state_dim, action_dim, learning_rate=learning_rate_tabular)
            self.optimizer = None

        # Learning rates
        self.learning_rate_nn      = learning_rate_nn
        self.learning_rate_tabular = learning_rate_tabular

        # Discount factor
        self.gamma = gamma

        # Coefficients for objective function
        self.coeff_dist   = coeff_dist
        self.coeff_vel    = coeff_vel
        self.coeff_energy = coeff_energy

        # Minimum epsilon for epsilon-greedy strategy
        self.epsilon_min = epsilon_min
    
    def update_dimensions(self, new_state_dim, new_action_dim):
        """Update model dimensions (for variable segment count)."""

        if new_state_dim != self.state_dim or new_action_dim != self.action_dim:

            # Update the dimensions
            self.state_dim  = new_state_dim
            self.action_dim = new_action_dim
            
            # Initializes the model based on those new dimensions
            if self.use_neural_network:
                self.model     = QNetwork(new_state_dim, new_action_dim)
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate_nn)
            else:
                self.model     = QTabular(new_state_dim, new_action_dim, learning_rate=self.learning_rate_tabular)
                self.optimizer = None

    def update_q_values(self, state, action, cost, next_state):
        """Update Q-values based on experience.

           The formula is: Q(S,A) ← Q(S,A) + learning_rate * [ (C + gamma * min_A' Q(S',A') ) - Q(S,A) ]
           where:
                - S/A are the current state/action
                - S' is the next state the agent moves to
                - A' is the best next action in state S'
                - gamma is the discount factor
                - C is the cost received for taking action A in state S
        """

        # If we use a neural network...
        if self.use_neural_network:
            with torch.no_grad():
                q_next = self.model.get_q_values(next_state)                      # Retrieve the next states achievable
                target = cost + self.gamma * torch.min(q_next).item()             # Define target = (C + gamma * min_A' Q(S',A') )
            self.model.update_q_values(state=state, action=action, target=target) # Update the Q-values

        # Otherwise...
        else:
            q_next = self.model.get_q_values(next_state)                          # Retrieve the next states achievable
            target = cost + self.gamma * np.min(q_next)                           # Define target = (C + gamma * min_A' Q(S',A') )
            self.model.update_q_values(state=state, action=action, target=target) # Update the Q-values

    def calculate_reward(self, creature, energy_cost):
        """Calculate reward based on creature state."""

        # Retrieve the current distance wandered from the start along the desired direction,
        # Based on the current head position
        distance = np.dot(creature.head_pos - creature.start_pos, creature.direction_vector)

        # Retrieve the future distance wandered in addition along the desired direction,
        # Based on the current head velocity
        speed     = np.linalg.norm(creature.head_vel)
        alignment = np.dot(creature.head_vel / (speed+1e-8), creature.direction_vector)

        # Store those information
        info = {'distance': distance,'speed': speed,'alignment': alignment}
        
        # Calculate reward components
        reward_speed    = alignment * speed * self.coeff_vel
        reward_distance = distance * self.coeff_dist
        penalty_energy  = energy_cost * self.coeff_energy
        reward          = reward_speed + reward_distance - penalty_energy
        
        return reward, info
    
    def select_action(self, state, epsilon):
        """Select action using epsilon-greedy strategy."""

        # Generate a random value between 0 and 1,
        # if it's less than epsilon, we explore (random action)
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        # Otherwise,
        # we exploit the learned values by choosing the action with the lowest Q-value for the current state
        else:

            # If we use a neural network...
            if self.use_neural_network:
                with torch.no_grad():
                    q_values = self.model.get_q_values(state)
                    return torch.argmin(q_values).item()
            
            # Otherwise...
            else:
                q_values = self.model.get_q_values(state)
                return int(np.argmin(q_values))
            
    def get_epsilon(self, episode, total_episodes):
        """Calculate current epsilon with linear decay."""
        return max(self.epsilon_min, 1.0 - (episode / total_episodes) * (1.0 - self.epsilon_min))
    
    def get_model_state(self):
        """Get current model state for checkpointing."""
        if self.use_neural_network:
            return self.model.state_dict()
        else:
            return self.model.q_table.copy()
    
# ==========================================================
#       TRAINING CHECKPOINT CLASS
# ==========================================================

class TrainingCheckpoint:
    """Checkpoint container for saving training state."""
    
    def __init__(self, episode, model_state, trajectory, max_distance, use_nn, num_segments):

        self.episode      = episode                     # Current episode number
        self.model_state  = copy.deepcopy(model_state)  # State of the learning model (Q-table or neural network weights)
        self.trajectory   = copy.deepcopy(trajectory)   # Trajectory of the creature for this episode
        self.max_distance = max_distance                # Max distance reached in this episode
        self.use_nn       = use_nn                      # Whether the model is a neural network or tabular model
        self.num_segments = num_segments                # Current number of segments of the creature

# ==========================================================
#       SIMULATION CLASS
# ==========================================================

def parse_simulation(filepath):
        
    DEFAULT = {"min_segments" : 2, "max_segments" : 10,
               "duration_simulation" : 30.0, "dt" : 0.01, "substeps" : 5,
               "episodes" : 300, "checkpoint_interval" : 20}
    
    config = {}
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                
                if "min_segments" in line and "=" in line:
                    config["min_segments"] = int(line.split("=")[1].strip())
                elif "max_segments" in line and "=" in line:
                    config["max_segments"] = int(line.split("=")[1].strip())
                elif "duration_simulation" in line and "=" in line:
                    config["duration_simulation"] = float(line.split("=")[1].strip())
                elif "dt" in line and "=" in line:
                    config["dt"] = float(line.split("=")[1].strip())
                elif "substeps" in line and "=" in line:
                    config["substeps"] = int(line.split("=")[1].strip())
                elif "episodes" in line and "=" in line:
                    config["episodes"] = int(line.split("=")[1].strip())
                elif "checkpoint_interval" in line and "=" in line:
                    config["checkpoint_interval"] = int(line.split("=")[1].strip())

            return config

    except FileNotFoundError:
        print(f"Configuration file '{filepath}' not found. Using default parameters.")
        return DEFAULT

class Simulation:
    """Main simulation class that orchestrates creature, physics, and learning."""
    
    def __init__(self,
                 min_segments=2, max_segments=10,
                 duration_simulation=30.0, dt=0.01, substeps=5,
                 episodes=300, checkpoint_interval=20):
        
        # Min/Max segments for variable number
        self.min_segments = int(min_segments)
        self.max_segments = int(max_segments)

        # Episode parameters
        self.episodes            = episodes              # Number of episodes
        self.checkpoint_interval = checkpoint_interval   # Save every checkpoint_interval for the visualization tool

        # Duration parameter
        self.dt                  = float(dt)                     # Time step for each update in sec
        self.duration_simulation = float(duration_simulation)    # Duration of the simulation in sec
        self.max_steps           = int(duration_simulation / dt) # Total number of update
        self.substeps            = int(substeps)                 # Substeps for physics update
        
        # Initialize the creature             
        config = parse_creature("Creature.txt")
        self.creature = Creature(**config)

        # Initialize the physics model
        config = parse_physics_model("PhysicsModel.txt") 
        self.physics_model = PhysicsModel(**config)    

        # Initialize the learning process
        config = parse_learning_process("LearningProcess.txt") 
        self.learning_process = LearningProcess(state_dim=self.creature.num_segments,
                                                action_dim= 3 * self.creature.num_segments if self.creature.fixed_segments else 3 * self.creature.num_segments + 2,
                                                **config)

        # Ensures that the Min/Max numbers of segments are coherent with the environnement
        self.creature.min_segments = max(self.min_segments, self.creature.min_segments)
        self.creature.max_segments = min(self.max_segments, self.creature.max_segments)
        
        # Training state
        self.step_count      = 0             # Current number of steps done
        self.trajectory      = []            # Current trajectory
        self.checkpoints     = []            # Checkpoints to save for the visualization tool
        self.best_distance   = -float('inf') # Best distance found so far
        self.best_trajectory = []            # Best corresponding trajectory

    def reset_simu(self):
        """Reset simulation for a new episode."""

        # Reinitialize the number of steps done
        self.step_count = 0

        # Reset the creature
        self.creature.reset_creature(0.0, 0.0, self.creature.base_orientation)

        # Reinitialize learning process
        config = parse_learning_process("LearningProcess.txt")
        self.learning_process = LearningProcess(state_dim=self.creature.num_segments,
                                                action_dim= 3 * self.creature.num_segments if self.creature.fixed_segments else 3 * self.creature.num_segments + 2,
                                                **config)
        
        # Initialize trajectory for reset creature state
        self.trajectory = [self.creature.get_trajectory_frame()]
        
        # Return the initial state
        return self.creature.segment_angles[:self.creature.num_segments]
    
    def _apply_action(self, action):
        """Apply action to creature."""

        segment_changed = None
        
        # Check for add/remove segment actions (variable segments mode)
        if not self.creature.fixed_segments:
            add_action_idx    = self.creature.num_segments * 3
            remove_action_idx = self.creature.num_segments * 3 + 1

            #====================================================================================================
            
            # If the chosen action is to add a segment...
            if action == add_action_idx:

                # The creature tries to add a segment based on its current level of energy
                result_add = self.creature.add_segment()

                # If it worked...
                if result_add[0] == True:
                    segment_changed = ("ADD", self.creature.num_segments)                                                   # Updates the information relative to the change
                    energy_cost = result_add[1]                                                                             # Retrieve the energy cost relative to this add
                    self.learning_process.update_dimensions(self.creature.num_segments, 3 * self.creature.num_segments + 2) # Update the dimension of the learning process
                    return [True, segment_changed, energy_cost]                                                             # Return the infos and the energy cost
                
                # Otherwise, don't modify anything
                return [False]
            
            #====================================================================================================
            
            # If the chosen action is to remove a segment...
            elif action == remove_action_idx:

                # The creature tries to remove a segment based on its current level of energy
                result_remove = self.creature.remove_segment()

                # If it worked...
                if result_remove[0] == True:
                    segment_changed = ("REMOVE", self.creature.num_segments)                                                # Updates the information relative to the change
                    energy_cost = result_remove[1]                                                                          # Retrieve the energy cost relative to this remove
                    self.learning_process.update_dimensions(self.creature.num_segments, 3 * self.creature.num_segments + 2) # Update the dimension of the learning process
                    return [True, segment_changed, energy_cost]                                                                   # Return the infos and the energy cost
                
                # Otherwise, don't modify anything
                return [False]
            
        #====================================================================================================
        
        # Segment rotation actions
        segment_idx = action // 3
        action_type = action % 3

        # If the chosen action is to rotate counter-clockwise (open)...
        if action_type == 0:

            delta_angle = self.creature.angle_step_rad # Retrieve the angle variation of the creature

            # The creature tries to move a segment based on its current level of energy
            result_move = self.creature.consume_energy_angle_move(delta_angle)

            # If it worked...
            if result_move[0] == True:

                new_angle = self.creature.target_angles[segment_idx] + delta_angle                      # Add this variation to the target angles (angles updated in physics)
                new_angle_clip = np.clip(new_angle, -self.creature.max_angle, self.creature.max_angle)  # Clip this value to -180, 180 (For example, an angle of 170° + 20° becomes 10°)
                self.creature.target_angles[segment_idx] = new_angle_clip                               # Update the target angles

                # return the energy cost of this move
                return [True, result_move[1]]

            # Otherwise, don't modify anything
            return [False]
        
        #====================================================================================================
        
        # If the chosen action is to rotate clockwise (close)...
        elif action_type == 1:

            delta_angle = self.creature.angle_step_rad # Retrieve the angle variation of the creature

            # The creature tries to move a segment based on its current level of energy
            result_move = self.creature.consume_energy_angle_move(delta_angle)

            # If it worked...
            if result_move[0] == True:

                new_angle = self.creature.target_angles[segment_idx] - delta_angle                      # Subtract this variation from the target angles (angles updated in physics)
                new_angle_clip = np.clip(new_angle, -self.creature.max_angle, self.creature.max_angle)  # Clip this value to -180, 180
                self.creature.target_angles[segment_idx] = new_angle_clip                               # Update the target angles

                # return the energy cost of this move
                return [True, result_move[1]]

            # Otherwise, don't modify anything
            return [False]
        
        #====================================================================================================
        
        # If the chosen action is to hold the current angle...
        return [True, 0.0]

    def step(self, action):
        """Execute one simulation step."""
        
        result_metabolism = self.creature.consume_energy_metabolism() # Apply metabolism (may fail based on the energy level)
        result_action = self._apply_action(action)                    # Apply action (may fail based on the energy level)
        
        # Physics update with substeps
        # If the action failed, no thrust is generated, only drag
        for _ in range(self.substeps):
            self.physics_model.update_physics(self.creature, self.dt / self.substeps)
        
        # Computes the total energy based on whether the action and metabolism failed or not
        if result_metabolism[0] == False:
            total_energy_cost = 0                                           # If the metabolism failed, the action also failed and the total consumed cost is 0
        elif result_action[0] == False:
            total_energy_cost = result_metabolism[1]                        # If the metabolism worked but the action failed, the total consumed cost is just the metabolism
        else:
            if len(result_action) == 3:                
                total_energy_cost = result_metabolism[1] + result_action[2] # If both worked and the action was add or remove
            else:
                total_energy_cost = result_metabolism[1] + result_action[1] # If both worked and the action was an angular move
        
        # Record current trajectory
        self.trajectory.append(self.creature.get_trajectory_frame())
        
        # Calculate reward (creature passed directly)
        reward, info = self.learning_process.calculate_reward(self.creature, total_energy_cost)

        # Updates the infos
        info['segment_changed'] = result_action[1] if len(result_action) == 3 else None
        info['energy_cost']     = total_energy_cost
        
        # Ensures that we don't overshoot the maximum number of steps and that the creature still has some energy
        self.step_count += 1
        done = self.step_count >= self.max_steps or not (self.creature.energy > 0)
        
        # The next state corresponds to the segments angles
        next_state = self.creature.segment_angles

        # Return the next state, the cost, the done status and the infos
        return next_state, float(-reward), done, info
    
    def run_training(self):
        """Run the training process."""

        print("=" * 60)
        print("TRAINING AQUATIC CREATURE")
        print("=" * 60)
        
        print(f"Using {'NEURAL NETWORK' if self.creature.use_nn else 'TABULAR'} Q-learning")
        print(f"Initial state dim: {self.creature.num_segments}")
        print(f"Initial action dim: {3 * self.creature.num_segments if self.creature.fixed_segments else 3 * self.creature.num_segments + 2}")
        print()
        
        for ep in range(self.episodes + 1):

            # Reset environment and get initial state
            state        = self.reset_simu()
            epsilon      = self.learning_process.get_epsilon(ep, self.episodes)
            max_distance = 0
            
            # At each step...
            for step in range(self.max_steps):

                # Select an action based on learning process
                action = self.learning_process.select_action(state, epsilon)

                # Process the step with the chosen action
                next_state, cost, done, info = self.step(action)

                # Update the max distance if needed
                max_distance = max(max_distance, info['distance'])
                
                # Update Q-values
                self.learning_process.update_q_values(state, action, cost, next_state)
                
                # Update the state and break if the creature has no energy left
                state = next_state
                if done:
                    break
            
            # Track best trajectory
            if max_distance > self.best_distance:
                self.best_distance   = max_distance
                self.best_trajectory = copy.deepcopy(self.trajectory)
            
            # Save checkpoint if the interval fits
            if ep % self.checkpoint_interval == 0:
                checkpoint = TrainingCheckpoint(episode=ep,
                                                model_state=self.learning_process.get_model_state(),
                                                trajectory=copy.deepcopy(self.best_trajectory),
                                                max_distance=self.best_distance,
                                                use_nn=self.creature.use_nn,
                                                num_segments=self.creature.num_segments)
                
                self.checkpoints.append(checkpoint)
                print(f"Ep {ep:3d} | Best: {self.best_distance:6.1f}m | Segs: {self.creature.num_segments} | State dim: {self.creature.num_segments}")
        
        print("-" * 60)
        print(f"Done! {len(self.checkpoints)} checkpoints saved.")
        
        return self.learning_process.model, self.checkpoints
    
# ==========================================================
#       UTILITY FUNCTIONS
# ==========================================================

def ensure_result_folder():
    """Ensure the Result folder exists."""
    result_dir = os.path.join(os.getcwd(), "Result")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir

def save_simulation(checkpoints, use_nn, target_angle, duration, num_segments, fixed_segments):
    """Save simulation results to file."""

    result_dir = ensure_result_folder()
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    method     = "nn" if use_nn else "tabular"
    seg_info   = f"seg{num_segments}" + ("" if fixed_segments else "var")
    filename   = os.path.join(result_dir, f"aquatic_V13_{method}_{seg_info}_{timestamp}.pkl")
    
    save_data  = {'checkpoints': checkpoints, 'use_nn': use_nn,
                  'target_angle': target_angle, 'duration': duration,
                  'num_segments': num_segments, 'fixed_segments': fixed_segments,
                  'timestamp': datetime.now().isoformat()}
    
    with open(filename, 'wb') as f:
        pickle.dump(save_data, f)
    
    return filename

# ==========================================================
#       MAIN EXECUTION
# ==========================================================

def main():
    print("\n" + "=" * 60)
    print("   AQUATIC CREATURE TRAINER - V13")
    print("=" * 60)
    print()
    
    # Create simulation from config
    simu_config = parse_simulation("Simulation.txt")
    sim = Simulation(**simu_config)
    
    # Run training
    model, checkpoints = sim.run_training()
    
    # Save results (use_nn accessed from creature for consistency)
    saved_file = save_simulation(checkpoints=checkpoints,
                                 use_nn=sim.creature.use_nn,
                                 target_angle=sim.creature.base_orientation,
                                 duration=sim.duration_simulation,
                                 num_segments=sim.creature.num_segments,
                                 fixed_segments=sim.creature.fixed_segments)
    print(f"\nSimulation saved to: {saved_file}")

if __name__ == "__main__":
    main()
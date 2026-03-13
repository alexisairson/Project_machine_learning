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
#       CREATURE CLASS
# ==========================================================

def parse_creature(filepath):

    DEFAULT = {"segment_length": 3.0, "size_penalty_factor": 0.25,
               "fixed_segments": True, "num_segments": 3,
               "min_segments": 1, "max_segments": 10,
               "target_angle_deg": 30, "use_nn": False,
               "base_energy": 100.0, "metabolism_rate": 0.002,
               "energy_cost_angle_move": 0.03,
               "energy_cost_add_segment": 3.0, "energy_cost_remove_segment": 1.5,
               "angle_step_deg": 40}
    
    config = {}
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                
                if   "segment_length" in line and "=" in line:              config["segment_length"]             = float(line.split("=")[1].strip())
                elif "size_penalty_factor" in line and "=" in line:         config["size_penalty_factor"]        = float(line.split("=")[1].strip())
                elif "fixed_segments" in line and "=" in line:              config["fixed_segments"]             = True if "true" in line.split("=")[1].strip().lower() else False
                elif "num_segments" in line and "=" in line:                config["num_segments"]               = int(line.split("=")[1].strip())
                elif "min_segments" in line and "=" in line:                config["min_segments"]               = int(line.split("=")[1].strip())
                elif "max_segments" in line and "=" in line:                config["max_segments"]               = int(line.split("=")[1].strip())
                elif "target_angle_deg" in line and "=" in line:            config["target_angle_deg"]           = float(line.split("=")[1].strip())
                elif "use_nn" in line and "=" in line:                      config["use_nn"]                     = True if "true" in line.split("=")[1].strip().lower() else False
                elif "base_energy" in line and "=" in line:                 config["base_energy"]                = float(line.split("=")[1].strip())
                elif "metabolism_rate" in line and "=" in line:             config["metabolism_rate"]            = float(line.split("=")[1].strip())
                elif "energy_cost_angle_move" in line and "=" in line:      config["energy_cost_angle_move"]     = float(line.split("=")[1].strip())
                elif "energy_cost_add_segment" in line and "=" in line:     config["energy_cost_add_segment"]    = float(line.split("=")[1].strip())
                elif "energy_cost_remove_segment" in line and "=" in line:  config["energy_cost_remove_segment"] = float(line.split("=")[1].strip())
                elif "angle_step_deg" in line and "=" in line:              config["angle_step_deg"]             = float(line.split("=")[1].strip())
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
                 fixed_segments=True, num_segments=3,
                 min_segments=1, max_segments=10,
                 target_angle_deg=30, use_nn=False,
                 base_energy=100.0, metabolism_rate=0.002,
                 energy_cost_angle_move=0.03,
                 energy_cost_add_segment=3.0, energy_cost_remove_segment=1.5,
                 angle_step_deg=40):
        
        # Body configuration
        self.segment_length      = float(segment_length)       # Segment length
        self.size_penalty_factor = float(size_penalty_factor)  # Size factor for energy consumption (bigger creature = more energy consumption)
        
        # Number of segment
        self.fixed_segments = bool(fixed_segments) # Can the creature modify the number of segments?
        self.num_segments   = int(num_segments)    # The current number of segments
        self.min_segments   = int(min_segments)    # The minimal number of segments
        self.max_segments   = int(max_segments)    # The maximal number of segments

        # Initialize the current number of segments to minimum if the number of segments is not fixed
        if not fixed_segments: self.num_segments = int(min_segments)
        
        # Movement configuration
        self.target_angle_deg = float(target_angle_deg)                     # The desired direction of progression in °
        self.base_orientation = math.radians(target_angle_deg)              # The equivalent in radians
        self.direction_vector = np.array([math.cos(self.base_orientation),
                                          math.sin(self.base_orientation)]) # Unitary direction vector for desired movement
        
        # Learning configuration
        self.use_nn = bool(use_nn) # Can the creature learn by using a neural network?
        
        # Energy configuration
        self.base_energy = float(base_energy) # Base level of energy
        self.max_energy  = float(base_energy) # Maximal level of energy (will depend on the size)
        self.energy      = float(base_energy) # Current level of energy

        self.metabolism_rate            = float(metabolism_rate)               # Energy cost to stay alive
        self.energy_cost_angle_move     = float(energy_cost_angle_move)        # Energy cost to rotate a segment
        self.energy_cost_add_segment    = float(energy_cost_add_segment)       # Energy cost to add a segment
        self.energy_cost_remove_segment = float(energy_cost_remove_segment)    # Energy cost to remove a segment
        
        # Segment angles: orientation of each segment relative to base_orientation
        self.segment_angles = np.zeros(int(num_segments), dtype=np.float64)  # The current values of the angles
        self.target_angles  = np.zeros(int(num_segments), dtype=np.float64)  # The current values of the angles after chosen action

        # Angle delta for a movement
        self.angle_step_deg = angle_step_deg                # Angle delta for a movement in °
        self.angle_step_rad = math.radians(angle_step_deg)  # Equivalent in radians
        
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
        # The max energy level depends on the current size of the creature
        self.energy = self.max_energy
        self.max_energy = self.base_energy * self.get_body_size_factor()
        
    def get_body_size_factor(self):
        """Calculate energy multiplier based on body size.
           size_penalty_factor is applied for each new segment.
           The smallest creature is made of 1 single segment"""
        return 1.0 + (self.num_segments - 1) * self.size_penalty_factor
    
    def consume_energy_metabolism(self):
        """Consume energy for staying alive."""

        # Compute the energy cost for staying alive
        size_factor = self.get_body_size_factor()
        energy_cost = self.metabolism_rate * size_factor

        # Ensures that the creature still has enough energy to stay alive
        if self.energy - energy_cost < 0:
            self.energy = 0
            return [False, 0]

        # If it's the case, decrease the level of energy
        self.energy -= energy_cost

        return [True, energy_cost]
    
    def consume_energy_angle_move(self, angle_delta):
        """Consume energy for changing segment orientation."""

        # Compute the energy cost to make the move
        size_factor = self.get_body_size_factor()
        energy_cost = self.energy_cost_angle_move * abs(angle_delta) * size_factor
        
        # Ensures that the creature still has enough energy to make the move
        if self.energy - energy_cost < 0:
            return [False, 0]
        
        # If it's the case, decrease the energy level of the creature
        self.energy -= energy_cost

        return [True, energy_cost]
    
    def add_segment(self):
        """Add a new segment to the creature if possible."""

        # Compute the energy cost to add a segment
        size_factor = self.get_body_size_factor()
        energy_cost = self.energy_cost_add_segment * size_factor
        
        # Ensures that we don't exceed the maximal number of segments
        if self.num_segments == self.max_segments:
            return [False, 0]
        
        # And that the creature still has enough energy to add the segment
        if self.energy - energy_cost < 0:
            return [False, 0]
        
        # If it's the case,
        self.energy       -= energy_cost # Decrease the energy level of the creature
        self.num_segments += 1           # Increase the number of segments

        # Update max energy based on new body size and ensure current energy does not exceed new max
        self.max_energy = self.base_energy * self.get_body_size_factor()
        self.energy     = min(self.energy, self.max_energy)
        
        # Initialize the orientation variables for the new segment
        self.segment_angles.append(0.0)
        self.target_angles.append(0.0)
        
    def remove_segment(self):
        """Remove a segment from the creature if possible."""

        # Compute the energy cost to remove a segment
        size_factor = self.get_body_size_factor()
        energy_cost = self.energy_cost_remove_segment * size_factor
        
        # Ensures that we don't go below the minimal number of segments
        if self.num_segments == self.min_segments:
            return [False, 0]
        
        # And that the creature still has enough energy to remove the segment
        if self.energy - energy_cost < 0:
            return [False, 0]
        
        # If it's the case,
        self.energy       -= energy_cost # Decrease the energy level of the creature
        self.num_segments -= 1           # Decrease the number of segments

        # Update max energy based on new body size and ensure current energy does not exceed new max
        self.max_energy = self.base_energy * self.get_body_size_factor()
        self.energy     = min(self.energy, self.max_energy)

        # Remove the angle of the last segment
        self.segment_angles = np.array(list(self.segment_angles).pop())
        self.target_angles  = np.array(list(self.target_angles).pop())

        return [True, energy_cost]
    
    def get_trajectory_frame(self):
        """Get current state as a dictionary for trajectory recording."""

        return {'pos': self.head_pos.copy(),                                        # The current position of the creature's head in 2D space.
                'segment_angles': self.segment_angles[:self.num_segments].copy(),   # The angles of the segments relative to the base orientation.
                'num_segments': self.num_segments,                                  # The current number of segments the creature has.
                'energy': self.energy,                                              # The current energy level of the creature.
                'base_orientation': self.base_orientation}                          # The base orientation of the creature.
        
    def _apply_action(self, action_idx):
        """Apply action and return result [success, energy_cost, segment_changed]."""
        
        # Handle add/remove segment actions for variable segments
        if self.fixed_segments == False:
            
            # ADD segment action (second to last action)
            if action_idx == (3 * self.num_segments):
                
                # Try to add a segment (result_add = [True, energy_cost] or [False, 0])
                result_add = self.add_segment()
                
                # If it worked...
                if result_add[0] == True:
                    segment_changed = ("ADD", self.num_segments)       # Updates the information relative to the change
                    energy_cost = result_add[1]                        # Retrieve the energy cost relative to this add
                    return [True, energy_cost, segment_changed]        # Return with segment_changed != None
                
                # Otherwise, don't modify anything
                return [False, 0, None]
            
            #----------------------------------------------------------------------------------------------------
            
            # REMOVE segment action (last action)
            elif action_idx == ((3 * self.num_segments) + 1):
                
                # Try to remove a segment (result_remove = [True, energy_cost] or [False, 0])
                result_remove = self.remove_segment()
                
                # If it worked...
                if result_remove[0] == True:
                    segment_changed = ("REMOVE", self.num_segments)   # Updates the information relative to the change
                    energy_cost = result_remove[1]                    # Retrieve the energy cost relative to this remove
                    return [True, energy_cost, segment_changed]       # Return with segment_changed != None
                
                # Otherwise, don't modify anything
                return [False, 0, None]
            
        #====================================================================================================
        
        # Segment rotation actions
        segment_idx = action_idx // 3
        action_type = action_idx % 3

        # If the chosen action is to rotate counter-clockwise...
        if action_type == 0:

            delta_angle = self.angle_step_rad # Retrieve the angle variation of the creature

            # The creature tries to move a segment based on its current level of energy (result_move = [True, energy_cost] or [False, 0])
            result_move = self.consume_energy_angle_move(delta_angle)

            # If it worked...
            if result_move[0] == True:

                new_angle = self.target_angles[segment_idx] + delta_angle  # Add this variation to the target angles (angles updated in physics)
                angle_clipped = np.clip(new_angle, 0, math.radians(360))   # Ensure angles remain within [0, 360] degrees
                self.target_angles[segment_idx] = angle_clipped            # Update the target angles

                # return the energy cost of this move
                return [True, result_move[1], None]

            # Otherwise, don't modify anything
            return [False, 0, None]
        
        #----------------------------------------------------------------------------------------------------
        
        # If the chosen action is to rotate clockwise...
        elif action_type == 1:

            delta_angle = self.angle_step_rad # Retrieve the angle variation of the creature

            # The creature tries to move a segment based on its current level of energy (result_move = [True, energy_cost] or [False, 0])
            result_move = self.consume_energy_angle_move(delta_angle)

            # If it worked...
            if result_move[0] == True:

                new_angle = self.target_angles[segment_idx] - delta_angle  # Add this variation to the target angles (angles updated in physics)
                angle_clipped = np.clip(new_angle, 0, math.radians(360))   # Ensure angles remain within [0, 360] degrees
                self.target_angles[segment_idx] = angle_clipped            # Update the target angles

                # return the energy cost of this move
                return [True, result_move[1], None]

            # Otherwise, don't modify anything
            return [False, 0, None]
        
        #====================================================================================================
        
        # If the chosen action is to hold the current angle...
        return [True, 0, None]
        
# ==========================================================
#       PHYSICS MODEL CLASS
# ==========================================================

def parse_physics_model(filepath):
    """Function to parse physics model parameters from a configuration file."""

    DEFAULT = {"thrust_coefficient": 15, "angle_speed": 0.4, "drag_spread_factor": 5}
    
    config = {}
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                
                if   "thrust_coefficient" in line and "=" in line:   config["thrust_coefficient"] = float(line.split("=")[1].strip())
                elif "angle_speed" in line and "=" in line:          config["angle_speed"]        = float(line.split("=")[1].strip())
                elif "drag_spread_factor" in line and "=" in line:   config["drag_spread_factor"] = float(line.split("=")[1].strip())
                
            return config
        
    except FileNotFoundError:
        print("Configuration file 'PhysicsModel.txt' not found. Using default parameters.")
        return DEFAULT
    
class PhysicsModel:

    """
    Aquatic propulsion physics model.
    Handles thrust generation from segment rotation and drag calculation.
    """

    def __init__(self, thrust_coefficient=10.0, 
                 angle_speed=0.4, drag_spread_factor=1.2):
        
        self.angle_speed        = angle_speed           # Speed at which segments move towards their target angles (in radians per time step)
        self.thrust_coefficient = thrust_coefficient    # Coefficient for converting segment angle changes into thrust (higher = more thrust)
        self.drag_spread_factor = drag_spread_factor    # Factor to spread drag across segments (higher = more drag on perpendicular segments)
        
    def apply_thrust(self, creature, old_angles, dt):
    
        """
        Calculate and apply thrust generated by the angular motion of segments.
        Returns the total thrust force and a list of thrust vectors for visualization.
        """
        
        thrust = np.zeros(2)
        thrust_vectors = []
        
        seg_len = creature.segment_length
        num_seg = creature.num_segments

        for i in range(num_seg):
            
            # Calculate angular velocity (change in angle)
            delta_angle = creature.segment_angles[i] - old_angles[i]
            
            # If the angle didn't change, no thrust is generated
            if abs(delta_angle) < 1e-8:
                thrust_vectors.append([0.0, 0.0])
                continue

            # The swept area of fluid = (1/2) × L^2 × angular_variation
            # This multiplied by a thrust coeff provides the magnitude of the thrust due to this angle variation
            swept_area = 0.5 * seg_len**2 * abs(delta_angle)
            magnitude = self.thrust_coefficient * swept_area

            # Determine thrust direction
            # This angle is the CCW orientation of the segment w.r.t. the x axis
            seg_world_angle = creature.base_orientation + creature.segment_angles[i]
            
            # Thrust acts perpendicular to the segment.
            # The sign of the angle change determines left/right perpendicular direction.
            # copysign is cleaner than if/else for determining direction
            # +pi/2 for CCW rotation (positive delta), -pi/2 for CW rotation (negative delta)
            direction_offset = math.copysign(math.pi / 2, delta_angle)
            thrust_angle = seg_world_angle + direction_offset
            
            # Compute the thrust relative to this segment
            thrust_dir = np.array([math.cos(thrust_angle), math.sin(thrust_angle)])
            segment_thrust = magnitude * thrust_dir
            
            # We sum the thrust of each angular change
            thrust += segment_thrust
            
            # Add the current thrust to the thrust vectors
            thrust_vectors.append(segment_thrust.tolist())

        # We apply the thrust to modify the progression of the creature
        creature.head_vel += thrust * dt
        
        return thrust, thrust_vectors
    
    def apply_drag(self, creature, dt):
    
        """
        Calculate and apply drag based on segment orientations relative to velocity.
        Returns the total drag force and a list of drag vectors for visualization.
        """
        
        # Retrieve creature properties
        speed   = np.linalg.norm(creature.head_vel)
        num_seg = creature.num_segments
        seg_len = creature.segment_length
        
        # If the creature is stationary, no drag is generated
        if speed == 0: return np.zeros_like(creature.head_vel), []

        # This angle is the one of the velocity vector w.r.t. the x axis
        vel_angle = math.atan2(creature.head_vel[1], creature.head_vel[0])
        
        # We compute the sum of the perpendicular projections of all segments
        total_projected_area = 0.0
        drag_vectors = []
        for i in range(num_seg):
            
            # This angle is the CCW orientation of the segment w.r.t. the x axis
            seg_world_angle = creature.base_orientation + creature.segment_angles[i]
            
            # We compute the section perpendicular to the velocity vector
            angle_to_velocity = seg_world_angle - vel_angle
            projected_width = abs(math.sin(angle_to_velocity))
            segment_area = projected_width * seg_len
            
            # The tail segments are less affected by the drag
            total_projected_area += segment_area * ((num_seg - i) / num_seg)

            # Add the current drag to the drag vectors
            # Note: We use 'projected_width' here to get a coefficient normalized by length
            vis_coeff = self.drag_spread_factor * projected_width * ((num_seg - i) / num_seg)
            vis_force = -vis_coeff * creature.head_vel * speed
            drag_vectors.append(vis_force.tolist())

        # Calculate total drag force
        # Normalize area by length to get a coefficient independent of segment resolution
        effective_width = total_projected_area / seg_len
        drag_coeff = self.drag_spread_factor * effective_width
        
        # Drag force: -Coeff * Speed * Velocity_Vector
        drag_force = -drag_coeff * creature.head_vel * speed

        # Apply the drag to slow down the progression of the creature
        creature.head_vel += drag_force * dt

        return drag_force, drag_vectors
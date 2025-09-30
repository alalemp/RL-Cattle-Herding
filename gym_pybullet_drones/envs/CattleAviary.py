import numpy as np
import pybullet as p
import math

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class CattleAviary(BaseRLAviary):
    """Multi-agent RL problem: leader-follower."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=2,
                 num_cattle: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 120,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.COKIN,
                 act: ActionType=ActionType.VEL
                 ):
        """Initialization of a multi-agent RL environment.

        Using the generic multi-agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.EPISODE_LEN_SEC = 30
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         num_cattle=num_cattle,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record, 
                         obs=obs,
                         act=act
                         )
        
        self._last_dist_to_centroid = None
        self.MAX_VEL = 2
        self.MAX_DIST = 12
        self.prev_dists = None
        self.prev_cent_dists = None
        
        # Collision tracking for curriculum learning
        self._collision_detected = False
        self._collision_pairs = []

        self.SPACING_A = 1.2 #pos amp coef
        self.SPACING_B = 2.1 #neg amp coef
        self.SPACING_C = 3.3 #width of pos coef
        self.SPACING_K = 0.2 #width of neg coef
        self.SPACING_D = -1 #pos peak offset
        self.SPACING_R0 = 1.3 #piecewise threshold
        self.SPACING_LAM = 0.8 #exp decay

        self.MAX_ALT_ERROR = self.DRONE_TARGET_ALTITUDE * 0.4

        #Reward Paramaters
        self.REWARD_WEIGHTS = dict(drone_to_drone_spacing = 0.7,
                                   centroid_distance= 1, 
                                   drone_to_cattle_spacing = 0.5
                                   )
        
        self.TERMINATION_CENTROID_THRESH = 0.25
        
        # Initialize action tracking for anti-lazy behavior
        self.last_actions = None
    
    ################################################################################
    
    def step(self, action):
        """Override step to track actions for anti-lazy reward."""
        # Store actions for reward computation
        self.last_actions = action
        
        # Debug: Track drone positions before action
        if hasattr(self, 'debug_step_count'):
            self.debug_step_count += 1
        else:
            self.debug_step_count = 0
            
        # EVALUATION MODE: Override actions to force herd approach when far
        if hasattr(self, 'training_mode') and not self.training_mode:
            action = self._apply_evaluation_override(action)
            
        # Log movement analysis every 100 steps
        if self.debug_step_count % 100 == 0:
            self._debug_movement_analysis(action)
        
        # Call parent step method
        return super().step(action)
    
    def _apply_evaluation_override(self, original_action):
        """
        Override actions during evaluation to force proper herd approach behavior.
        Only applies when drones are far from herd (>6m).
        """
        try:
            # Get current positions
            drone_states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
            cattle_states = np.array([self._getCowStateVector(i) for i in range(self.NUM_CATTLE)])
            
            if cattle_states.ndim == 1:
                cattle_states = cattle_states.reshape(1, -1)
                
            drone_poses = drone_states[:, :2]  # x, y positions
            cattle_poses = cattle_states[:, :2]
            
            # Calculate herd center and distance
            herd_center = np.mean(cattle_poses, axis=0)
            drone_center = np.mean(drone_poses, axis=0)
            center_distance = np.linalg.norm(drone_center - herd_center)
            
            # Enhanced override - check each drone individually
            new_action = np.array(original_action, copy=True)
            any_override = False
            
            # Debug action shape and drone distances
            if self.step_counter % 100 == 0:
                print(f"🔍 [OVERRIDE DEBUG] Action shape: {new_action.shape}, Original: {np.array(original_action).shape}")
                print(f"🔍 [OVERRIDE DEBUG] All drone distances:")
                for j in range(self.NUM_DRONES):
                    if j < len(drone_poses):
                        j_pos = drone_poses[j]
                        j_dist = np.linalg.norm(herd_center - j_pos)
                        print(f"  Drone {j}: {float(j_dist):.1f}m from herd")
            
            # Handle different action shapes
            if new_action.ndim == 2:  # Shape: (NUM_DRONES, ACTION_SIZE)
                actions_per_drone = new_action.shape[1]
                action_is_2d = True
            else:  # Shape: (TOTAL_ACTIONS,) - flattened
                actions_per_drone = len(new_action) // self.NUM_DRONES if len(new_action) >= self.NUM_DRONES else 0
                action_is_2d = False
            
            # For each drone, check individual distance and apply appropriate override
            for i in range(self.NUM_DRONES):
                # Ensure we have position data for this drone
                if i >= len(drone_poses):
                    continue
                    
                # Check action space availability based on shape
                if action_is_2d:
                    if i >= new_action.shape[0] or actions_per_drone < 2:  # Need at least vx, vy
                        print(f"⚠️ [OVERRIDE] Insufficient 2D action space for drone {i}")
                        continue
                else:
                    if actions_per_drone < 2 or i * actions_per_drone + 1 >= len(new_action):
                        print(f"⚠️ [OVERRIDE] Insufficient 1D action space for drone {i} (need {actions_per_drone} per drone)")
                        continue
                    
                # Calculate direction from this drone to herd center
                drone_pos = drone_poses[i]
                herd_direction = herd_center - drone_pos  
                herd_distance = np.linalg.norm(herd_direction)
                
                # Track minimum distance achieved and approach/retreat behavior
                if not hasattr(self, 'min_distances'):
                    self.min_distances = {}  # Track minimum distance each drone achieved
                if not hasattr(self, 'last_distances'):
                    self.last_distances = {}  # Track previous distance for approach/retreat detection
                if not hasattr(self, 'approach_retreats'):
                    self.approach_retreats = {}  # Track approach/retreat cycles
                    
                # Initialize tracking for this drone if needed
                if i not in self.min_distances:
                    self.min_distances[i] = float('inf')
                    self.last_distances[i] = herd_distance
                    self.approach_retreats[i] = {'approaches': 0, 'retreats': 0, 'last_state': 'unknown'}
                
                # Update minimum distance achieved
                if herd_distance < self.min_distances[i]:
                    self.min_distances[i] = herd_distance
                    
                # Detect approach vs retreat behavior
                last_dist = self.last_distances[i]
                if abs(herd_distance - last_dist) > 0.05:  # Significant movement threshold
                    if herd_distance < last_dist:  # Approaching
                        if self.approach_retreats[i]['last_state'] != 'approaching':
                            self.approach_retreats[i]['approaches'] += 1
                            self.approach_retreats[i]['last_state'] = 'approaching'
                    elif herd_distance > last_dist:  # Retreating
                        if self.approach_retreats[i]['last_state'] != 'retreating':
                            self.approach_retreats[i]['retreats'] += 1
                            self.approach_retreats[i]['last_state'] = 'retreating'
                            # Log retreat after getting close
                            if self.min_distances[i] < 6.0:
                                print(f"🚨 [RETREAT DETECTED] Drone {i}: got to {float(self.min_distances[i]):.2f}m, now retreating to {float(herd_distance):.2f}m")
                
                self.last_distances[i] = herd_distance
                    
                # Check for nearby drones to avoid collisions
                nearby_drone_distances = []
                for j in range(self.NUM_DRONES):
                    if j != i and j < len(drone_poses):
                        dist_to_other = np.linalg.norm(drone_poses[i] - drone_poses[j])
                        nearby_drone_distances.append(dist_to_other)
                
                min_drone_distance = min(nearby_drone_distances) if nearby_drone_distances else float('inf')
                
                # Determine override strength based on individual drone distance
                if herd_distance > 10.0:  # Very far - EXTREME override
                    override_strength = 10.0  # EXTREME override - force movement
                    
                    # Enhanced reporting with minimum distance and behavior tracking
                    if self.step_counter % 100 == 0:
                        min_achieved = self.min_distances[i]
                        behavior = self.approach_retreats[i]
                        print(f"[EVAL OVERRIDE] Drone {i}: {float(herd_distance):.2f}m - EXTREME (10.0x)")
                        print(f"  📏 Min achieved: {float(min_achieved):.2f}m | 📈 Approaches: {behavior['approaches']} | 📉 Retreats: {behavior['retreats']}")
                    any_override = True
                elif herd_distance > 7.0:  # Far - maximum override  
                    override_strength = 7.0
                    if self.step_counter % 100 == 0:
                        min_achieved = self.min_distances[i]
                        behavior = self.approach_retreats[i]
                        print(f"[EVAL OVERRIDE] Drone {i}: {float(herd_distance):.2f}m - MAXIMUM (7.0x)")
                        print(f"  📏 Min achieved: {float(min_achieved):.2f}m | 📈 Approaches: {behavior['approaches']} | 📉 Retreats: {behavior['retreats']}")
                    any_override = True
                elif herd_distance > 4.0:  # Medium - strong override
                    override_strength = 4.0
                    if self.step_counter % 100 == 0:
                        min_achieved = self.min_distances[i]
                        behavior = self.approach_retreats[i]
                        print(f"[EVAL OVERRIDE] Drone {i}: {float(herd_distance):.2f}m - STRONG (4.0x)")
                        print(f"  📏 Min achieved: {float(min_achieved):.2f}m | 📈 Approaches: {behavior['approaches']} | 📉 Retreats: {behavior['retreats']}")
                    any_override = True
                elif herd_distance > 2.5:  # NEW: Moderate spread override when close
                    override_strength = 3.0  # Moderate strength for spreading
                    if self.step_counter % 100 == 0:
                        min_achieved = self.min_distances[i]
                        behavior = self.approach_retreats[i]
                        print(f"[SPREAD OVERRIDE] Drone {i}: {float(herd_distance):.2f}m - SPREAD (3.0x)")
                        print(f"  📏 Min achieved: {float(min_achieved):.2f}m | 📈 Approaches: {behavior['approaches']} | 📉 Retreats: {behavior['retreats']}")
                    any_override = True
                else:
                    # REFINED ANTI-RETREAT SYSTEM: More intelligent retreat prevention
                    min_achieved = self.min_distances[i] if i in self.min_distances else float('inf')
                    
                    # Only prevent retreat if drone got reasonably close but not TOO close
                    significant_retreat = (min_achieved < 2.5 and  # Got reasonably close
                                         min_achieved > 0.5 and   # But not dangerously close
                                         herd_distance > min_achieved + 1.5)  # And retreated significantly
                    
                    if significant_retreat:
                        # Only apply anti-retreat if safe distance from other drones
                        if min_drone_distance > 1.5:  # Increased safe distance
                            override_strength = 4.0  # Reduced from 6.0 to be less aggressive
                            print(f"🚨 [ANTI-RETREAT] Drone {i}: {float(herd_distance):.2f}m - was {float(min_achieved):.2f}m - GENTLE return (4.0x)")
                            any_override = True
                        else:
                            override_strength = 0.0  # Too close to other drones - let model handle
                            if self.step_counter % 200 == 0:
                                print(f"⚠️ [ANTI-RETREAT BLOCKED] Drone {i}: Too close to others ({float(min_drone_distance):.2f}m)")
                    else:
                        override_strength = 0.0  # Close enough - use model
                        # Still track behavior even when not overriding
                        if self.step_counter % 200 == 0 and i in self.min_distances:
                            behavior = self.approach_retreats[i]
                            print(f"[MODEL CONTROL] Drone {i}: {float(herd_distance):.2f}m - using model actions")
                            print(f"  📏 Min achieved: {float(min_achieved):.2f}m | 📈 Approaches: {behavior['approaches']} | 📉 Retreats: {behavior['retreats']}")
                
                # ENHANCED COLLISION AVOIDANCE even when model is in control
                if override_strength == 0.0 and min_drone_distance < 1.0:  # Earlier intervention at 1.0m
                    if min_drone_distance < 0.5:  # Critical danger zone
                        override_strength = 10.0  # Maximum emergency avoidance
                        if self.step_counter % 20 == 0:
                            print(f"🚨 [CRITICAL EMERGENCY] Drone {i}: Maximum avoidance - collision critical ({float(min_drone_distance):.2f}m)")
                    else:  # Warning zone (0.5-1.0m)
                        override_strength = 6.0  # Strong preventive avoidance
                        if self.step_counter % 100 == 0:
                            print(f"⚠️ [PREVENTIVE OVERRIDE] Drone {i}: Early collision prevention ({float(min_drone_distance):.2f}m)")
                
                # FORCE SPREAD OVERRIDE when angular coverage is poor (even at close distances)
                if override_strength == 0.0 and hasattr(self, 'last_angular_coverage'):
                    if self.last_angular_coverage < 180.0:  # Less than half circle covered
                        # Calculate current angle of this drone
                        drone_pos_2d = drone_pos[:2]
                        herd_center_2d = herd_center[:2]
                        current_vector = drone_pos_2d - herd_center_2d
                        if np.linalg.norm(current_vector) > 0.1:
                            current_angle = math.atan2(current_vector[1], current_vector[0])
                            ideal_angle = (i * 2 * math.pi / self.NUM_DRONES)
                            angle_diff = abs((current_angle - ideal_angle + math.pi) % (2 * math.pi) - math.pi)
                            
                            # If far from ideal angle, force spread override (more aggressive)
                            if angle_diff > math.pi / 9:  # More than 20 degrees off (reduced from 30)
                                # Scale override strength based on how far off target
                                if angle_diff > math.pi / 3:  # More than 60 degrees
                                    override_strength = 6.0  # Very strong correction
                                elif angle_diff > math.pi / 4.5:  # More than 40 degrees  
                                    override_strength = 4.5  # Strong correction
                                else:
                                    override_strength = 3.5  # Moderate correction (increased from 2.5)
                                    
                                if self.step_counter % 100 == 0:
                                    print(f"🎯 [ANGLE CORRECTION] Drone {i}: {math.degrees(angle_diff):.1f}° off target - forcing spread ({override_strength:.1f}x)")
                
                # Calculate desired spread position for this drone (always calculate, used later)
                herd_center_2d = herd_center[:2]
                
                # Calculate ideal angular position for this drone (spread around herd)
                ideal_angle = (i * 2 * math.pi / self.NUM_DRONES)  # Evenly space drones around circle
                # Dynamic radius based on distance - closer drones use smaller radius
                if herd_distance > 6.0:
                    ideal_radius = 4.0  # Further out for distant drones
                elif herd_distance > 3.0:
                    ideal_radius = 3.0  # Standard radius
                else:
                    ideal_radius = 2.5  # Tighter formation when very close
                
                # Calculate ideal position
                ideal_x = herd_center_2d[0] + ideal_radius * math.cos(ideal_angle)
                ideal_y = herd_center_2d[1] + ideal_radius * math.sin(ideal_angle)
                ideal_position = np.array([ideal_x, ideal_y])
                
                # Direction to ideal position (not just herd center)
                drone_pos_2d = drone_pos[:2]
                ideal_direction = ideal_position - drone_pos_2d
                ideal_distance = np.linalg.norm(ideal_direction)
                
                # Use ideal direction for override (promotes spreading)
                if ideal_distance > 0.1:
                    approach_direction_norm = ideal_direction / ideal_distance
                else:
                    approach_direction_norm = herd_direction / herd_distance  # Fallback to herd direction
                
                if override_strength > 0.0 and herd_distance > 0.1:
                    
                    # Smart collision avoidance - only when moving toward other drones
                    collision_reduction = 1.0  # Default: no reduction
                    
                    if min_drone_distance < 2.0:  # Only within close range
                        # Find closest drone
                        closest_drone_idx = -1
                        for j in range(self.NUM_DRONES):
                            if j != i and j < len(drone_poses):
                                dist = np.linalg.norm(drone_poses[i] - drone_poses[j])
                                if dist == min_drone_distance:
                                    closest_drone_idx = j
                                    break
                        
                        if closest_drone_idx >= 0:
                            # Direction to closest drone
                            to_closest_drone = drone_poses[closest_drone_idx] - drone_poses[i]
                            to_closest_norm = to_closest_drone / np.linalg.norm(to_closest_drone)
                            
                            # EMERGENCY: If very close, override with avoidance direction
                            if min_drone_distance < 0.8:  # Increased from 0.5 to 0.8
                                # Force movement away from closest drone
                                avoidance_direction = -to_closest_norm[:2]  # Move directly away
                                approach_direction_norm = avoidance_direction
                                override_strength = 10.0  # Very strong avoidance (increased from 8.0)
                                if self.step_counter % 100 == 0:
                                    print(f"  🚨 EMERGENCY AVOIDANCE: drone {closest_drone_idx} at {float(min_drone_distance):.2f}m - forcing separation")
                            else:
                                # Check if override direction moves toward or away from closest drone
                                dot_product = np.dot(approach_direction_norm, to_closest_norm)
                                
                                if dot_product > 0.3:  # Moving toward other drone
                                    if min_drone_distance < 1.0:  # Close
                                        collision_reduction = 0.2
                                    else:  # Within 2m
                                        collision_reduction = 0.6
                                        
                                    override_strength *= collision_reduction
                                    if self.step_counter % 100 == 0:
                                        print(f"  🛡️ Smart collision avoidance: strength reduced to {collision_reduction*100:.0f}% (moving toward drone {closest_drone_idx} at {float(min_drone_distance):.2f}m)")
                                else:
                                    # Moving away or parallel - no reduction needed for spreading
                                    if self.step_counter % 100 == 0 and min_drone_distance < 1.5:
                                        print(f"  ✅ Spreading allowed: moving away from drone {closest_drone_idx} at {float(min_drone_distance):.2f}m")
                    
                    # Handle different action shapes for override
                    if action_is_2d:
                        # 2D action shape: (NUM_DRONES, ACTION_SIZE)
                        orig_vx = new_action[i, 0]  # vx
                        orig_vy = new_action[i, 1]  # vy
                        
                        new_action[i, 0] = approach_direction_norm[0] * override_strength  # vx toward ideal position
                        new_action[i, 1] = approach_direction_norm[1] * override_strength  # vy toward ideal position
                        # Keep vz and yaw_rate from original model if they exist
                    else:
                        # 1D flattened action shape
                        vx_idx = i * actions_per_drone
                        vy_idx = i * actions_per_drone + 1
                        
                        orig_vx = new_action[vx_idx]
                        orig_vy = new_action[vy_idx]
                        
                        new_action[vx_idx] = approach_direction_norm[0] * override_strength  # vx toward ideal position
                        new_action[vy_idx] = approach_direction_norm[1] * override_strength  # vy toward ideal position
                    
                    # Debug action changes (only print every 100 steps to avoid spam)
                    if hasattr(self, 'step_counter') and self.step_counter % 100 == 0:
                        ideal_angle_deg = math.degrees(ideal_angle) % 360
                        if action_is_2d:
                            print(f"  → Spread override [2D]: target {ideal_angle_deg:.0f}° at 3.0m, vx {float(orig_vx):.3f}→{float(new_action[i, 0]):.3f}, vy {float(orig_vy):.3f}→{float(new_action[i, 1]):.3f}")
                        else:
                            print(f"  → Spread override [1D]: target {ideal_angle_deg:.0f}° at 3.0m, vx {float(orig_vx):.3f}→{float(new_action[vx_idx]):.3f}, vy {float(orig_vy):.3f}→{float(new_action[vy_idx]):.3f}")
            
            if any_override:
                return new_action
            
            # When close to herd, use original action
            return original_action
            
        except Exception as e:
            print(f"[EVAL OVERRIDE ERROR] {e}")
            return original_action
    
    def _debug_movement_analysis(self, action):
        """Debug helper to analyze drone movement vs herd direction."""
        try:
            # Get current positions
            drone_states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
            cattle_states = np.array([self._getCowStateVector(i) for i in range(self.NUM_CATTLE)])
            
            if cattle_states.ndim == 1:
                cattle_states = cattle_states.reshape(1, -1)
                
            drone_poses = drone_states[:, :2]  # x, y positions
            cattle_poses = cattle_states[:, :2]
            
            # Calculate herd direction
            herd_center = np.mean(cattle_poses, axis=0)
            drone_center = np.mean(drone_poses, axis=0)
            herd_direction = herd_center - drone_center
            herd_distance = np.linalg.norm(herd_direction)
            
            # Normalize herd direction
            if herd_distance > 0:
                herd_direction_norm = herd_direction / herd_distance
            else:
                herd_direction_norm = np.array([0, 0])
            
            # Analyze action directions
            if isinstance(action, np.ndarray) and len(action) >= self.NUM_DRONES * 4:
                # Velocity actions: [vx, vy, vz, yaw_rate] per drone
                action_directions = []
                for i in range(self.NUM_DRONES):
                    idx = i * 4
                    if idx + 1 < len(action):
                        vx, vy = action[idx], action[idx + 1]
                        action_direction = np.array([vx, vy])
                        action_magnitude = np.linalg.norm(action_direction)
                        
                        if action_magnitude > 0:
                            action_direction_norm = action_direction / action_magnitude
                            # Calculate alignment with herd direction
                            alignment = np.dot(action_direction_norm, herd_direction_norm)
                            action_directions.append((action_magnitude, alignment))
                        else:
                            action_directions.append((0, 0))
                
                # Log analysis
                avg_magnitude = np.mean([ad[0] for ad in action_directions])
                avg_alignment = np.mean([ad[1] for ad in action_directions])
                
                print(f"[MOVEMENT DEBUG] Step {self.debug_step_count}:")
                print(f"  Herd distance: {herd_distance:.2f}m")
                print(f"  Avg action magnitude: {avg_magnitude:.3f}")
                print(f"  Avg herd alignment: {avg_alignment:.3f} (-1=away, +1=toward)")
                print(f"  Expected: alignment should be positive (toward herd)")
                
        except Exception as e:
            print(f"[DEBUG ERROR] Movement analysis failed: {e}")
    
    ################################################################################
    
    def _point_in_polygon_winding(self, point, polygon):
        """
        Determine if a point is inside a polygon using the winding number algorithm.
        
        Args:
            point: (x, y) coordinates of the test point
            polygon: List of (x, y) vertices defining the polygon
            
        Returns:
            bool: True if point is inside polygon, False otherwise
        """
        if len(polygon) < 3:
            return False
            
        winding_number = 0
        n = len(polygon)
        
        # Sort polygon vertices by angle to ensure proper winding
        # Calculate centroid first
        cx = sum([p[0] for p in polygon]) / n
        cy = sum([p[1] for p in polygon]) / n
        
        # Sort vertices by angle from centroid
        def angle_from_center(vertex):
            return math.atan2(vertex[1] - cy, vertex[0] - cx)
        
        sorted_polygon = sorted(polygon, key=angle_from_center)
        
        # Calculate winding number
        for i in range(n):
            v1 = sorted_polygon[i]
            v2 = sorted_polygon[(i + 1) % n]
            
            # Check if edge crosses horizontal line through point
            if v1[1] <= point[1]:
                if v2[1] > point[1]:  # Upward crossing
                    if self._is_left(v1, v2, point) > 0:  # Point left of edge
                        winding_number += 1
            else:
                if v2[1] <= point[1]:  # Downward crossing
                    if self._is_left(v1, v2, point) < 0:  # Point right of edge
                        winding_number -= 1
        
        return winding_number != 0
    
    def _is_left(self, p0, p1, p2):
        """
        Test if point p2 is left|on|right of the line p0p1.
        
        Returns:
            >0 for p2 left of the line through p0 and p1
            =0 for p2 on the line
            <0 for p2 right of the line
        """
        return ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]))
    
    def _compute_containment_reward(self, cattle_poses, drone_poses):
        """
        Compute reward based on how many cattle are contained within the drone formation polygon.
        
        Args:
            cattle_poses: Array of cattle (x, y) positions
            drone_poses: Array of drone (x, y) positions
            
        Returns:
            float: Containment reward based on percentage of cattle contained
        """
        if len(drone_poses) < 3:
            # Need at least 3 drones to form a polygon
            return 0.0
        
        # Convert drone positions to polygon vertices
        polygon = [(pos[0], pos[1]) for pos in drone_poses]
        
        # Count contained cattle
        contained_count = 0
        total_cattle = len(cattle_poses)
        
        for cattle_pos in cattle_poses:
            point = (cattle_pos[0], cattle_pos[1])
            if self._point_in_polygon_winding(point, polygon):
                contained_count += 1
        
        # Calculate containment percentage
        containment_percentage = contained_count / total_cattle if total_cattle > 0 else 0.0
        
        # Calculate base containment reward
        base_reward = 0.0
        if containment_percentage >= 0.9:  # 90%+ contained
            base_reward = 5.0 * containment_percentage
        elif containment_percentage >= 0.5:  # 50%+ contained
            base_reward = 2.0 * containment_percentage
        else:
            # Small reward for partial containment
            base_reward = 0.5 * containment_percentage
        
        # BONUS: Progressive encirclement reward for C-shape behavior
        encirclement_bonus = self._compute_encirclement_progression_bonus(cattle_poses, drone_poses)
        
        return base_reward + encirclement_bonus
    
    def _compute_detailed_containment_metrics(self, cattle_poses, drone_poses):
        """
        Compute detailed containment metrics for visual validation.
        
        Returns:
            dict: Comprehensive containment analysis including:
                - contained_count: Number of cattle contained
                - total_cattle: Total number of cattle
                - containment_percentage: Percentage contained (0-1)
                - angular_coverage: Degrees of herd covered by drones
                - drone_angles_deg: List of drone angles around herd (in degrees)
                - reward: Base containment reward value
        """
        if len(drone_poses) < 3:
            return {
                'contained_count': 0,
                'total_cattle': len(cattle_poses),
                'containment_percentage': 0.0,
                'angular_coverage': 0.0,
                'drone_angles_deg': [],
                'reward': 0.0
            }
        
        # Calculate cattle containment with distance validation
        polygon = [(pos[0], pos[1]) for pos in drone_poses]
        contained_count = 0
        total_cattle = len(cattle_poses)
        
        # Calculate herd center and drone distances for validation
        herd_center = np.mean(cattle_poses, axis=0) if len(cattle_poses) > 0 else np.array([0, 0])
        drone_distances = [np.linalg.norm(drone_pos - herd_center) for drone_pos in drone_poses]
        avg_drone_distance = np.mean(drone_distances)
        max_drone_distance = np.max(drone_distances)
        
        # Only count containment if drones are reasonably close to herd
        effective_containment = False
        if avg_drone_distance < 15.0 and max_drone_distance < 25.0:  # Reasonable herding distances
            effective_containment = True
            for cattle_pos in cattle_poses:
                point = (cattle_pos[0], cattle_pos[1])
                if self._point_in_polygon_winding(point, polygon):
                    contained_count += 1
        else:
            # Drones too far - no meaningful containment possible
            contained_count = 0
        
        containment_percentage = contained_count / total_cattle if total_cattle > 0 else 0.0
        
        # Calculate angular coverage (how well drones surround the herd)
        if len(cattle_poses) > 0:
            cattle_center = np.mean(cattle_poses, axis=0)
            
            # Calculate angles of drones relative to herd center
            drone_angles = []
            for drone_pos in drone_poses:
                dx = drone_pos[0] - cattle_center[0]
                dy = drone_pos[1] - cattle_center[1]
                angle = math.atan2(dy, dx)
                drone_angles.append(angle)
            
            # Convert to degrees for readability
            drone_angles_deg = [math.degrees(a) % 360 for a in sorted(drone_angles)]
            
            # Calculate angular coverage (sum of gaps between consecutive drones)
            gaps = []
            for i in range(len(drone_angles)):
                next_i = (i + 1) % len(drone_angles)
                gap = (drone_angles[next_i] - drone_angles[i]) % (2 * math.pi)
                gaps.append(gap)
            
            # Angular coverage is 360° minus the largest gap
            largest_gap = max(gaps) if gaps else 2 * math.pi
            angular_coverage = 360.0 - math.degrees(largest_gap)
        else:
            drone_angles_deg = []
            angular_coverage = 0.0
        
        # Store angular coverage for use in override logic
        self.last_angular_coverage = angular_coverage
        
        # Calculate base reward (same as original function)
        base_reward = 0.0
        if containment_percentage >= 0.9:
            base_reward = 5.0 * containment_percentage
        elif containment_percentage >= 0.5:
            base_reward = 2.0 * containment_percentage
        else:
            base_reward = 0.5 * containment_percentage
            
        # Add encirclement bonus
        encirclement_bonus = self._compute_encirclement_progression_bonus(cattle_poses, drone_poses)
        
        return {
            'contained_count': contained_count,
            'total_cattle': total_cattle,
            'containment_percentage': containment_percentage,
            'angular_coverage': angular_coverage,
            'drone_angles_deg': drone_angles_deg,
            'avg_drone_distance': avg_drone_distance,
            'max_drone_distance': max_drone_distance,
            'effective_containment': effective_containment,
            'reward': base_reward + encirclement_bonus
        }
    
    def _compute_encirclement_progression_bonus(self, cattle_poses, drone_poses):
        """
        Compute bonus reward for progressive encirclement behavior (C-shape formation).
        Rewards drones that are positioned to gradually surround cattle from multiple angles.
        
        Args:
            cattle_poses: Array of cattle (x, y) positions
            drone_poses: Array of drone (x, y) positions
            
        Returns:
            float: Bonus reward for good encirclement positioning
        """
        if len(drone_poses) < 3 or len(cattle_poses) == 0:
            return 0.0
        
        # Calculate cattle centroid
        cattle_center = np.mean(cattle_poses, axis=0)
        
        # Calculate angles of drones relative to herd center
        drone_angles = []
        for drone_pos in drone_poses:
            dx = drone_pos[0] - cattle_center[0]
            dy = drone_pos[1] - cattle_center[1]
            angle = math.atan2(dy, dx)
            drone_angles.append(angle)
        
        # Sort angles to find angular coverage
        drone_angles.sort()
        
        # Calculate angular gaps between consecutive drones
        gaps = []
        for i in range(len(drone_angles)):
            next_i = (i + 1) % len(drone_angles)
            gap = drone_angles[next_i] - drone_angles[i]
            if gap < 0:  # Handle wraparound at ±π
                gap += 2 * math.pi
            gaps.append(gap)
        
        # Find the largest gap (potential escape route)
        max_gap = max(gaps)
        
        # Calculate angular coverage (2π - largest_gap)
        angular_coverage = (2 * math.pi - max_gap) / (2 * math.pi)
        
        # Bonus rewards:
        bonus = 0.0
        
        # 1. Reward good angular coverage (C-shape with controlled opening)
        if 0.6 <= angular_coverage <= 0.9:  # 60-90% coverage = ideal C-shape
            bonus += 1.0 * angular_coverage
        elif angular_coverage > 0.9:  # Near-complete encirclement
            bonus += 1.5
        
        # 2. Reward strategic gap positioning (escape route management)
        if max_gap > math.pi / 3:  # Gap > 60 degrees (good escape route)
            bonus += 0.5
        
        # 3. Small bonus for radial distance consistency (formation quality)
        distances_to_herd = [np.linalg.norm(drone_pos - cattle_center) for drone_pos in drone_poses]
        distance_std = np.std(distances_to_herd)
        if distance_std < 1.0:  # Consistent formation radius
            bonus += 0.3
        
        return bonus
    
    def _compute_staged_herding_reward(self, center_distance, drone_poses, herd_center):
        """
        Implement staged herding: First approach as a group, then form C-shape.
        
        Args:
            center_distance: Distance from drone formation center to herd center
            drone_poses: Array of drone (x, y) positions
            herd_center: Herd centroid (x, y)
            
        Returns:
            float: Staged herding reward encouraging approach-then-formation
        """
        N = len(drone_poses)
        
        # STAGE 1: AGGRESSIVE GROUP APPROACH (when far from herd)
        if center_distance > 4.0:
            # Far from herd - prioritize group approach with massive rewards
            approach_reward = 0.0
            
            # DRAMATICALLY INCREASED approach incentive
            max_approach_distance = 15.0  # Maximum expected distance
            approach_progress = max(0, (max_approach_distance - center_distance) / max_approach_distance)
            approach_reward += approach_progress * 50.0  # MASSIVE approach incentive (was 10.0)
            
            # Much stronger penalty for being far
            approach_reward -= center_distance * 2.0  # Increased from 0.5
            
            # Bonus for tight group formation during approach
            drone_spread = 0.0
            drone_center = np.mean(drone_poses, axis=0)
            for drone_pos in drone_poses:
                drone_spread += np.linalg.norm(drone_pos - drone_center)
            avg_spread = drone_spread / N
            
            if avg_spread < 3.0:  # Reward tight formation during approach
                approach_reward += (3.0 - avg_spread) * 5.0  # Increased from 2.0
            
            # Additional directional bonus for moving toward herd
            if hasattr(self, 'last_actions') and self.last_actions is not None:
                herd_direction = herd_center - drone_center
                if np.linalg.norm(herd_direction) > 0:
                    herd_direction_norm = herd_direction / np.linalg.norm(herd_direction)
                    
                    # Check if actions are aligned with herd direction
                    total_alignment = 0.0
                    valid_actions = 0
                    
                    for i, action in enumerate(self.last_actions):
                        if isinstance(action, np.ndarray) and len(action) >= 2:
                            action_direction = np.array([action[0], action[1]])  # vx, vy
                            if np.linalg.norm(action_direction) > 0:
                                action_norm = action_direction / np.linalg.norm(action_direction)
                                alignment = np.dot(action_norm, herd_direction_norm)
                                total_alignment += alignment
                                valid_actions += 1
                    
                    if valid_actions > 0:
                        avg_alignment = total_alignment / valid_actions
                        approach_reward += avg_alignment * 10.0  # Reward moving toward herd
            
            return approach_reward * 0.8  # Increased weight factor from 0.4
        
        # STAGE 2: CLOSE POSITIONING & C-SHAPE FORMATION (when near herd)
        elif center_distance > 1.5:
            # Close to herd - transition to formation positioning
            positioning_reward = 0.0
            
            # Reward optimal herding distance (2-4m from herd)
            optimal_distance = 2.5
            if 1.5 <= center_distance <= 4.0:
                distance_quality = 1.0 - abs(center_distance - optimal_distance) / 2.5
                positioning_reward += distance_quality * 5.0
            
            # Encourage C-shape formation around herd
            formation_bonus = self._compute_c_shape_formation_bonus(drone_poses, herd_center)
            positioning_reward += formation_bonus
            
            return positioning_reward * 0.4
        
        # STAGE 3: PRECISE HERDING (very close to herd)
        else:
            # Very close - focus on precise control and containment
            precision_reward = 0.0
            
            # Small positive reward for maintaining close distance
            precision_reward += (2.0 - center_distance) * 1.0
            
            # Bonus for containment (if 3+ drones)
            if N >= 3:
                # This will be handled by containment reward separately
                precision_reward += 1.0  # Base proximity bonus
            
            return precision_reward * 0.4
    
    def _compute_c_shape_formation_bonus(self, drone_poses, herd_center):
        """
        Compute bonus for C-shape formation around the herd.
        
        Args:
            drone_poses: Array of drone (x, y) positions
            herd_center: Herd centroid (x, y)
            
        Returns:
            float: Bonus reward for good C-shape positioning
        """
        N = len(drone_poses)
        if N < 3:
            return 0.0
        
        # Calculate angles of drones relative to herd center
        angles = []
        for drone_pos in drone_poses:
            dx = drone_pos[0] - herd_center[0]
            dy = drone_pos[1] - herd_center[1]
            angle = math.atan2(dy, dx)
            angles.append(angle)
        
        # Sort angles
        angles.sort()
        
        # Calculate angular gaps
        gaps = []
        for i in range(len(angles)):
            next_i = (i + 1) % len(angles)
            gap = angles[next_i] - angles[i]
            if gap < 0:
                gap += 2 * math.pi
            gaps.append(gap)
        
        # Find largest gap (desired escape route)
        max_gap = max(gaps)
        
        # Calculate coverage (should be 60-80% for good C-shape)
        coverage = (2 * math.pi - max_gap) / (2 * math.pi)
        
        # Reward C-shape coverage between 60-80%
        if 0.6 <= coverage <= 0.8:
            return (coverage - 0.5) * 4.0  # Peak reward at 70% coverage
        else:
            return 0.0
    
    def _compute_action_magnitude_reward(self, center_distance):
        """
        Reward drones for taking meaningful actions, especially when far from herd.
        Combat "lazy" behavior where drones barely move.
        
        Args:
            center_distance: Distance from drone formation to herd center
            
        Returns:
            float: Reward for active behavior
        """
        # Get last actions if available
        if not hasattr(self, 'last_actions') or self.last_actions is None:
            return 0.0
        
        action_reward = 0.0
        
        # Calculate average action magnitude
        action_magnitudes = []
        for action in self.last_actions:
            if isinstance(action, np.ndarray):
                magnitude = np.linalg.norm(action)
            else:
                magnitude = abs(action)
            action_magnitudes.append(magnitude)
        
        avg_action_magnitude = np.mean(action_magnitudes)
        
        # Reward active behavior, especially when far from herd
        if center_distance > 4.0:
            # Far from herd - strongly reward active movement
            if avg_action_magnitude > 0.1:  # Meaningful action
                action_reward += avg_action_magnitude * 5.0  # Strong bonus for activity
            else:
                action_reward -= 2.0  # Penalty for laziness when far
        elif center_distance > 2.0:
            # Medium distance - moderate reward for activity
            if avg_action_magnitude > 0.05:
                action_reward += avg_action_magnitude * 2.0
            else:
                action_reward -= 1.0  # Mild penalty for laziness
        
        # Cap the reward to prevent exploitation
        return np.clip(action_reward * 0.1, -0.5, 1.0)  # Weight factor 0.1
    
    ################################################################################
    
    def compute_formation_reward(self):
        """
        Compute spacing and cohesion reward for drone formation.
        
        Returns:
        - float: formation reward value
        """
        drone_states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        drones_poses = drone_states[:, :2]
        
        # Formation parameters
        target_spacing = 1.5  # Desired spacing between neighboring drones
        collision_threshold = 0.3  # Minimum safe distance
        max_reward_distance = 3.0  # Distance beyond which no reward is given
        
        N = drones_poses.shape[0]
        if N < 2:
            return 0.0
        
        total_reward = 0.0
        
        # Spacing reward: encourage optimal spacing between all pairs
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(drones_poses[i] - drones_poses[j])
                
                # Collision penalty (strong negative)
                if dist < collision_threshold:
                    total_reward -= 10.0
                # Optimal spacing reward (positive Gaussian)
                elif dist <= max_reward_distance:
                    spacing_reward = np.exp(-((dist - target_spacing) ** 2) / (2 * 0.5 ** 2))
                    total_reward += spacing_reward
        
        # Normalize by number of pairs
        pair_count = N * (N - 1) / 2
        total_reward = total_reward / pair_count if pair_count > 0 else 0.0
        
        # Add cohesion reward to prevent agents from flying away
        centroid = np.mean(drones_poses, axis=0)
        cohesion_reward = 0.0
        for i in range(N):
            dist_to_centroid = np.linalg.norm(drones_poses[i] - centroid)
            # Penalize agents that are too far from group centroid
            if dist_to_centroid > 3.0:
                cohesion_reward -= dist_to_centroid * 0.5
            else:
                cohesion_reward += 0.1  # Small reward for staying close
        
        cohesion_reward = cohesion_reward / N
        total_reward += cohesion_reward
        return total_reward

    ################################################################################
    
    def _computeReward(self):
        """
        Compute the reward for formation control in reinforcement learning.
        Includes formation maintenance, herding effectiveness, and stability rewards.
       
        Returns:
        - reward: float, total reward based on the formation and herding performance.
        """
        # Reset collision tracking for this step
        self._collision_detected = False
        self._collision_pairs = []
        # Get state data
        cattle_centroid = self.HerdCentroid()
        drone_states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        
        # Handle zero cattle case (Phase 1 curriculum learning)
        if self.NUM_CATTLE == 0:
            cattle_states = np.array([]).reshape(0, 13)  # Empty array with correct shape
            cattle_poses = np.array([]).reshape(0, 2)    # Empty array for positions
        else:
            cattle_states = np.array([self._getCowStateVector(i) for i in range(self.NUM_CATTLE)])
            # Ensure cattle_states is 2D even with 1 cow
            if cattle_states.ndim == 1:
                cattle_states = cattle_states.reshape(1, -1)
            cattle_poses = cattle_states[:, :2]

        drones_poses = drone_states[:, :2]  # x, y positions only
        
        N = self.NUM_DRONES
        if N < 2:
            return 0.0
        
        # Calculate herd-drone distance first (needed for distance_factor)
        herd_center = cattle_centroid[:2]
        drone_center = np.mean(drones_poses, axis=0)
        center_distance = np.linalg.norm(drone_center - herd_center)
        distance_factor = max(0.1, np.exp(-center_distance / 4.0))  # Reduce other rewards when far from herd
        
        # 1. FORMATION SPACING REWARD (Weight: 0.3) - ENHANCED COLLISION AVOIDANCE
        spacing_reward = 0.0
        target_spacing = 1.75  # Match spawn spacing
        
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(drones_poses[i] - drones_poses[j])
                
                # STRONG BUT REASONABLE COLLISION PENALTIES
                if dist < 0.25:  # Critical collision zone
                    spacing_reward -= 100.0  # Strong penalty
                    self._collision_detected = True
                    # Track collision details for curriculum learning
                    if not hasattr(self, '_collision_pairs'):
                        self._collision_pairs = []
                    self._collision_pairs.append((i, j, dist, 'critical'))
                elif dist < 0.4:  # Dangerous proximity  
                    spacing_reward -= 50.0  # Moderate penalty
                    self._collision_detected = True
                    if not hasattr(self, '_collision_pairs'):
                        self._collision_pairs = []
                    self._collision_pairs.append((i, j, dist, 'dangerous'))
                elif dist < 0.6:  # Too close zone
                    spacing_reward -= 20.0  # Mild penalty
                    if not hasattr(self, '_collision_pairs'):
                        self._collision_pairs = []
                    self._collision_pairs.append((i, j, dist, 'close'))
                elif dist < target_spacing * 2.5:  # Only reward reasonable distances
                    # Gaussian reward peaked at target spacing
                    spacing_reward += 2.0 * np.exp(-((dist - target_spacing) ** 2) / (2 * 0.6 ** 2))
        
        # Normalize by number of pairs and reduce weight when far from herd
        pair_count = N * (N - 1) / 2
        spacing_reward = (spacing_reward / pair_count) * 0.3 * distance_factor if pair_count > 0 else 0.0
        
        # 2. STAGED HERDING REWARD - APPROACH THEN FORMATION (Weight: 0.4)
        herd_centering_reward = self._compute_staged_herding_reward(center_distance, drones_poses, herd_center)
        
        # 3. FORMATION COHESION REWARD (Weight: 0.2) - ENHANCED ISOLATION PREVENTION
        cohesion_reward = 0.0
        drone_centroid = np.mean(drones_poses, axis=0)
        
        for i in range(N):
            dist_to_formation_center = np.linalg.norm(drones_poses[i] - drone_centroid)
            
            # MODERATE ISOLATION PENALTIES FOR STABILITY
            if dist_to_formation_center > 8.0:  # Severe isolation
                cohesion_reward -= 50.0  # Strong but reasonable penalty
            elif dist_to_formation_center > 6.0:  # Strong isolation  
                cohesion_reward -= 25.0  # Moderate penalty
            elif dist_to_formation_center > 4.5:  # Moderate isolation
                cohesion_reward -= 10.0  # Mild penalty
            elif dist_to_formation_center > 3.0:  # Slight isolation
                cohesion_reward -= dist_to_formation_center * 2.0  # Gradual penalty
            else:
                # Reward staying within reasonable formation bounds
                cohesion_reward += 0.5
        
        cohesion_reward = (cohesion_reward / N) * 0.15 * distance_factor
        
        # 4. HERDING EFFECTIVENESS REWARD (Weight: 0.1)
        # Reward for keeping cattle together
        effectiveness_reward = 0.0
        if self.NUM_CATTLE > 1:
            cattle_spread = 0.0
            cattle_centroid_2d = cattle_centroid[:2]
            for i in range(self.NUM_CATTLE):
                dist_to_herd_center = np.linalg.norm(cattle_poses[i] - cattle_centroid_2d)
                cattle_spread += dist_to_herd_center
            
            # Reward for keeping cattle compact
            avg_cattle_spread = cattle_spread / self.NUM_CATTLE
            effectiveness_reward = max(0, 2.0 - avg_cattle_spread) * 0.1
        
        # 5. FORMATION STRUCTURE REWARD (Weight: 0.1)
        structure_reward = self._compute_formation_structure_reward(drones_poses) * 0.1
        
        # 6. CONTAINMENT REWARD - EVALUATION MODE BOOST WITH DETAILED METRICS
        containment_reward = 0.0
        if N >= 3:  # Need at least 3 drones to form a meaningful polygon
            # Get detailed containment metrics for validation
            containment_metrics = self._compute_detailed_containment_metrics(cattle_poses, drones_poses)
            base_containment_reward = containment_metrics['reward']
            
            if hasattr(self, 'training_mode') and not self.training_mode:
                # EVALUATION MODE: Massive boost to containment reward to encourage surrounding
                containment_reward = base_containment_reward * 0.5  # 10x boost!
                
                # Detailed logging every 100 steps for visual validation
                if self.step_counter % 100 == 0:
                    metrics = containment_metrics
                    print(f"🎯 [CONTAINMENT ANALYSIS] Step {self.step_counter}")
                    print(f"  📊 Cattle contained: {metrics['contained_count']}/{metrics['total_cattle']} ({metrics['containment_percentage']:.1%})")
                    print(f"  🔄 Angular coverage: {metrics['angular_coverage']:.1f}° (ideal: 360°)")
                    print(f"  📐 Drone angles: {[f'{a:.0f}°' for a in metrics['drone_angles_deg']]}")
                    print(f"  � Drone distances: avg={metrics['avg_drone_distance']:.1f}m, max={metrics['max_drone_distance']:.1f}m")
                    print(f"  �💰 Containment reward: {base_containment_reward:.2f} → {containment_reward:.2f} (10x boost)")
                    
                    # Enhanced visual formation assessment with distance validation
                    if not metrics['effective_containment']:
                        print(f"  🚫 NO CONTAINMENT: Drones too far (avg: {metrics['avg_drone_distance']:.1f}m, max: {metrics['max_drone_distance']:.1f}m)")
                    elif metrics['containment_percentage'] > 0.8 and metrics['angular_coverage'] > 270:
                        print(f"  ✅ EXCELLENT ENCIRCLEMENT: {metrics['containment_percentage']:.1%} contained, {metrics['angular_coverage']:.0f}° coverage")
                    elif metrics['containment_percentage'] > 0.8:
                        print(f"  ⚠️  GOOD CONTAINMENT, POOR SPREAD: {metrics['containment_percentage']:.1%} contained, only {metrics['angular_coverage']:.0f}° coverage")
                    elif metrics['containment_percentage'] > 0.5:
                        print(f"  🔶 PARTIAL ENCIRCLEMENT: {metrics['containment_percentage']:.1%} cattle contained")
                    else:
                        print(f"  ❌ POOR ENCIRCLEMENT: {metrics['containment_percentage']:.1%} cattle contained")
                        
            else:
                # TRAINING MODE: Original reduced weight
                containment_reward = base_containment_reward * 0.05
        
        # Apply collision penalty multiplier - severely reduce all positive rewards during collisions
        collision_multiplier = 1.0
        if hasattr(self, '_collision_detected') and self._collision_detected:
            collision_multiplier = 0.0  # Zero out ALL positive rewards during collisions
            self._collision_detected = False  # Reset for next step
        
        # Apply collision multiplier to positive rewards only
        if herd_centering_reward > 0:
            herd_centering_reward *= collision_multiplier
        if cohesion_reward > 0:
            cohesion_reward *= collision_multiplier
        if effectiveness_reward > 0:
            effectiveness_reward *= collision_multiplier
        if structure_reward > 0:
            structure_reward *= collision_multiplier
        if containment_reward > 0:
            containment_reward *= collision_multiplier
        
        # 7. ACTION MAGNITUDE REWARD - Combat "lazy" behavior
        action_magnitude_reward = self._compute_action_magnitude_reward(center_distance)
        
        # Combine all rewards
        total_reward = spacing_reward + herd_centering_reward + cohesion_reward + effectiveness_reward + structure_reward + containment_reward + action_magnitude_reward
        
        # Add small positive baseline to encourage exploration
        total_reward += 0.1
        
        # Store reward components for logging
        if not hasattr(self, 'reward_components'):
            self.reward_components = {
                'spacing': [],
                'centering': [],
                'cohesion': [],
                'effectiveness': [],
                'structure': [],
                'containment': [],
                'action_magnitude': []
            }
        
        self.reward_components['spacing'].append(spacing_reward)
        self.reward_components['centering'].append(herd_centering_reward)
        self.reward_components['cohesion'].append(cohesion_reward)
        self.reward_components['effectiveness'].append(effectiveness_reward)
        self.reward_components['structure'].append(structure_reward)
        self.reward_components['containment'].append(containment_reward)
        self.reward_components['action_magnitude'].append(action_magnitude_reward)
        
        # Keep only last 100 values for efficiency
        for key in self.reward_components:
            if len(self.reward_components[key]) > 100:
                self.reward_components[key] = self.reward_components[key][-100:]
        
        # Debug output disabled for clean training logs
        # if spacing_reward < -50.0 or cohesion_reward < -50.0:
        #     print(f"[MAJOR PENALTY] Spacing={spacing_reward:.1f}, Cohesion={cohesion_reward:.1f}, Total={total_reward:.1f}")
        
        return total_reward

    def _compute_formation_structure_reward(self, drones_poses):
        """
        Reward for maintaining structured formations (line, V-shape, etc.)
        """
        N = len(drones_poses)
        if N < 3:
            return 0.0
        
        # Try different formation patterns and return best reward
        line_reward = self._evaluate_line_formation(drones_poses)
        v_reward = self._evaluate_v_formation(drones_poses)
        
        # Return the best formation reward
        return max(line_reward, v_reward)
    
    def _evaluate_line_formation(self, drones_poses):
        """Evaluate how well drones form a line"""
        N = len(drones_poses)
        if N < 3:
            return 0.0
        
        # Sort drones by x-coordinate to find line direction
        sorted_indices = np.argsort(drones_poses[:, 0])
        sorted_poses = drones_poses[sorted_indices]
        
        # Calculate line from first to last drone
        if N == 2:
            return 1.0  # Two drones always form a perfect line
        
        start_point = sorted_poses[0]
        end_point = sorted_poses[-1]
        line_vector = end_point - start_point
        line_length = np.linalg.norm(line_vector)
        
        if line_length < 0.1:  # Drones too close together
            return 0.0
        
        line_unit = line_vector / line_length
        
        # Calculate deviation from line for each intermediate drone
        total_deviation = 0.0
        for i in range(1, N-1):
            point = sorted_poses[i]
            # Vector from start to this point
            to_point = point - start_point
            # Project onto line direction
            projection_length = np.dot(to_point, line_unit)
            projection_point = start_point + projection_length * line_unit
            # Distance from line
            deviation = np.linalg.norm(point - projection_point)
            total_deviation += deviation
        
        # Convert to reward (lower deviation = higher reward)
        avg_deviation = total_deviation / max(1, N-2)
        line_reward = np.exp(-avg_deviation / 0.5)  # Decay with deviation
        
        return line_reward
    
    def _evaluate_v_formation(self, drones_poses):
        """Evaluate how well drones form a V-formation"""
        N = len(drones_poses)
        if N < 3:
            return 0.0
        
        # Find the drone that could be the apex (furthest forward in y-direction)
        center_y = np.mean(drones_poses[:, 1])
        apex_candidates = []
        
        for i, pos in enumerate(drones_poses):
            if pos[1] > center_y - 0.5:  # Near or ahead of center
                apex_candidates.append((i, pos))
        
        if not apex_candidates:
            return 0.0
        
        best_v_reward = 0.0
        
        for apex_idx, apex_pos in apex_candidates:
            # Calculate V-formation score with this apex
            other_drones = [drones_poses[i] for i in range(N) if i != apex_idx]
            
            if len(other_drones) < 2:
                continue
            
            # Split other drones into left and right wings
            left_wing = []
            right_wing = []
            
            for pos in other_drones:
                relative_x = pos[0] - apex_pos[0]
                if relative_x < -0.2:  # Left wing
                    left_wing.append(pos)
                elif relative_x > 0.2:  # Right wing
                    right_wing.append(pos)
            
            if len(left_wing) == 0 or len(right_wing) == 0:
                continue
            
            # Check symmetry and alignment
            v_score = 0.0
            
            # Reward balanced wings
            wing_balance = 1.0 - abs(len(left_wing) - len(right_wing)) / max(len(left_wing), len(right_wing))
            v_score += wing_balance * 0.5
            
            # Reward proper V-angle (wings behind apex)
            left_behind = all(pos[1] < apex_pos[1] + 0.5 for pos in left_wing)
            right_behind = all(pos[1] < apex_pos[1] + 0.5 for pos in right_wing)
            
            if left_behind and right_behind:
                v_score += 0.5
            
            best_v_reward = max(best_v_reward, v_score)
        
        return best_v_reward

    ################################################################################


    # def _computeReward(self):
    #     #Data
    #     cattle_centroid = self.HerdCentroid()
    #     drone_centroid = self.DroneCentroid()
    #     drone_states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
    #     cattle_states = np.array([self._getCowStateVector(i) for i in range(self.NUM_CATTLE)])

    #     drones_poses = drone_states[:, :2]
    #     cattle_poses = cattle_states[:, :2]

    #     #Drone to Drone Spacing
    #     drone_to_drone_spacing_reward = 0.0
    #     for i in range(self.NUM_DRONES):
    #         pos_i = drones_poses[i]

    #         other_dists = np.linalg.norm(drones_poses[:] - pos_i, axis=1)
    #         other_dists[i] = np.inf  # ignore self

    #         nearest_two = np.partition(other_dists, 1)[:2]  # get two smallest distances
    #         for dist in nearest_two:
    #             rwd = self.SpacingRewardValue(dist)
    #             drone_to_drone_spacing_reward += rwd

    #     #Average over all drones and terms
    #     drone_to_drone_spacing_reward /= self.NUM_DRONES * 2  # 2 nearest

    #     # #Centroid Distance Reward
    #     # # cent_dist = np.linalg.norm(drone_centroid - cattle_centroid)
    #     # # if self.prev_cent_dists is None:
    #     # #     self.prev_cent_dists = cent_dist
    #     # # cent_dist_change = self.prev_cent_dists - cent_dist
    #     # # centroid_distance_reward = cent_dist_change / (0.2 + 1e-6)
    #     # # self.prev_cent_dists = cent_dist

    #     # cent_dist = np.linalg.norm(drone_centroid - cattle_centroid)
    #     # centroid_distance_reward = 1.0 - (cent_dist / (self.MAX_DIST + 1e-6))
    #     # centroid_distance_reward = np.clip(centroid_distance_reward, -1.0, 1.0)

    #     # #Drone to Cattle Spacing
    #     # drone_to_cattle_spacing_reward = 0.0
    #     # for i in range(self.NUM_DRONES):
    #     #     pos_i = drones_poses[i]
    #     #     dists_to_cattle = np.linalg.norm(cattle_poses - pos_i, axis=1)
    #     #     closest_dist = np.min(dists_to_cattle)
    #     #     drone_to_cattle_spacing_reward += self.SpacingRewardValue(closest_dist)

    #     # #Average over drones
    #     # drone_to_cattle_spacing_reward /= self.NUM_DRONES

    #     small_step_penalty = 0.01

    #     # #Combine Rewards
    #     # r = (
    #     #      centroid_distance_reward * self.REWARD_WEIGHTS["centroid_distance"]
    #     #      + drone_to_drone_spacing_reward * self.REWARD_WEIGHTS["drone_to_drone_spacing"]
    #     #      + drone_to_cattle_spacing_reward * self.REWARD_WEIGHTS["drone_to_cattle_spacing"]
    #     #      - small_step_penalty
    #     # )

    #     # Use formation reward instead of old spacing reward
    #     formation_reward = self.compute_formation_reward()
        
    #     #Combine Rewards
    #     r = (
    #          formation_reward * 2.0  # Scale up formation reward
    #          - small_step_penalty
    #     )

    #     # Check if the small step penalty is grater than drone to drone spacing reward
    #     if abs(drone_to_drone_spacing_reward) < small_step_penalty:
    #         #Print warning
    #         print(f"WARNING: Small step penalty ({small_step_penalty}) is greater than drone to drone spacing reward ({drone_to_drone_spacing_reward}). Adjust reward weights.")


    #     #End of Episode Rewards
    #     done = self._computeTerminated() or self._computeTruncated()
    #     if done:

    #         # #Reward for drone centorid being near cattle centroid
    #         # if cent_dist < self.TERMINATION_CENTROID_THRESH:
    #         #     r+= 50
    #         # else:
    #         #     r -= cent_dist * 1.5

    #         #Reward for effectivness
    #         # effectiveness = self.eval_system.calculate_effectiveness(cattle_poses,drones_poses)
    #         effectiveness = self.eval_system.calculate_effectiveness_spacing(drones_poses)
    #         if effectiveness == 100: #missive reward for herding all cattle
    #             r += 100
    #         elif effectiveness == 0: #negative reward for herding nothing
    #             r -= 25
    #         else:
    #             r +=  effectiveness/10 #bonus 1 - 9 points for number of cattle herded

    #     return float(r)

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value based on mission success."""

        # During evaluation mode, disable early termination to see full behavior
        if hasattr(self, 'training_mode') and not self.training_mode:
            return False

        drone_states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        
        # Handle zero cattle case (Phase 1 curriculum learning)  
        if self.NUM_CATTLE == 0:
            cattle_states = np.array([]).reshape(0, 13)
            cattle_poses = np.array([]).reshape(0, 2)
        else:
            cattle_states = np.array([self._getCowStateVector(i) for i in range(self.NUM_CATTLE)])
            if cattle_states.ndim == 1:
                cattle_states = cattle_states.reshape(1, -1)
            cattle_poses = cattle_states[:, :2]

        drones_poses = drone_states[:, :2]

        cattle_centroid = self.HerdCentroid()
        drone_centroid = self.DroneCentroid()
        cent_dist = np.linalg.norm(drone_centroid - cattle_centroid, axis=-1)
        
        # SUCCESS CONDITION 1: Perfect formation + herding (strict)
        if cent_dist < 0.5:  # Relaxed from 0.25 to allow learning
            effectiveness = self.eval_system.calculate_effectiveness(cattle_poses, drones_poses)
            if effectiveness > 85:
                # Check if formation is maintained
                formation_quality = self._evaluate_formation_quality(drones_poses)
                if formation_quality > 0.7:
                    print(f"SUCCESS: Perfect formation + herding! Effectiveness: {effectiveness:.1f}, Formation: {formation_quality:.2f}")
                    return True

        # SUCCESS CONDITION 2: Sustained good performance (during training)
        if hasattr(self, 'training_mode') and self.training_mode:
            # Track recent reward performance
            if not hasattr(self, 'recent_rewards'):
                self.recent_rewards = []
            
            current_reward = self._computeReward()
            self.recent_rewards.append(current_reward)
            
            # Keep only last 50 steps
            if len(self.recent_rewards) > 50:
                self.recent_rewards.pop(0)
            
            # Early success if sustained high performance
            if len(self.recent_rewards) >= 50:
                avg_reward = np.mean(self.recent_rewards)
                if avg_reward > 1.5:  # High average reward
                    print(f"SUCCESS: Sustained good performance! Avg reward: {avg_reward:.2f}")
                    return True

        return False
    
    def _evaluate_formation_quality(self, drones_poses):
        """Evaluate overall formation quality (0-1 score)"""
        N = len(drones_poses)
        if N < 2:
            return 1.0
        
        # Check spacing consistency
        spacing_score = 0.0
        target_spacing = 1.75
        pair_count = 0
        
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(drones_poses[i] - drones_poses[j])
                # Score based on how close to target spacing
                spacing_score += np.exp(-((dist - target_spacing) ** 2) / (2 * 0.5 ** 2))
                pair_count += 1
        
        spacing_score = spacing_score / pair_count if pair_count > 0 else 0.0
        
        # Check formation structure
        structure_score = max(
            self._evaluate_line_formation(drones_poses),
            self._evaluate_v_formation(drones_poses)
        )
        
        # Combine scores
        return (spacing_score * 0.6 + structure_score * 0.4)


    ################################################################################
    
    def _computeTruncated(self):
        """Computes whether the current episode should be truncated due to unsafe drone states.

        Returns
        -------
        bool
            True if the episode should be truncated, False otherwise.
        """
        cattle_centroid = self.HerdCentroid()
        drone_centroid = self.DroneCentroid()
        cent_dist = np.linalg.norm(drone_centroid - cattle_centroid, axis=-1)

        drone_states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        
        # Handle zero cattle case (Phase 1 curriculum learning)
        if self.NUM_CATTLE == 0:
            cattle_states = np.array([]).reshape(0, 13)
            cattle_poses = np.array([]).reshape(0, 2)
        else:
            cattle_states = np.array([self._getCowStateVector(i) for i in range(self.NUM_CATTLE)])
            if cattle_states.ndim == 1:
                cattle_states = cattle_states.reshape(1, -1)
            cattle_poses = cattle_states[:, :2]

        drones_poses = drone_states[:, :2]

        # FAILURE 1: Altitude safety (disabled during evaluation to see full behavior)
        if hasattr(self, 'training_mode') and self.training_mode:
            for i in range(self.NUM_DRONES):
                z = drone_states[i][2]
                if abs(z - self.DRONE_TARGET_ALTITUDE) > self.MAX_ALT_ERROR:
                    print(f"TRUNCATED: Drone {i} altitude loss: {z:.2f}m (target: {self.DRONE_TARGET_ALTITUDE:.2f}m)")
                    return True    

        # FAILURE 2: Collision detection (actual collisions, not just distance)
        collision_threshold = 0.2  # Very close proximity
        for i in range(self.NUM_DRONES):
            for j in range(i + 1, self.NUM_DRONES):
                dist = np.linalg.norm(drones_poses[i] - drones_poses[j])
                if dist < collision_threshold:
                    print(f"TRUNCATED: Collision between drones {i} and {j}: {dist:.2f}m")
                    return True

        # FAILURE 3: Formation breakdown (disabled during evaluation to see full behavior)
        if hasattr(self, 'training_mode') and self.training_mode:
            formation_broken = False
            max_formation_distance = 8.0  # Increased from 5.0 for more flexibility
            
            for i in range(self.NUM_DRONES):
                pos_i = drones_poses[i]
                other_dists = np.linalg.norm(drones_poses - pos_i, axis=1)
                other_dists[i] = np.inf  # Ignore self
                
                # Check if drone is isolated (too far from ALL others)
                if np.all(other_dists > max_formation_distance):
                    print(f"TRUNCATED: Drone {i} isolated from formation, min distance: {np.min(other_dists):.2f}m")
                    formation_broken = True
                    break
            
            if formation_broken:
                return True

        # FAILURE 4: Mission area boundary (disabled during evaluation to see full behavior)
        if hasattr(self, 'training_mode') and self.training_mode:
            mission_boundary = 15.0  # Maximum distance from herd center
            if cent_dist > mission_boundary:
                print(f"TRUNCATED: Formation too far from herd: {cent_dist:.2f}m")
                return True

        # FAILURE 5: Sustained poor performance (during training)
        if hasattr(self, 'training_mode') and self.training_mode:
            if not hasattr(self, 'recent_rewards'):
                self.recent_rewards = []
            
            # Track recent rewards (updated in _computeTerminated)
            if len(self.recent_rewards) >= 100:  # Longer window for failure
                avg_reward = np.mean(self.recent_rewards[-100:])
                if avg_reward < -0.5:  # Consistently poor performance
                    print(f"TRUNCATED: Sustained poor performance: {avg_reward:.2f}")
                    return True
        
        # TIMEOUT: Episode time limit
        elapsed_sec = self.step_counter / self.CTRL_FREQ
        if elapsed_sec > self.EPISODE_LEN_SEC:
            if self.is_evaluating:
                self.evaluation_episode_trigger()
                print("TRUNCATED: Episode time limit reached!")
            return True

        return False


    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
    
    ################################################################################

    def SpacingRewardValue(self, r):
        """
        Compute Spacing Reward
            
        Returns:
        float
        """
        A = self.SPACING_A
        B = self.SPACING_B
        c = self.SPACING_C
        k = self.SPACING_K
        d = self.SPACING_D
        r0 = self.SPACING_R0
        lam = self.SPACING_LAM

        if r <= r0:
            return A * np.exp(-((r - d)**2) / (2 * c**2)) - B * np.exp(-(r**2) / (2 * k**2))
        else:
            fr0 = A * np.exp(-((r0 - d)**2) / (2 * c**2)) - B * np.exp(-(r0**2) / (2 * k**2))
            C = fr0 / np.exp(-lam * r0)
            return C * np.exp(-lam * r)
        
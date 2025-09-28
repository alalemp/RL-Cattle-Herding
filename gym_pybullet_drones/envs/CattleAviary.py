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
        # Get state data
        cattle_centroid = self.HerdCentroid()
        drone_states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        cattle_states = np.array([self._getCowStateVector(i) for i in range(self.NUM_CATTLE)])

        drones_poses = drone_states[:, :2]  # x, y positions only
        cattle_poses = cattle_states[:, :2]
        
        N = self.NUM_DRONES
        if N < 2:
            return 0.0
        
        # Calculate herd-drone distance first (needed for distance_factor)
        herd_center = cattle_centroid[:2]
        drone_center = np.mean(drones_poses, axis=0)
        center_distance = np.linalg.norm(drone_center - herd_center)
        distance_factor = max(0.1, np.exp(-center_distance / 4.0))  # Reduce other rewards when far from herd
        
        # 1. FORMATION SPACING REWARD (Weight: 0.3)
        spacing_reward = 0.0
        target_spacing = 1.75  # Match spawn spacing
        collision_threshold = 0.3
        
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(drones_poses[i] - drones_poses[j])
                
                if dist < collision_threshold:
                    # Strong collision penalty
                    spacing_reward -= 5.0
                elif dist < target_spacing * 2.5:  # Only reward reasonable distances
                    # Gaussian reward peaked at target spacing
                    spacing_reward += 2.0 * np.exp(-((dist - target_spacing) ** 2) / (2 * 0.6 ** 2))
        
        # Normalize by number of pairs and reduce weight when far from herd
        pair_count = N * (N - 1) / 2
        spacing_reward = (spacing_reward / pair_count) * 0.3 * distance_factor if pair_count > 0 else 0.0
        
        # 2. HERD CENTERING REWARD (Weight: 0.5 - INCREASED!)
        
        # Much stronger herd centering with penalty for being far
        if center_distance < 2.0:
            # Strong positive reward for being close
            herd_centering_reward = (2.0 - center_distance) * 2.0 * 0.5
        else:
            # Strong negative penalty for being far
            herd_centering_reward = -center_distance * 0.3
        
        # 3. FORMATION COHESION REWARD (Weight: 0.2)
        cohesion_reward = 0.0
        drone_centroid = np.mean(drones_poses, axis=0)
        
        for i in range(N):
            dist_to_formation_center = np.linalg.norm(drones_poses[i] - drone_centroid)
            # Reward staying within reasonable formation bounds
            if dist_to_formation_center < 4.0:
                cohesion_reward += 0.5
            else:
                # Penalty for being too far from formation
                cohesion_reward -= dist_to_formation_center * 0.2
        
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
        
        # 6. CONTAINMENT REWARD (Weight: 0.15) - NEW WINDING NUMBER BASED (PHASE 2)
        containment_reward = 0.0
        if N >= 3:  # Need at least 3 drones to form a meaningful polygon
            containment_reward = self._compute_containment_reward(cattle_poses, drones_poses) * 0.15
        
        # Combine all rewards
        total_reward = spacing_reward + herd_centering_reward + cohesion_reward + effectiveness_reward + structure_reward + containment_reward
        
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
                'containment': []
            }
        
        self.reward_components['spacing'].append(spacing_reward)
        self.reward_components['centering'].append(herd_centering_reward)
        self.reward_components['cohesion'].append(cohesion_reward)
        self.reward_components['effectiveness'].append(effectiveness_reward)
        self.reward_components['structure'].append(structure_reward)
        self.reward_components['containment'].append(containment_reward)
        
        # Keep only last 100 values for efficiency
        for key in self.reward_components:
            if len(self.reward_components[key]) > 100:
                self.reward_components[key] = self.reward_components[key][-100:]
        
        # Debug output every 100 steps (optional) - COMMENTED OUT FOR PERFORMANCE
        # if hasattr(self, 'step_counter') and self.step_counter % 100 == 0:
        #     print(f"[DEBUG] Step {self.step_counter}: Spacing={spacing_reward:.3f}, Centering={herd_centering_reward:.3f}, "
        #           f"Cohesion={cohesion_reward:.3f}, Structure={structure_reward:.3f}, Containment={containment_reward:.3f}, Total={total_reward:.3f}")
        
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

        drone_states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        cattle_states = np.array([self._getCowStateVector(i) for i in range(self.NUM_CATTLE)])

        drones_poses = drone_states[:, :2] 
        cattle_poses = cattle_states[:, :2]

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
        cattle_states = np.array([self._getCowStateVector(i) for i in range(self.NUM_CATTLE)])

        drones_poses = drone_states[:, :2] 
        cattle_poses = cattle_states[:, :2]

        # FAILURE 1: Altitude safety
        for i in range(self.NUM_DRONES):
            z = drone_states[i][2]
            if abs(z - self.DRONE_TARGET_ALTITUDE) > self.MAX_ALT_ERROR:
                if self.is_evaluating:
                    self.evaluation_episode_trigger()
                    print(f"TRUNCATED: Drone {i} altitude loss: {z:.2f}m")
                return True    

        # FAILURE 2: Collision detection (actual collisions, not just distance)
        collision_threshold = 0.2  # Very close proximity
        for i in range(self.NUM_DRONES):
            for j in range(i + 1, self.NUM_DRONES):
                dist = np.linalg.norm(drones_poses[i] - drones_poses[j])
                if dist < collision_threshold:
                    print(f"TRUNCATED: Collision between drones {i} and {j}: {dist:.2f}m")
                    return True

        # FAILURE 3: Formation breakdown (drones too far apart)
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

        # FAILURE 4: Mission area boundary
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
        
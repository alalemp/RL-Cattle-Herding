import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from collections import deque

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType

class BaseRLAviary(BaseAviary):
    """Base single and multi-agent environment class for reinforcement learning."""
    
    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=2,
                 num_cattle: int =1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.COKIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a generic single and multi-agent RL environment.

        Attributes `vision_attributes` and `dynamics_attributes` are selected
        based on the choice of `obs` and `act`; `obstacles` is set to True 
        and overridden with landmarks for vision applications; 
        `user_debug_gui` is set to False for performance.

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
            The type of action space (1 or 3D; RPMS, thurst and torques, waypoint or velocity with PID control; etc.)

        """
        #### Create a buffer for the last .5 sec of actions ########
        self.ACTION_BUFFER_SIZE = int(ctrl_freq//2)
        self.action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE)
        ####
        vision_attributes = True if obs == ObservationType.RGB else False
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        self.ACTION_SPACE = 4
        #### Create integrated controllers #########################
        if act in [ActionType.PID, ActionType.VEL, ActionType.ONE_D_PID]:
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
                self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]
            else:
                print("[ERROR] in BaseRLAviary.__init()__, no controller is available for the specified drone_model")
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
                         obstacles=True, # Add obstacles for RGB observations and/or FlyThruGate
                         user_debug_gui=False, # Remove of RPM sliders from all single agent learning aviaries
                         vision_attributes=vision_attributes,
                         )
        #### Set a limit on the maximum target speed ###############
        if act == ActionType.VEL:
            # Increased speed limit for cattle herding - need to cover larger distances
            if hasattr(self, 'training_mode') and not self.training_mode:
                # EVALUATION MODE: Much higher speed limit for override system
                self.SPEED_LIMIT = 2.0 * self.MAX_SPEED_KMH * (1000/3600)  # ~16.7 m/s for evaluation
                print(f"[EVAL MODE] Speed limit increased to {self.SPEED_LIMIT:.1f} m/s for aggressive override")
            else:
                # TRAINING MODE: Normal speed limit
                self.SPEED_LIMIT = 0.6 * self.MAX_SPEED_KMH * (1000/3600)  # 5.0 m/s for training

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            A Box of size NUM_DRONES x 4, 3, or 1, depending on the action type.

        """
        if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
            size = 4
            self.ACTION_SPACE = 4
        elif self.ACT_TYPE==ActionType.PID:
            size = 3
            self.ACTION_SPACE = 3
        elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
            size = 1
            self.ACTION_SPACE = 1
        else:
            print("[ERROR] in BaseRLAviary._actionSpace()")
            exit()
        act_lower_bound = np.array([-1*np.ones(size) for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([+1*np.ones(size) for i in range(self.NUM_DRONES)])
        #
        # Clear existing action buffer and reinitialize with correct dimensions
        self.action_buffer.clear()
        for i in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros((self.NUM_DRONES,size)))
        #
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    ###############################################################################

    def _reinitializeControllers(self):
        """Reinitialize controllers when NUM_DRONES changes."""
        if hasattr(self, 'ACT_TYPE') and self.ACT_TYPE in [ActionType.PID, ActionType.VEL, ActionType.ONE_D_PID]:
            if hasattr(self, 'DRONE_MODEL') and self.DRONE_MODEL in [DroneModel.CF2X, DroneModel.CF2P]:
                self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(self.NUM_DRONES)]

    ###############################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: the input to n-th drone, `action[n]` can be of length
        1, 3, or 4, and represent RPMs, desired thrust and torques, or the next
        target position to reach using PID control.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, or 4 and represent 
        RPMs, desired thrust and torques, the next target position to reach 
        using PID control, a desired velocity vector, etc.

        Parameters
        ----------
        action : ndarray
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        self.action_buffer.append(action)
        rpm = np.zeros((self.NUM_DRONES,4))
        for k in range(self.NUM_DRONES):
            target = action[k, :]
            if self.ACT_TYPE == ActionType.RPM:
                rpm[k,:] = np.array(self.HOVER_RPM * (1+0.05*target))
            elif self.ACT_TYPE == ActionType.PID:
                state = self._getDroneStateVector(k)
                next_pos = self._calculateNextStep(
                    current_position=state[0:3],
                    destination=target,
                    step_size=1,
                    )
                rpm_k, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=next_pos
                                                        )
                rpm[k,:] = rpm_k
            elif self.ACT_TYPE == ActionType.VEL:
                state = self._getDroneStateVector(k)
                # Normalize horizontal component of velocity target
                horiz_vel = target[0:2]
                norm = np.linalg.norm(horiz_vel)
                if norm != 0:
                    v_unit_xy = horiz_vel / norm
                else:
                    v_unit_xy = np.zeros(2)

                # Construct full 3D target velocity, zero vertical component since altitude is fixed
                target_vel = np.array([
                    v_unit_xy[0],
                    v_unit_xy[1],
                    0.0
                ]) * (self.SPEED_LIMIT * abs(target[3]))

                # Define fixed target position: current x, y plus desired altitude
                target_pos = np.array([
                    state[0],
                    state[1],
                    self.DRONE_TARGET_ALTITUDE
                ])

                # Keep current yaw by using current yaw angle (state[9] assumed to be yaw)
                target_rpy = np.array([0.0, 0.0, state[9]])

                temp, _, _ = self.ctrl[k].computeControl(
                    control_timestep=self.CTRL_TIMESTEP,
                    cur_pos=state[0:3],
                    cur_quat=state[3:7],
                    cur_vel=state[10:13],
                    cur_ang_vel=state[13:16],
                    target_pos=target_pos,
                    target_rpy=target_rpy,
                    target_vel=target_vel
                )
                rpm[k, :] = temp
            elif self.ACT_TYPE == ActionType.ONE_D_RPM:
                rpm[k,:] = np.repeat(self.HOVER_RPM * (1+0.05*target), 4)
            elif self.ACT_TYPE == ActionType.ONE_D_PID:
                state = self._getDroneStateVector(k)
                res, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3]+0.1*np.array([0,0,target[0]])
                                                        )
                rpm[k,:] = res
            else:
                print("[ERROR] in BaseRLAviary._preprocessAction()")
                exit()
        return rpm


    ################################################################################
   
    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        gym.spaces.Box
            A Box() of shape (MAX_NUM_DRONES, obs_length), where obs_length depends on observation type.
        """
        if self.OBS_TYPE != ObservationType.COKIN:
            print("[ERROR] in BaseRLAviary._observationSpace()")
            return None
        
        lo = -np.inf
        hi = np.inf

        # Own state vector: x, y, z + roll, pitch, yaw + linear vel + angular vel
        own_state_len = 3 + 3 + 3 + 3  # 12

        # Total observation length: own state + neighbors + nearby cattle + action buffer
        obs_len = own_state_len + self.MAX_NEIGHBORS * 2 + self.MAX_NEARBY_CATTLE * 2 + self.ACTION_BUFFER_SIZE * self.ACTION_SPACE

        obs_lower_bound = np.full((self.MAX_NUM_DRONES, obs_len), lo, dtype=np.float32)
        obs_upper_bound = np.full((self.MAX_NUM_DRONES, obs_len), hi, dtype=np.float32)

        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)


    ################################################################################

    def _computeObs(self):
        """Returns the current observations of all drones, padded to MAX_NUM_DRONES (decentralized)."""

        if self.OBS_TYPE != ObservationType.COKIN:
            print("[ERROR] in BaseRLAviary._computeObs()")
            return None

        N = self.NUM_DRONES
        M = self.NUM_CATTLE

        # Determine the length of each drone's observation
        obs_len_per_drone = 12 + self.MAX_NEIGHBORS*2 + self.MAX_NEARBY_CATTLE*2 + self.ACTION_BUFFER_SIZE * self.ACTION_SPACE

        # Pre-allocate full observation array with zeros
        obs = np.zeros((self.MAX_NUM_DRONES, obs_len_per_drone), dtype=np.float32)

        obs_dim = obs.shape[1]

        # Get herd centroid for relative positioning
        herd_center = self.HerdCentroid()[:2]  # x, y coordinates only
        
        for i in range(N):
            obs_vec = self._getDroneStateVector(i)
            drone_pos = obs_vec[0:3]
            
            # Make drone position relative to herd center for better generalization
            drone_pos_relative = drone_pos.copy()
            drone_pos_relative[0] -= herd_center[0]
            drone_pos_relative[1] -= herd_center[1]

            # Own state: herd-relative position, RPY, linear vel, angular vel
            obs_i = list(np.hstack([
                drone_pos_relative,   # x, y, z relative to herd center
                obs_vec[7:10],        # roll, pitch, yaw
                obs_vec[10:13],       # linear velocity vx, vy, vz
                obs_vec[13:16],       # angular velocity wx, wy, wz
            ]))

            # Relative positions of nearby drones (relative to current drone's herd-relative position)
            rel_neighbors = []
            for j in range(N):
                if i == j:
                    continue
                other_pos = self._getDroneStateVector(j)[0:2]  # x, y only
                # Make other drone position relative to herd center first
                other_pos_relative = other_pos - herd_center
                # Then relative to current drone's herd-relative position
                rel_neighbors.append(other_pos_relative - drone_pos_relative[0:2])
            while len(rel_neighbors) < self.MAX_NEIGHBORS:
                rel_neighbors.append(np.zeros(2))
            rel_neighbors = np.array(rel_neighbors[:self.MAX_NEIGHBORS]).flatten()
            obs_i.extend(rel_neighbors)

            # Relative positions of nearby cattle (relative to current drone's herd-relative position)
            rel_cattle = []
            for j in range(M):
                cow_pos = self._getCowStateVector(j)[0:2]
                # Make cattle position relative to herd center first
                cow_pos_relative = cow_pos - herd_center
                # Then relative to current drone's herd-relative position
                rel_cattle.append(cow_pos_relative - drone_pos_relative[0:2])
            while len(rel_cattle) < self.MAX_NEARBY_CATTLE:
                rel_cattle.append(np.zeros(2))
            rel_cattle = np.array(rel_cattle[:self.MAX_NEARBY_CATTLE]).flatten()
            obs_i.extend(rel_cattle)

            # Action buffer
            for k in range(self.ACTION_BUFFER_SIZE):
                obs_i.extend(self.action_buffer[k][i, :])

            # Truncate/pad to obs_dim
            obs_i_array = np.array(obs_i, dtype=np.float32)
            if len(obs_i_array) > obs_dim:
                obs_i_array = obs_i_array[:obs_dim]
            elif len(obs_i_array) < obs_dim:
                obs_i_array = np.pad(obs_i_array, (0, obs_dim - len(obs_i_array)), 'constant')

            obs[i, :] = obs_i_array

        # Debug observation space (optional, every 200 steps) - COMMENTED OUT FOR PERFORMANCE
        # if hasattr(self, 'step_counter') and self.step_counter % 200 == 0 and N > 0:
        #     print(f"[DEBUG] Obs - Herd center: {herd_center}, Drone 0 abs pos: {self._getDroneStateVector(0)[:2]}, "
        #           f"Drone 0 rel pos: {obs[0][:2]}")

        return obs

    ################################################################################  
    
    def HerdCentroid(self):
        """
        Calculates the center of the herd and updates the visual centroid marker to that location.
        """
        # Handle case where there are no cattle (e.g., Phase 1 curriculum learning)
        if self.NUM_CATTLE == 0:
            # Return origin as default centroid when no cattle present
            centroid = np.array([0.0, 0.0, self.DRONE_TARGET_ALTITUDE+0.5])
        else:
            cattle_states = np.array([self._getCowStateVector(i) for i in range(self.NUM_CATTLE)])
            # Ensure cattle_states is 2D even with 1 cow
            if cattle_states.ndim == 1:
                cattle_states = cattle_states.reshape(1, -1)
            
            cattle_positions = cattle_states[:, 0:3]
            
            # Compute XY centroid
            centroid_xy = np.mean(cattle_positions[:, :2], axis=0)
            centroid = np.array([centroid_xy[0], centroid_xy[1], self.DRONE_TARGET_ALTITUDE+0.5])

        # Update visual marker position (safely handle marker updates)
        try:
            if hasattr(self, 'CattleCentroidMarker') and self.CattleCentroidMarker is not None:
                p.resetBasePositionAndOrientation(
                    self.CattleCentroidMarker,
                    centroid,
                    p.getQuaternionFromEuler([0, 0, 0]),
                    physicsClientId=self.CLIENT
                )
        except Exception as e:
            # Marker update failed, continue without visual marker
            pass

        return centroid

    ################################################################################

    def DroneCentroid(self):
        """
        Calculates the center of the drones and updates the visual centroid marker to that location.
        """

        drone_states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        drone_positions = drone_states[:, 0:3]

        # Compute XY centroid
        centroid_xy = np.mean(drone_positions[:, :2], axis=0)
        centroid = np.array([centroid_xy[0], centroid_xy[1], self.DRONE_TARGET_ALTITUDE+0.5])

        # Update visual marker position
        p.resetBasePositionAndOrientation(
            self.DroneCentroidMarker,
            centroid,
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.CLIENT
        )

        return centroid


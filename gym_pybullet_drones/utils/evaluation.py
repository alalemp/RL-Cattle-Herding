import os
import pickle
import numpy as np

class evaluator():
    def __init__(self):
        # Episode-level
        self.total_drone_distances = []
        self.total_time_taken = []
        self.total_effectiveness = []
        self.total_number_of_drones = []
        self.total_drone_poses = []
        self.total_cattle_poses = []
        self.total_drone_vel = []
        self.total_cattle_vel = []

        # Timestep-level: store as list-of-lists per episode
        self.drone_distances_per_step = []
        self.effectiveness_per_step = []
        self.time_per_step = []
        self.drone_poses_per_step = []
        self.cattle_poses_per_step = []
        self.drone_vel_per_step = []
        self.cattle_vel_per_step = []

        # Temporary buffers for current episode
        self.curr_drone_poses = []
        self.curr_cattle_poses = []
        self.curr_drone_vel = []
        self.curr_cattle_vel = []
        self.curr_drone_distances = []
        self.curr_effectiveness = []
        self.curr_time = []


    # --- Called every timestep ---
    def append_timestep_data(self, drone_distances, timestep_time, effectiveness, drone_poses, cattle_poses, drone_vel, cattle_vel):
        self.curr_drone_distances.append(drone_distances)
        self.curr_time.append(timestep_time)
        self.curr_effectiveness.append(effectiveness)
        self.curr_drone_poses.append(drone_poses)
        self.curr_cattle_poses.append(cattle_poses)
        self.curr_drone_vel.append(drone_vel)
        self.curr_cattle_vel.append(cattle_vel)

    
    # --- Called at the end of an episode ---
    def append_episode_data(self, drone_distances, num_drones, time, effectiveness):
        # Save episode-level data
        self.total_drone_distances.append(drone_distances)
        self.total_number_of_drones.append(num_drones)
        self.total_time_taken.append(time)
        self.total_effectiveness.append(effectiveness)

        # Save timestep-level data for this episode
        self.drone_poses_per_step.append(self.curr_drone_poses)
        self.cattle_poses_per_step.append(self.curr_cattle_poses)
        self.drone_vel_per_step.append(self.curr_drone_vel)
        self.cattle_vel_per_step.append(self.curr_cattle_vel)
        self.drone_distances_per_step.append(self.curr_drone_distances)
        self.time_per_step.append(self.curr_time)
        self.effectiveness_per_step.append(self.curr_effectiveness)

        # Reset temporary buffers
        self.curr_drone_poses = []
        self.curr_cattle_poses = []
        self.curr_drone_vel = []
        self.curr_cattle_vel = []
        self.curr_drone_distances = []
        self.curr_effectiveness = []
        self.curr_time = []

    def save_evaluation_data(self, save_path="evaluation_data.pkl"):
        eval_data = {
            "distances": self.total_drone_distances,
            "num_drones": self.total_number_of_drones,
            "time_taken": self.total_time_taken,
            "effectiveness": self.total_effectiveness,

            "distances_per_step": self.drone_distances_per_step,
            "time_per_step": self.time_per_step,
            "effectiveness_per_step": self.effectiveness_per_step,

            "drone_poses_per_step": self.drone_poses_per_step,
            "cattle_poses_per_step": self.cattle_poses_per_step,
            "drone_vel_per_step": self.drone_vel_per_step,
            "cattle_vel_per_step": self.cattle_vel_per_step,

        }

        with open(save_path, "wb") as f:
            pickle.dump(eval_data, f)

        print(f"Evaluation data saved to {os.path.abspath(save_path)}")

    def calculate_effectiveness(self, cattle_poses, drones_poses):
        # Convert to numpy array
        polygon = np.array(drones_poses)
        cattle_poses = np.array(cattle_poses)

        # Ensure 2D shape: each row = [x, y]
        if polygon.ndim == 1:
            polygon = polygon.reshape(-1, 2)
        elif polygon.ndim == 3:  # sometimes hull returns shape (n,1,2)
            polygon = polygon.reshape(-1, 2)

        if cattle_poses.ndim == 1:
            cattle_poses = cattle_poses.reshape(-1, 2)
        elif cattle_poses.ndim == 3:
            cattle_poses = cattle_poses.reshape(-1, 2)

        total_herded_cattle = 0

        for cow_pose in cattle_poses:
            px, py = cow_pose
            wn = 0
            n = len(polygon)

            for i in range(n):
                x1, y1 = polygon[i]
                x2, y2 = polygon[(i + 1) % n]

                if y1 <= py:
                    if y2 > py and is_left((x1, y1), (x2, y2), (px, py)) > 0:
                        wn += 1
                else:
                    if y2 <= py and is_left((x1, y1), (x2, y2), (px, py)) < 0:
                        wn -= 1

            if wn:
                total_herded_cattle += 1 

        effectiveness = total_herded_cattle / len(cattle_poses) * 100 if len(cattle_poses) > 0 else 0
        return effectiveness

    def calculate_effectiveness_spacing(self, drones_poses):
        """
        Alternative effectiveness calculation based on drone spacing.
        Returns 100% effectiveness if the two closest neighboring drones are exactly 0.5 units apart.
        Uses a Gaussian-like function to smoothly decay effectiveness as distance deviates from 0.5.
        
        Parameters:
        -----------
        drones_poses : array-like
            Positions of drones, shape (n_drones, 2) for 2D coordinates
            
        Returns:
        --------
        float
            Effectiveness percentage (0-100)
        """
        # Convert to numpy array and ensure 2D shape
        drones_poses = np.array(drones_poses)
        
        if drones_poses.ndim == 1:
            drones_poses = drones_poses.reshape(-1, 2)
        elif drones_poses.ndim == 3:
            drones_poses = drones_poses.reshape(-1, 2)
        
        # Need at least 2 drones to calculate spacing
        if len(drones_poses) < 2:
            return 0.0
            
        # Calculate all pairwise distances between drones
        distances = []
        n_drones = len(drones_poses)
        
        for i in range(n_drones):
            for j in range(i + 1, n_drones):
                dist = np.linalg.norm(drones_poses[i] - drones_poses[j])
                distances.append(dist)
        
        # Find the minimum distance (closest neighbors)
        min_distance = min(distances)
        
        # Target distance for optimal spacing
        target_distance = 0.5
        
        # Calculate effectiveness using Gaussian-like decay
        # 100% at target distance, decreasing as distance deviates
        sigma = 0.1  # Controls how quickly effectiveness drops off
        effectiveness = 100 * np.exp(-((min_distance - target_distance) ** 2) / (2 * sigma ** 2))
        
        return float(effectiveness)



def is_left(p0, p1, p2):
    """Tests if p2 is left of the directed line p0 -> p1"""
    return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
            
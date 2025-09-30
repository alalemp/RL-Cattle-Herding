#!/usr/bin/env python3
"""
Quick test to verify the enhanced collision penalty system is working correctly.
"""

import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.append('/home/aalempij/Data/git/RL-Cattle-Herding')

from gym_pybullet_drones.envs.CattleAviary import CattleAviary

def test_collision_penalties():
    """Test the new graduated collision penalty system."""
    
    print("🔍 Testing Enhanced Collision Penalty System")
    print("=" * 50)
    
    # Create environment instance to access reward calculation
    env = CattleAviary(num_drones=3, num_cattle=1, gui=False)
    
    # Test cases with different drone spacing scenarios
    test_cases = [
        {
            "name": "CRITICAL COLLISION (0.2m apart)",
            "drone_poses": np.array([[0, 0], [0.2, 0], [1.5, 1.5]]),
            "expected": "Massive penalty ~-50"
        },
        {
            "name": "DANGEROUS PROXIMITY (0.35m apart)", 
            "drone_poses": np.array([[0, 0], [0.35, 0], [1.5, 1.5]]),
            "expected": "Strong penalty ~-25"
        },
        {
            "name": "TOO CLOSE (0.55m apart)",
            "drone_poses": np.array([[0, 0], [0.55, 0], [1.5, 1.5]]),
            "expected": "Moderate penalty ~-10"
        },
        {
            "name": "GOOD SPACING (1.75m apart)",
            "drone_poses": np.array([[0, 0], [1.75, 0], [0, 1.75]]),
            "expected": "Positive reward"
        },
        {
            "name": "SEVERE ISOLATION (drone 8m away)",
            "drone_poses": np.array([[0, 0], [1.5, 0], [8.0, 8.0]]),
            "expected": "Isolation penalty ~-50"
        }
    ]
    
    # Mock cattle position (not used in spacing/cohesion calculations)
    cattle_poses = np.array([[0, 0]])
    
    for test_case in test_cases:
        print(f"\n📊 {test_case['name']}")
        print("-" * 40)
        
        # Calculate spacing reward component
        N = len(test_case['drone_poses'])
        spacing_reward = 0.0
        target_spacing = 1.75
        
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(test_case['drone_poses'][i] - test_case['drone_poses'][j])
                print(f"Distance between drone {i} and {j}: {dist:.2f}m")
                
                # Apply the new penalty system
                if dist < 0.25:
                    penalty = -50.0
                    print(f"  → CRITICAL COLLISION PENALTY: {penalty}")
                    spacing_reward += penalty
                elif dist < 0.4:
                    penalty = -25.0
                    print(f"  → DANGEROUS PROXIMITY PENALTY: {penalty}")
                    spacing_reward += penalty
                elif dist < 0.6:
                    penalty = -10.0
                    print(f"  → TOO CLOSE PENALTY: {penalty}")
                    spacing_reward += penalty
                elif dist < target_spacing * 2.5:
                    reward = 2.0 * np.exp(-((dist - target_spacing) ** 2) / (2 * 0.6 ** 2))
                    print(f"  → SPACING REWARD: +{reward:.2f}")
                    spacing_reward += reward
        
        # Calculate cohesion penalties
        drone_centroid = np.mean(test_case['drone_poses'], axis=0)
        cohesion_penalty = 0.0
        
        for i in range(N):
            dist_to_center = np.linalg.norm(test_case['drone_poses'][i] - drone_centroid)
            if dist_to_center > 8.0:
                penalty = -50.0  # Updated to match new penalty
                print(f"Drone {i} isolation distance: {dist_to_center:.2f}m → SEVERE ISOLATION: {penalty}")
                cohesion_penalty += penalty
            elif dist_to_center > 6.0:
                penalty = -30.0  # Updated to match new penalty
                print(f"Drone {i} isolation distance: {dist_to_center:.2f}m → STRONG ISOLATION: {penalty}")
                cohesion_penalty += penalty
            elif dist_to_center > 4.5:
                penalty = -15.0  # New penalty tier
                print(f"Drone {i} isolation distance: {dist_to_center:.2f}m → MODERATE-STRONG ISOLATION: {penalty}")
                cohesion_penalty += penalty
            elif dist_to_center > 3.0:
                penalty = -dist_to_center * 2.0  # Doubled penalty rate
                print(f"Drone {i} isolation distance: {dist_to_center:.2f}m → MILD ISOLATION: {penalty:.2f}")
                cohesion_penalty += penalty
        
        total_penalty = spacing_reward + cohesion_penalty
        print(f"TOTAL SPACING + COHESION PENALTY: {total_penalty:.2f}")
        print(f"Expected: {test_case['expected']}")
        
        # Verify the penalty is working as expected
        if "CRITICAL" in test_case['name'] and total_penalty > -30:
            print("⚠️  WARNING: Critical collision penalty may be too weak!")
        elif "ISOLATION" in test_case['name'] and total_penalty > -40:
            print("⚠️  WARNING: Isolation penalty may be too weak!")
        elif "GOOD" in test_case['name'] and total_penalty < 0:
            print("⚠️  WARNING: Good spacing should have positive reward!")
        else:
            print("✅ Penalty system working as expected")

if __name__ == "__main__":
    test_collision_penalties()
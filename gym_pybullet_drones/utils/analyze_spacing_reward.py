import numpy as np
import matplotlib.pyplot as plt

class SpacingAnalyzer:
    def __init__(self):
        # Parameters from CattleAviary
        self.SPACING_A = 1.2  # pos amp coef
        self.SPACING_B = 2.1  # neg amp coef
        self.SPACING_C = 3.3  # width of pos coef
        self.SPACING_K = 0.2  # width of neg coef
        self.SPACING_D = -1   # pos peak offset
        self.SPACING_R0 = 1.3 # piecewise threshold
        self.SPACING_LAM = 0.8 # exp decay
    
    def SpacingRewardValue(self, r):
        """
        Compute Spacing Reward - copied from CattleAviary
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
    
    def find_optimal_spacing(self, search_range=(0, 5), num_points=1000):
        """Find the distance that gives maximum reward"""
        distances = np.linspace(search_range[0], search_range[1], num_points)
        rewards = [self.SpacingRewardValue(d) for d in distances]
        
        max_idx = np.argmax(rewards)
        optimal_distance = distances[max_idx]
        max_reward = rewards[max_idx]
        
        return optimal_distance, max_reward
    
    def plot_reward_function(self, distance_range=(0, 5)):
        """Plot the spacing reward function"""
        distances = np.linspace(distance_range[0], distance_range[1], 1000)
        rewards = [self.SpacingRewardValue(d) for d in distances]
        
        # Find optimal point
        optimal_dist, max_reward = self.find_optimal_spacing()
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        plt.plot(distances, rewards, 'b-', linewidth=2, label='Spacing Reward Function')
        
        # Mark optimal point
        plt.plot(optimal_dist, max_reward, 'ro', markersize=10, 
                label=f'Optimal Spacing: {optimal_dist:.3f} units\nMax Reward: {max_reward:.3f}')
        
        # Mark the piecewise threshold
        threshold_reward = self.SpacingRewardValue(self.SPACING_R0)
        plt.axvline(x=self.SPACING_R0, color='orange', linestyle='--', alpha=0.7,
                   label=f'Piecewise Threshold: {self.SPACING_R0} units')
        
        # Mark zero reward line
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        # Annotations for different regions
        plt.annotate('Close Range\n(Gaussian Combination)', 
                    xy=(0.5, -1), fontsize=10, ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.annotate('Far Range\n(Exponential Decay)', 
                    xy=(3.5, 0.2), fontsize=10, ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        plt.xlabel('Distance Between Drones (units)', fontsize=12)
        plt.ylabel('Reward Value', fontsize=12)
        plt.title('Drone Spacing Reward Function\nOptimal Formation Distance Analysis', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.xlim(distance_range)
        
        # Add parameter info as text box
        param_text = f"""Parameters:
A (pos amp): {self.SPACING_A}
B (neg amp): {self.SPACING_B} 
C (pos width): {self.SPACING_C}
K (neg width): {self.SPACING_K}
D (peak offset): {self.SPACING_D}
λ (decay): {self.SPACING_LAM}"""
        
        plt.text(0.02, 0.98, param_text, transform=plt.gca().transAxes, 
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('spacing_reward_function.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return optimal_dist, max_reward

# Run the analysis
if __name__ == "__main__":
    analyzer = SpacingAnalyzer()
    
    print("Analyzing Drone Spacing Reward Function...")
    print("=" * 50)
    
    # Find optimal spacing
    optimal_dist, max_reward = analyzer.find_optimal_spacing()
    
    print(f"Optimal Spacing Distance: {optimal_dist:.3f} units")
    print(f"Maximum Reward Value: {max_reward:.3f}")
    print()
    
    # Test specific distances
    test_distances = [0.1, 0.5, 1.0, 1.3, 2.0, 3.0, 5.0]
    print("Reward values at key distances:")
    for dist in test_distances:
        reward = analyzer.SpacingRewardValue(dist)
        print(f"  Distance {dist:4.1f}: Reward = {reward:8.4f}")
    
    print()
    print("Creating visualization...")
    
    # Plot the function
    analyzer.plot_reward_function()
    
    print("Analysis complete! Check 'spacing_reward_function.png' for the plot.")
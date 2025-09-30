#!/usr/bin/env python3
"""
Advanced Curriculum Learning Framework for Cattle Herding RL
============================================================

This module implements a sophisticated curriculum learning system that progressively
teaches drones collision avoidance before introducing full herding task complexity.

Phases:
1. Collision Avoidance Mastery (0-25k steps)
2. Formation + Basic Herding (25k-75k steps)  
3. Full Task Complexity (75k+ steps)

Author: AI Assistant
Date: September 29, 2025
"""

import numpy as np
import time
from collections import deque
from typing import Dict, List, Tuple, Optional

from gym_pybullet_drones.envs.CattleAviary import CattleAviary
from stable_baselines3.common.callbacks import BaseCallback


class AdvancedCurriculumCattleAviary(CattleAviary):
    """
    Enhanced Cattle Herding environment with advanced curriculum learning.
    
    Features:
    - Phase-based learning progression
    - Collision-aware reward scaling
    - Adaptive curriculum advancement
    - Performance monitoring and metrics
    """
    
    def __init__(self, eval_episode_length=None, **kwargs):
        """
        Initialize the curriculum learning environment.
        
        Parameters:
        -----------
        eval_episode_length : float, optional
            Custom episode length for evaluation mode
        **kwargs : dict
            Additional arguments passed to CattleAviary
        """
        super().__init__(**kwargs)
        
        # Training mode and evaluation settings
        self.training_mode = True
        self.eval_episode_length = eval_episode_length
        
        # Curriculum learning state
        self.curriculum_phase = 0
        self.total_training_steps = 0
        self.phase_start_step = 0
        
        # Performance tracking
        self.collision_history = deque(maxlen=100)  # Last 100 episodes
        self.episode_collision_count = 0
        self.episode_step_count = 0
        self.phase_performance_history = []
        
        # Curriculum phase definitions
        self.curriculum_phases = {
            0: {  # Phase 1: Collision Avoidance Mastery
                'name': 'Collision Avoidance',
                'min_steps': 25000,
                'collision_penalty_multiplier': 3.0,
                'task_reward_weight': 0.1,
                'formation_reward_weight': 0.3,
                'collision_avoidance_weight': 0.6,
                'max_collisions_per_episode': 2.0,
                'cattle_enabled': True,  # Keep cattle but focus on collision avoidance
                'episode_length_sec': 15,  # Shorter episodes
                'success_criteria': 'avg_collisions < 2.0 for 50 episodes'
            },
            1: {  # Phase 2: Formation + Basic Herding
                'name': 'Formation + Basic Herding',
                'min_steps': 50000,
                'collision_penalty_multiplier': 2.0,
                'task_reward_weight': 0.5,
                'formation_reward_weight': 0.3,
                'collision_avoidance_weight': 0.2,
                'max_collisions_per_episode': 1.0,
                'cattle_enabled': True,
                'episode_length_sec': 20,
                'success_criteria': 'avg_collisions < 1.0 and basic_herding_success > 0.7'
            },
            2: {  # Phase 3: Full Task Complexity
                'name': 'Full Task Complexity',
                'min_steps': float('inf'),
                'collision_penalty_multiplier': 1.5,  # Still emphasize collision avoidance
                'task_reward_weight': 1.0,
                'formation_reward_weight': 0.3,
                'collision_avoidance_weight': 0.1,
                'max_collisions_per_episode': 0.5,
                'cattle_enabled': True,
                'episode_length_sec': 30,
                'success_criteria': 'full_task_completion with minimal_collisions'
            }
        }
        
        # Episode length schedule (backup for transitions)
        self.episode_length_schedule = {
            0: 15 * self.CTRL_FREQ,      # Phase 1: 15 seconds
            25000: 20 * self.CTRL_FREQ,  # Phase 2: 20 seconds
            75000: 30 * self.CTRL_FREQ,  # Phase 3: 30 seconds
        }
        
        print(f"🎓 [CURRICULUM] Initialized Advanced Curriculum Learning")
        print(f"    Starting Phase: {self.curriculum_phases[0]['name']}")
        print(f"    Collision Focus: {self.curriculum_phases[0]['collision_avoidance_weight']:.1%}")
        
    def reset(self, seed=None, options=None):
        """Reset environment with curriculum-aware configuration."""
        
        # Track episode performance before reset
        if hasattr(self, 'episode_collision_count'):
            self.collision_history.append(self.episode_collision_count)
            self._evaluate_curriculum_progression()
        
        # Reset episode counters
        self.episode_collision_count = 0
        self.episode_step_count = 0
        
        # Apply curriculum-specific episode length
        if hasattr(self, 'training_mode') and not self.training_mode and self.eval_episode_length:
            self.EPISODE_LEN_SEC = self.eval_episode_length
        else:
            current_phase = self.curriculum_phases[self.curriculum_phase]
            self.EPISODE_LEN_SEC = current_phase['episode_length_sec']
        
        return super().reset(seed, options)
    
    def step(self, action):
        """Step function with curriculum-aware collision tracking."""
        self.episode_step_count += 1
        self.total_training_steps += 1
        
        # Store collision state before step
        old_collision_detected = getattr(self, '_collision_detected', False)
        
        # Execute step
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Track collisions
        new_collision_detected = getattr(self, '_collision_detected', False)
        if new_collision_detected and not old_collision_detected:
            self.episode_collision_count += 1
        
        # Apply curriculum-aware reward modification
        modified_reward = self._apply_curriculum_reward_scaling(reward)
        
        # Add curriculum info
        info['curriculum_phase'] = self.curriculum_phase
        info['phase_name'] = self.curriculum_phases[self.curriculum_phase]['name']
        info['episode_collisions'] = self.episode_collision_count
        info['total_training_steps'] = self.total_training_steps
        
        return obs, modified_reward, terminated, truncated, info
    
    def _apply_curriculum_reward_scaling(self, base_reward: float) -> float:
        """
        Apply curriculum-specific reward scaling.
        
        Parameters:
        -----------
        base_reward : float
            Original reward from parent class
            
        Returns:
        --------
        float
            Modified reward based on curriculum phase
        """
        current_phase = self.curriculum_phases[self.curriculum_phase]
        
        # Phase 1: Focus heavily on collision avoidance
        if self.curriculum_phase == 0:
            # Initialize default values
            collision_penalty = 0.0
            formation_component = 0.0
            
            # Extract collision penalty component (negative spacing reward)
            if hasattr(self, 'reward_components'):
                spacing_reward = self.reward_components.get('spacing', [0])[-1] if self.reward_components.get('spacing') else 0
                
                # Amplify collision penalties
                if spacing_reward < 0:  # Collision penalty
                    collision_penalty = spacing_reward * current_phase['collision_penalty_multiplier']
                    base_reward += collision_penalty - spacing_reward  # Replace with amplified penalty
                else:
                    # No collision penalty, just use base reward components
                    collision_penalty = 0.0
                
                # Extract formation rewards (positive spacing reward)
                formation_component = max(0, spacing_reward)  # Only positive spacing rewards
            
            # Combine rewards with curriculum weights
            modified_reward = (formation_component * current_phase['formation_reward_weight'] + 
                             collision_penalty * current_phase['collision_avoidance_weight'])
            
            # Small positive baseline for exploration
            modified_reward += 0.05
            return modified_reward
        
        # Phase 2: Balance collision avoidance with basic herding
        elif self.curriculum_phase == 1:
            # Apply moderate collision penalty amplification
            collision_multiplier = current_phase['collision_penalty_multiplier']
            if getattr(self, '_collision_detected', False):
                base_reward *= (1.0 - (collision_multiplier - 1.0) * 0.5)
            
            # Scale task rewards
            return base_reward * current_phase['task_reward_weight']
        
        # Phase 3: Full complexity with collision awareness
        else:
            # Apply mild collision penalty amplification
            collision_multiplier = current_phase['collision_penalty_multiplier']
            if getattr(self, '_collision_detected', False):
                base_reward *= (1.0 - (collision_multiplier - 1.0) * 0.3)
            
            return base_reward
    
    def _evaluate_curriculum_progression(self):
        """
        Evaluate whether to advance to the next curriculum phase.
        """
        if len(self.collision_history) < 10:  # Need minimum history
            return
            
        current_phase = self.curriculum_phases[self.curriculum_phase]
        steps_in_phase = self.total_training_steps - self.phase_start_step
        
        # Check minimum steps requirement
        if steps_in_phase < current_phase['min_steps']:
            return
        
        # Evaluate phase-specific success criteria
        should_advance = False
        
        if self.curriculum_phase == 0:  # Phase 1: Collision Avoidance
            recent_avg_collisions = np.mean(list(self.collision_history)[-50:]) if len(self.collision_history) >= 50 else float('inf')
            should_advance = recent_avg_collisions < current_phase['max_collisions_per_episode']
            
            if should_advance:
                print(f"🎓 [CURRICULUM] Phase 1 SUCCESS! Avg collisions: {recent_avg_collisions:.2f}")
        
        elif self.curriculum_phase == 1:  # Phase 2: Formation + Basic Herding
            recent_avg_collisions = np.mean(list(self.collision_history)[-50:]) if len(self.collision_history) >= 50 else float('inf')
            should_advance = recent_avg_collisions < current_phase['max_collisions_per_episode']
            
            if should_advance:
                print(f"🎓 [CURRICULUM] Phase 2 SUCCESS! Avg collisions: {recent_avg_collisions:.2f}")
        
        # Advance to next phase if criteria met
        if should_advance and self.curriculum_phase < len(self.curriculum_phases) - 1:
            self._advance_curriculum_phase()
    
    def _advance_curriculum_phase(self):
        """Advance to the next curriculum phase."""
        old_phase = self.curriculum_phase
        self.curriculum_phase += 1
        self.phase_start_step = self.total_training_steps
        
        new_phase = self.curriculum_phases[self.curriculum_phase]
        
        print(f"🎓 [CURRICULUM] PHASE ADVANCEMENT: {old_phase} → {self.curriculum_phase}")
        print(f"    New Phase: {new_phase['name']}")
        print(f"    Task Weight: {new_phase['task_reward_weight']:.1%}")
        print(f"    Collision Focus: {new_phase['collision_avoidance_weight']:.1%}")
        print(f"    Episode Length: {new_phase['episode_length_sec']}s")
        print(f"    Training Steps: {self.total_training_steps:,}")
        
        # Log phase transition
        self.phase_performance_history.append({
            'phase': old_phase,
            'end_step': self.total_training_steps,
            'avg_collisions': np.mean(list(self.collision_history)[-50:]) if len(self.collision_history) >= 50 else 0,
            'episodes_completed': len(self.collision_history)
        })
    
    def get_curriculum_status(self) -> Dict:
        """
        Get current curriculum learning status and metrics.
        
        Returns:
        --------
        Dict
            Dictionary containing curriculum status and performance metrics
        """
        current_phase = self.curriculum_phases[self.curriculum_phase]
        steps_in_phase = self.total_training_steps - self.phase_start_step
        
        return {
            'curriculum_phase': self.curriculum_phase,
            'phase_name': current_phase['name'],
            'total_training_steps': self.total_training_steps,
            'steps_in_current_phase': steps_in_phase,
            'min_steps_required': current_phase['min_steps'],
            'phase_progress': min(1.0, steps_in_phase / current_phase['min_steps']),
            'recent_avg_collisions': np.mean(list(self.collision_history)[-10:]) if len(self.collision_history) >= 10 else 0,
            'collision_threshold': current_phase['max_collisions_per_episode'],
            'collision_criteria_met': np.mean(list(self.collision_history)[-50:]) < current_phase['max_collisions_per_episode'] if len(self.collision_history) >= 50 else False,
            'phase_history': self.phase_performance_history
        }
    
    def print_curriculum_status(self):
        """Print formatted curriculum status."""
        status = self.get_curriculum_status()
        
        print(f"\n🎓 CURRICULUM LEARNING STATUS")
        print(f"=" * 50)
        print(f"Current Phase: {status['phase_name']} (Phase {status['curriculum_phase']})")
        print(f"Training Steps: {status['total_training_steps']:,}")
        print(f"Phase Progress: {status['phase_progress']:.1%}")
        print(f"Recent Avg Collisions: {status['recent_avg_collisions']:.2f}")
        print(f"Collision Threshold: {status['collision_threshold']:.1f}")
        print(f"Criteria Met: {'✅' if status['collision_criteria_met'] else '❌'}")
        
        if status['phase_history']:
            print(f"\nPhase History:")
            for phase_record in status['phase_history']:
                print(f"  Phase {phase_record['phase']}: {phase_record['avg_collisions']:.2f} collisions @ {phase_record['end_step']:,} steps")


class CurriculumCallback(BaseCallback):
    """
    Callback for monitoring curriculum learning progress during training.
    """
    
    def __init__(self, log_frequency: int = 1000, verbose: int = 0):
        """
        Initialize curriculum monitoring callback.
        
        Parameters:
        -----------
        log_frequency : int
            How often to log curriculum status (in training steps)
        verbose : int
            Verbosity level
        """
        super().__init__(verbose)
        self.log_frequency = log_frequency
        self.last_log_step = 0
        
    def _on_step(self) -> bool:
        """
        Called at each training step.
        
        Returns:
        --------
        bool
            True to continue training, False to stop
        """
        # Check if we should log curriculum status
        if self.num_timesteps - self.last_log_step >= self.log_frequency:
            try:
                # Get first environment instance from the vectorized environment
                if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
                    env = self.training_env.envs[0]
                    
                    if isinstance(env, AdvancedCurriculumCattleAviary):
                        env.print_curriculum_status()
                        self.last_log_step = self.num_timesteps
                        
            except Exception as e:
                if self.verbose > 0:
                    print(f"[CURRICULUM] Error logging status: {e}")
        
        return True
# exp3_cell_onoff.py
"""
EXP3 Algorithm for Cell On/Off Energy Optimization
Author: Assistant
Description: Implements EXP3 multi-armed bandit algorithm for dynamic cell on/off control
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque
import json
from pathlib import Path
from datetime import datetime


class EXP3Algorithm:
    """
    EXP3 (Exponential-weight algorithm for Exploration and Exploitation)
    for cell on/off optimization
    """
    
    def __init__(self, n_cells: int, n_off_cells: int, 
                 learning_rate: float = 0.1,
                 min_selection_per_arm: bool = True,
                 warm_up_episodes: int = 100,
                 enable_warm_up: bool = True):
        """
        Initialize EXP3 algorithm
        
        Parameters:
        -----------
        n_cells : int
            Total number of cells
        n_off_cells : int
            Number of cells to turn off
        learning_rate : float
            Learning rate (gamma) for EXP3
        min_selection_per_arm : bool
            Ensure each arm is selected at least once
        warm_up_episodes : int
            Number of warm-up episodes for exploration
        enable_warm_up : bool
            Enable ε-greedy warm-up phase
        """
        self.n_cells = n_cells
        self.n_off_cells = n_off_cells
        self.learning_rate = learning_rate
        self.min_selection_per_arm = min_selection_per_arm
        self.warm_up_episodes = warm_up_episodes
        self.enable_warm_up = enable_warm_up
        
        # Generate all possible combinations of cells to turn off
        from itertools import combinations
        self.arms = list(combinations(range(n_cells), n_off_cells))
        self.n_arms = len(self.arms)
        
        # Initialize weights and probabilities
        self.weights = np.ones(self.n_arms)
        self.probabilities = np.ones(self.n_arms) / self.n_arms
        
        # Tracking variables
        self.episode_count = 0
        self.arm_selection_count = np.zeros(self.n_arms)
        self.cumulative_rewards = []
        self.cumulative_regret = []
        self.arm_selection_history = []
        self.reward_history = []
        self.weight_history = []
        self.probability_history = []
        
        # Baseline performance
        self.baseline_reward = None
        self.optimal_reward = None
        
        # Stabilization tracking
        self.weight_stability_threshold = 0.01
        self.stabilization_episode = None
        
    def select_arm(self) -> Tuple[int, List[int]]:
        """
        Select an arm (combination of cells to turn off) based on current probabilities
        
        Returns:
        --------
        arm_index : int
            Index of selected arm
        cells_to_off : List[int]
            List of cell indices to turn off
        """
        # Warm-up phase: random exploration
        if self.enable_warm_up and self.episode_count < self.warm_up_episodes:
            arm_index = np.random.choice(self.n_arms)
        else:
            # Ensure minimum selection if enabled
            if self.min_selection_per_arm:
                unselected = np.where(self.arm_selection_count == 0)[0]
                if len(unselected) > 0:
                    arm_index = np.random.choice(unselected)
                else:
                    arm_index = np.random.choice(self.n_arms, p=self.probabilities)
            else:
                arm_index = np.random.choice(self.n_arms, p=self.probabilities)
        
        self.arm_selection_count[arm_index] += 1
        self.arm_selection_history.append(arm_index)
        
        return arm_index, list(self.arms[arm_index])
    
    def update(self, arm_index: int, reward: float):
        """
        Update weights and probabilities based on received reward
        
        Parameters:
        -----------
        arm_index : int
            Index of the arm that was selected
        reward : float
            Normalized reward (0-1)
        """
        self.episode_count += 1
        self.reward_history.append(reward)
        
        # Calculate cumulative reward
        if len(self.cumulative_rewards) == 0:
            self.cumulative_rewards.append(reward)
        else:
            self.cumulative_rewards.append(self.cumulative_rewards[-1] + reward)
        
        # Skip weight update during warm-up
        if self.enable_warm_up and self.episode_count <= self.warm_up_episodes:
            return
        
        # Calculate estimated reward
        estimated_reward = reward / self.probabilities[arm_index]
        
        # Update weight for selected arm
        self.weights[arm_index] *= np.exp(self.learning_rate * estimated_reward / self.n_arms)
        
        # Update probabilities
        total_weight = np.sum(self.weights)
        exploitation_probs = self.weights / total_weight
        exploration_prob = self.learning_rate / self.n_arms
        
        self.probabilities = (1 - self.learning_rate) * exploitation_probs + \
                           exploration_prob * np.ones(self.n_arms)
        
        # Store history
        self.weight_history.append(self.weights.copy())
        self.probability_history.append(self.probabilities.copy())
        
        # Check for stabilization
        if len(self.probability_history) > 10:
            recent_probs = np.array(self.probability_history[-10:])
            prob_variance = np.var(recent_probs, axis=0).max()
            if prob_variance < self.weight_stability_threshold and self.stabilization_episode is None:
                self.stabilization_episode = self.episode_count
    
    def calculate_regret(self, optimal_reward: float):
        """
        Calculate cumulative regret
        
        Parameters:
        -----------
        optimal_reward : float
            Reward of the optimal arm
        """
        if len(self.reward_history) == 0:
            return
        
        instant_regret = optimal_reward - self.reward_history[-1]
        if len(self.cumulative_regret) == 0:
            self.cumulative_regret.append(instant_regret)
        else:
            self.cumulative_regret.append(self.cumulative_regret[-1] + instant_regret)


class EXP3CellOnOff:
    """
    Main scenario class for EXP3-based cell on/off control
    Inherits from Scenario class in KISS simulator
    """
    
    def __init__(self, sim, config: Dict):
        """
        Initialize EXP3 Cell On/Off scenario
        
        Parameters:
        -----------
        sim : Sim
            KISS/AIMM simulator instance
        config : Dict
            Configuration dictionary
        """
        self.sim = sim
        self.config = config
        
        # Extract configuration parameters
        self.n_cells_to_off = config.get('n_cells_to_off', 3)
        self.interval = config.get('interval', 1.0)
        self.delay = config.get('delay', 0.0)
        self.learning_rate = config.get('exp3_learning_rate', 0.1)
        self.enable_warm_up = config.get('enable_warm_up', True)
        self.warm_up_episodes = config.get('warm_up_episodes', 100)
        self.min_selection_per_arm = config.get('min_selection_per_arm', True)
        self.reward_normalization = config.get('reward_normalization', 'minmax')
        
        # Get total number of cells
        self.n_cells = len(sim.cells)
        
        # Initialize EXP3 algorithm
        self.exp3 = EXP3Algorithm(
            n_cells=self.n_cells,
            n_off_cells=self.n_cells_to_off,
            learning_rate=self.learning_rate,
            min_selection_per_arm=self.min_selection_per_arm,
            warm_up_episodes=self.warm_up_episodes,
            enable_warm_up=self.enable_warm_up
        )
        
        # State tracking
        self.current_off_cells = []
        self.previous_off_cells = []
        self.switching_cost_history = []
        self.energy_savings_history = []
        self.throughput_history = []
        self.efficiency_history = []
        
        # Baseline measurements
        self.baseline_power = None
        self.baseline_throughput = None
        self.baseline_efficiency = None
        
        # Reward normalization parameters
        self.reward_min = float('inf')
        self.reward_max = float('-inf')
        self.reward_buffer = deque(maxlen=100)
        
    def calculate_network_metrics(self) -> Dict[str, float]:
        """
        Calculate network metrics with proper energy modeling
        """
        total_throughput = 0
        total_power = 0
        active_cells = 0
        
        for i, cell in enumerate(self.sim.cells):
            if i not in self.current_off_cells:
                # Cell is ON
                active_cells += 1
                
                # Get number of attached UEs (load indicator)
                n_attached_ues = len(cell.attached) if hasattr(cell, 'attached') else 0
                
                # Calculate cell throughput
                cell_throughput = 0
                if hasattr(cell, 'reports') and 'throughput_Mbps' in cell.reports:
                    for ue_id in cell.reports['throughput_Mbps']:
                        tp_report = cell.reports['throughput_Mbps'][ue_id]
                        if tp_report and len(tp_report) > 1:
                            cell_throughput += np.sum(tp_report[1])
                
                total_throughput += cell_throughput
                
                # Dynamic power model based on load
                # P_cell = P_static + P_dynamic * load_factor
                P_static = 130  # Watts (idle power)
                P_max_dynamic = 200  # Watts (max dynamic power at full load)
                
                # Load factor based on attached UEs and throughput
                max_ues_per_cell = 20  # Assumed max UEs per cell
                load_factor = min(1.0, n_attached_ues / max_ues_per_cell)
                
                # Alternative: load factor based on throughput
                if cell_throughput > 0:
                    max_throughput_per_cell = 100  # Mbps
                    throughput_factor = min(1.0, cell_throughput / max_throughput_per_cell)
                    load_factor = max(load_factor, throughput_factor)
                
                # Calculate actual power
                P_dynamic = P_max_dynamic * load_factor
                cell_power_watts = P_static + P_dynamic
                
                # Amplifier efficiency (typical 25-30%)
                amplifier_efficiency = 0.28
                total_cell_power = cell_power_watts / amplifier_efficiency
                
                total_power += total_cell_power / 1000  # Convert to kW
                
            else:
                # Cell is OFF/Sleep mode
                sleep_power_watts = 10  # Deep sleep mode
                total_power += sleep_power_watts / 1000  # kW
        
        # Calculate efficiency
        efficiency = total_throughput / total_power if total_power > 0 else 0
        
        return {
            'throughput': total_throughput,
            'power': total_power,
            'efficiency': efficiency,
            'active_cells': active_cells,
            'load_factor_avg': total_throughput / (active_cells * 50) if active_cells > 0 else 0
        }
    
    def calculate_reward(self, metrics: Dict[str, float]) -> float:
        """
        Calculate reward based on network efficiency
        
        Parameters:
        -----------
        metrics : Dict
            Network performance metrics
            
        Returns:
        --------
        reward : float
            Normalized reward value
        """
        # Primary reward: network efficiency (throughput/power)
        raw_reward = metrics['efficiency']
        
        # Update reward bounds for normalization
        self.reward_buffer.append(raw_reward)
        self.reward_min = min(self.reward_min, raw_reward)
        self.reward_max = max(self.reward_max, raw_reward)
        
        # Normalize reward
        if self.reward_normalization == 'minmax':
            if self.reward_max > self.reward_min:
                normalized_reward = (raw_reward - self.reward_min) / (self.reward_max - self.reward_min)
            else:
                normalized_reward = 0.5
        elif self.reward_normalization == 'zscore':
            if len(self.reward_buffer) > 1:
                mean_reward = np.mean(self.reward_buffer)
                std_reward = np.std(self.reward_buffer)
                if std_reward > 0:
                    normalized_reward = (raw_reward - mean_reward) / std_reward
                    normalized_reward = (normalized_reward + 3) / 6  # Map to [0, 1]
                    normalized_reward = np.clip(normalized_reward, 0, 1)
                else:
                    normalized_reward = 0.5
            else:
                normalized_reward = 0.5
        else:
            normalized_reward = raw_reward
        
        return normalized_reward
    
    def calculate_switching_cost(self) -> int:
        """
        Calculate the switching cost (number of cells that changed state)
        
        Returns:
        --------
        cost : int
            Number of cells that switched state
        """
        if len(self.previous_off_cells) == 0:
            return 0
        
        prev_set = set(self.previous_off_cells)
        curr_set = set(self.current_off_cells)
        
        # Cells that were turned on or off
        switching_cost = len(prev_set.symmetric_difference(curr_set))
        
        return switching_cost
    
    def apply_cell_configuration(self, cells_to_off: List[int]):
        """
        Apply the cell on/off configuration
        
        Parameters:
        -----------
        cells_to_off : List[int]
            List of cell indices to turn off
        """
        # Store previous configuration
        self.previous_off_cells = self.current_off_cells.copy()
        self.current_off_cells = cells_to_off
        
        # Apply configuration
        for i, cell in enumerate(self.sim.cells):
            if i in cells_to_off:
                # Turn cell OFF - set power to 0 (not -inf to avoid log issues)
                cell.set_power_dBm(-100)  # Very low power, effectively off
                print(f"t={self.sim.env.now:.1f}: Cell[{i}] turned OFF")
            elif i in self.previous_off_cells and i not in cells_to_off:
                # Turn cell back ON
                original_power = self.config.get('power_dBm', 43.0)
                cell.set_power_dBm(original_power)
                print(f"t={self.sim.env.now:.1f}: Cell[{i}] turned ON")
        
        # Trigger handovers for affected UEs
        if hasattr(self.sim, 'mme') and self.sim.mme is not None:
            self.sim.mme.do_handovers()
    
    def loop(self):
        """
        Main loop for the EXP3 cell on/off scenario
        """
        # Initial delay
        if self.delay > 0:
            yield self.sim.env.timeout(self.delay)
        
        # Measure baseline (all cells ON)
        print(f"t={self.sim.env.now:.1f}: Measuring baseline performance...")
        baseline_metrics = self.calculate_network_metrics()
        self.baseline_power = baseline_metrics['power']
        self.baseline_throughput = baseline_metrics['throughput']
        self.baseline_efficiency = baseline_metrics['efficiency']
        self.exp3.baseline_reward = self.baseline_efficiency
        
        print(f"Baseline - Throughput: {self.baseline_throughput:.2f} Mbps, "
              f"Power: {self.baseline_power:.2f} kW, "
              f"Efficiency: {self.baseline_efficiency:.2f} Mbps/kW")
        
        # Main control loop
        while True:
            # Select action using EXP3
            arm_index, cells_to_off = self.exp3.select_arm()
            
            # Apply cell configuration
            self.apply_cell_configuration(cells_to_off)
            
            # Wait for network to stabilize
            yield self.sim.env.timeout(self.interval)
            
            # Calculate metrics and reward
            metrics = self.calculate_network_metrics()
            reward = self.calculate_reward(metrics)
            
            # Update EXP3 algorithm
            self.exp3.update(arm_index, reward)
            
            # Calculate additional metrics
            switching_cost = self.calculate_switching_cost()
            energy_savings = (self.baseline_power - metrics['power']) / self.baseline_power * 100
            
            # Store history
            self.switching_cost_history.append(switching_cost)
            self.energy_savings_history.append(energy_savings)
            self.throughput_history.append(metrics['throughput'])
            self.efficiency_history.append(metrics['efficiency'])
            
            # Calculate regret if we know optimal configuration
            if self.exp3.optimal_reward is not None:
                self.exp3.calculate_regret(self.exp3.optimal_reward)
            
            # Log current state
            if self.exp3.episode_count % 10 == 0:
                print(f"t={self.sim.env.now:.1f} Episode {self.exp3.episode_count}: "
                      f"Cells OFF: {cells_to_off}, "
                      f"Reward: {reward:.3f}, "
                      f"Efficiency: {metrics['efficiency']:.2f} Mbps/kW, "
                      f"Energy Savings: {energy_savings:.1f}%")
            
            yield self.sim.env.timeout(self.interval)
    
    def save_results(self, output_dir: str):
        """
        Save experiment results to files
        
        Parameters:
        -----------
        output_dir : str
            Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare results dictionary
        results = {
            'config': self.config,
            'baseline': {
                'power': self.baseline_power,
                'throughput': self.baseline_throughput,
                'efficiency': self.baseline_efficiency
            },
            'exp3': {
                'n_arms': self.exp3.n_arms,
                'episodes': self.exp3.episode_count,
                'stabilization_episode': self.exp3.stabilization_episode,
                'arm_selection_count': self.exp3.arm_selection_count.tolist(),
                'cumulative_rewards': self.exp3.cumulative_rewards,
                'cumulative_regret': self.exp3.cumulative_regret,
                'final_probabilities': self.exp3.probabilities.tolist()
            },
            'metrics': {
                'switching_cost': self.switching_cost_history,
                'energy_savings': self.energy_savings_history,
                'throughput': self.throughput_history,
                'efficiency': self.efficiency_history
            }
        }
        
        # Save as JSON
        with open(output_path / 'exp3_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save detailed history as TSV
        history_df = pd.DataFrame({
            'episode': range(1, self.exp3.episode_count + 1),
            'arm_selected': self.exp3.arm_selection_history,
            'reward': self.exp3.reward_history,
            'cumulative_reward': self.exp3.cumulative_rewards,
            'throughput': self.throughput_history[:self.exp3.episode_count],
            'efficiency': self.efficiency_history[:self.exp3.episode_count],
            'energy_savings': self.energy_savings_history[:self.exp3.episode_count],
            'switching_cost': self.switching_cost_history[:self.exp3.episode_count]
        })
        
        history_df.to_csv(output_path / 'exp3_history.tsv', sep='\t', index=False)
        
        # Save probability evolution
        if len(self.exp3.probability_history) > 0:
            prob_df = pd.DataFrame(self.exp3.probability_history)
            prob_df.to_csv(output_path / 'exp3_probabilities.tsv', sep='\t', index=False)
        
        print(f"Results saved to {output_path}")


# Integration with KISS simulator
def create_exp3_scenario(sim, config):
    """
    Factory function to create EXP3 scenario for KISS simulator
    
    Parameters:
    -----------
    sim : Sim
        KISS/AIMM simulator instance
    config : Dict
        Configuration dictionary
        
    Returns:
    --------
    scenario : EXP3CellOnOff
        EXP3 scenario instance
    """
    return EXP3CellOnOff(sim, config)
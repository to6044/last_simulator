"""
EXP3 Algorithm for Cell On/Off Control
Uses RIC class for algorithm implementation and Scenario for simulation setup
"""
import numpy as np
from sys import stderr
from AIMM_simulator import RIC, Scenario
from itertools import combinations
import json


class EXP3CellOnOffRIC(RIC):
    """
    RIC implementation for EXP3 algorithm-based cell on/off control.
    Selects n cells to turn off from k total cells to optimize network efficiency.
    """
    
    def __init__(self, sim, interval=10, n_cells_to_off=3, learning_rate=0.1, 
                 exploration_rate=0.1, verbosity=1):
        """
        Initialize EXP3 RIC.
        
        Parameters:
        -----------
        sim : Sim
            Simulator instance
        interval : float
            Time interval between RIC actions
        n_cells_to_off : int
            Number of cells to turn off
        learning_rate : float
            EXP3 learning rate (eta)
        exploration_rate : float
            EXP3 exploration parameter (gamma)
        verbosity : int
            Debug output level
        """
        super().__init__(sim, interval, verbosity)
        self.n_cells_to_off = n_cells_to_off
        self.eta = learning_rate  # Learning rate
        self.gamma = exploration_rate  # Exploration rate
        
        # Generate all possible combinations of cells to turn off
        self.k_cells = len(sim.cells)
        self.arms = list(combinations(range(self.k_cells), n_cells_to_off))
        self.n_arms = len(self.arms)
        
        # Initialize weights uniformly
        self.weights = np.ones(self.n_arms)
        
        # History tracking
        self.history = []
        self.current_arm = None
        self.cells_on_state = [True] * self.k_cells  # Track which cells are on
        
        # Energy model references (will be set from scenario)
        self.cell_energy_models = None
        
        print(f"EXP3 RIC initialized: {self.k_cells} cells, turning off {n_cells_to_off} cells", file=stderr)
        print(f"Total arms (combinations): {self.n_arms}", file=stderr)
    
    def set_energy_models(self, energy_models):
        """Set reference to cell energy models."""
        self.cell_energy_models = energy_models
    
    def get_probabilities(self):
        """Calculate arm selection probabilities using EXP3 formula."""
        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum == 0:
            weight_sum = 1e-10
        
        # EXP3 probability calculation
        probs = (1 - self.gamma) * (self.weights / weight_sum) + self.gamma / self.n_arms
        
        # Ensure probabilities sum to 1
        probs = probs / np.sum(probs)
        return probs
    
    def select_arm(self):
        """Select an arm (combination of cells to turn off) based on probabilities."""
        probs = self.get_probabilities()
        arm_idx = np.random.choice(self.n_arms, p=probs)
        return arm_idx, self.arms[arm_idx]
    
    def turn_cells_off(self, cells_to_off):
        """Turn off specified cells."""
        for cell_idx in cells_to_off:
            if self.cells_on_state[cell_idx]:
                # Set cell power to minimum (effectively off)
                self.sim.cells[cell_idx].set_power_dBm(-np.inf)
                self.cells_on_state[cell_idx] = False
                if self.verbosity > 0:
                    print(f"t={self.sim.env.now:.2f} Cell[{cell_idx}] turned OFF", file=stderr)
    
    def turn_cells_on(self, cells_to_on):
        """Turn on specified cells."""
        for cell_idx in cells_to_on:
            if not self.cells_on_state[cell_idx]:
                # Restore cell power to original value (20 dBm)
                self.sim.cells[cell_idx].set_power_dBm(20.0)
                self.cells_on_state[cell_idx] = True
                if self.verbosity > 0:
                    print(f"t={self.sim.env.now:.2f} Cell[{cell_idx}] turned ON", file=stderr)
    
    def calculate_reward(self):
        """
        Calculate reward as network efficiency (throughput/power).
        Returns normalized reward in [0, 1].
        """
        total_throughput = 0.0
        total_power = 0.0
        
        # Calculate total throughput
        for cell in self.sim.cells:
            if self.cells_on_state[cell.i]:
                cell_throughput = cell.get_average_throughput()
                if cell_throughput is not None and cell_throughput > 0:
                    total_throughput += cell_throughput
        
        # Calculate total power consumption
        if self.cell_energy_models:
            for cell_idx, is_on in enumerate(self.cells_on_state):
                if is_on:
                    energy_model = self.cell_energy_models.get(cell_idx)
                    if energy_model:
                        power_watts = energy_model.get_cell_power_watts(self.sim.env.now)
                        total_power += power_watts
        
        # Calculate efficiency (avoid division by zero)
        if total_power > 0:
            efficiency = total_throughput / (total_power / 1000)  # Convert watts to kW
        else:
            efficiency = 0.0
        
        # Normalize reward to [0, 1] range
        # Using sigmoid-like normalization
        normalized_reward = 1.0 / (1.0 + np.exp(-0.1 * (efficiency - 10)))
        
        return normalized_reward, efficiency, total_throughput, total_power
    
    def update_weights(self, arm_idx, reward):
        """Update weights using EXP3 algorithm."""
        probs = self.get_probabilities()
        
        # Estimated reward for the chosen arm
        estimated_reward = reward / probs[arm_idx]
        
        # Update weight for chosen arm
        self.weights[arm_idx] *= np.exp(self.eta * estimated_reward / self.n_arms)
        
        # Prevent weight explosion
        max_weight = 1e10
        if np.max(self.weights) > max_weight:
            self.weights = self.weights / np.max(self.weights) * max_weight
    
    def trigger_handovers(self):
        """Trigger handovers for UEs affected by cell changes."""
        if self.sim.mme:
            self.sim.mme.do_handovers()
    
    def loop(self):
        """Main loop of EXP3 RIC."""
        print(f'EXP3 RIC started at {float(self.sim.env.now):.2f}', file=stderr)
        
        # Initial wait
        yield self.sim.env.timeout(self.interval)
        
        while True:
            # Select arm (cells to turn off)
            arm_idx, cells_to_off = self.select_arm()
            
            # Determine cells to turn on (previously off but not in current selection)
            cells_to_on = []
            for i in range(self.k_cells):
                if not self.cells_on_state[i] and i not in cells_to_off:
                    cells_to_on.append(i)
            
            # Apply actions
            self.turn_cells_off(cells_to_off)
            self.turn_cells_on(cells_to_on)
            
            # Trigger handovers for affected UEs
            self.trigger_handovers()
            
            # Wait for network to stabilize
            yield self.sim.env.timeout(0.1)
            
            # Calculate reward
            reward, efficiency, throughput, power = self.calculate_reward()
            
            # Update weights
            self.update_weights(arm_idx, reward)
            
            # Log event
            event_data = {
                'time': float(self.sim.env.now),
                'arm_idx': arm_idx,
                'cells_off': list(cells_to_off),
                'reward': reward,
                'efficiency': efficiency,
                'throughput': throughput,
                'power': power,
                'weights': self.weights.tolist()
            }
            self.history.append(event_data)
            
            if self.verbosity > 0:
                print(f"t={self.sim.env.now:.2f} Arm {arm_idx} selected, "
                      f"Cells OFF: {cells_to_off}, Reward: {reward:.4f}, "
                      f"Efficiency: {efficiency:.2f}", file=stderr)
            
            # Wait for next interval
            yield self.sim.env.timeout(self.interval)
    
    def finalize(self):
        """Called at end of simulation."""
        print(f"EXP3 RIC finalized. Total events: {len(self.history)}", file=stderr)
        
        # Find best arm based on final weights
        best_arm_idx = np.argmax(self.weights)
        best_cells_off = self.arms[best_arm_idx]
        print(f"Best arm: {best_arm_idx}, Cells to turn off: {best_cells_off}", file=stderr)


class EXP3CellOnOffScenario(Scenario):
    """
    Scenario setup for EXP3 cell on/off simulation.
    Configures the network topology and initial conditions.
    """
    
    def __init__(self, sim, config_dict, interval=1.0, verbosity=0):
        """
        Initialize scenario from configuration.
        
        Parameters:
        -----------
        sim : Sim
            Simulator instance
        config_dict : dict
            Configuration dictionary
        interval : float
            Time interval between scenario actions
        verbosity : int
            Debug output level
        """
        super().__init__(sim, None, interval, verbosity)
        self.config = config_dict
        
        # Parse EXP3 specific configuration
        self.n_cells_to_off = config_dict.get('exp3_n_cells_to_off', 3)
        self.learning_rate = config_dict.get('exp3_learning_rate', 0.1)
        self.exploration_rate = config_dict.get('exp3_exploration_rate', 0.1)
        
    def loop(self):
        """Main loop of scenario - handles initial setup only."""
        # Initial setup is done in main function
        # This loop can be used for dynamic scenario changes if needed
        while True:
            yield self.sim.env.timeout(self.interval)
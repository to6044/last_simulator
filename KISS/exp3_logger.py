"""
Extended MyLogger for EXP3 Cell On/Off Simulation
Saves detailed history to TSV file
"""
import numpy as np
from AIMM_simulator import Logger
from pathlib import Path
import json


class EXP3Logger(Logger):
    """
    Extended logger that saves EXP3 simulation results to TSV file.
    Tracks cell states, energy sumption, throughput, and rewards.
    """
    
    def __init__(self, sim, logging_interval=1.0, cell_energy_models=None, 
                 logfile_path="exp3_results.tsv", ric=None):
        """
        Initialize EXP3 logger.
        
        Parameters:
        -----------
        sim : Sim
            Simulator instance
        logging_interval : float
            Time interval between logging
        cell_energy_models : dict
            Dictionary of cell energy models
        logfile_path : str
            Path to output TSV file
        ric : EXP3CellOnOffRIC
            Reference to RIC for accessing EXP3 state
        """
        # Open TSV file for writing
        self.logfile_path = Path(logfile_path)
        self.logfile_path.parent.mkdir(parents=True, exist_ok=True)
        f = open(self.logfile_path, 'w')
        
        # Initialize parent Logger
        super().__init__(sim, func=None, header='', f=f, 
                        logging_interval=logging_interval)
        
        self.cell_energy_models = cell_energy_models
        self.ric = ric
        
        # Write TSV header
        self.write_header()
        
        # Statistics tracking
        self.stats = {
            'total_energy': 0.0,
            'total_throughput': 0.0,
            'handover_count': 0,
            'sample_count': 0
        }
    
    def write_header(self):
        """Write TSV file header."""
        headers = [
            'time',
            'n_cells_on',
            'cells_off',
            'total_throughput_mbps',
            'total_power_watts',
            'efficiency',
            'n_ues_attached',
            'avg_cqi',
            'avg_sinr_db',
            'exp3_reward',
            'exp3_arm_idx',
            'exp3_best_arm'
        ]
        
        # Add per-cell columns
        n_cells = len(self.sim.cells)
        for i in range(n_cells):
            headers.extend([
                f'cell_{i}_on',
                f'cell_{i}_power_watts',
                f'cell_{i}_throughput_mbps',
                f'cell_{i}_n_ues'
            ])
        
        self.f.write('\t'.join(headers) + '\n')
        self.f.flush()
    
    def get_cell_metrics(self, cell_idx):
        """Get metrics for a specific cell."""
        cell = self.sim.cells[cell_idx]
        
        # Check if cell is on
        is_on = self.ric.cells_on_state[cell_idx] if self.ric else True
        
        # Get power consumption
        power_watts = 0.0
        if is_on and self.cell_energy_models:
            energy_model = self.cell_energy_models.get(cell_idx)
            if energy_model:
                power_watts = energy_model.get_cell_power_watts(self.sim.env.now)
        
        # Get throughput
        throughput = 0.0
        if is_on:
            tp = cell.get_average_throughput()
            if tp is not None and tp > 0:
                throughput = tp
        
        # Count attached UEs
        n_ues = cell.get_nattached() if is_on else 0
        
        return {
            'on': int(is_on),
            'power': power_watts,
            'throughput': throughput,
            'n_ues': n_ues
        }
    
    def get_network_metrics(self):
        """Calculate network-wide metrics."""
        total_throughput = 0.0
        total_power = 0.0
        total_ues = 0
        total_cqi = 0.0
        total_sinr = 0.0
        cqi_count = 0
        
        for cell_idx, cell in enumerate(self.sim.cells):
            metrics = self.get_cell_metrics(cell_idx)
            total_throughput += metrics['throughput']
            total_power += metrics['power']
            total_ues += metrics['n_ues']
            
            # Get CQI values
            if self.ric.cells_on_state[cell_idx]:
                for ue_idx in cell.reports.get('cqi', {}):
                    cqi_report = cell.reports['cqi'].get(ue_idx)
                    if cqi_report and len(cqi_report) > 1:
                        cqi_val = np.mean(cqi_report[1])
                        if not np.isnan(cqi_val):
                            total_cqi += cqi_val
                            cqi_count += 1
                
                # Get SINR if using extended UE class
                for ue in self.sim.UEs:
                    if hasattr(ue, 'get_sinr_from_cell') and ue.serving_cell == cell:
                        sinr = ue.get_sinr_from_cell(cell)
                        if sinr is not None:
                            if isinstance(sinr, np.ndarray):
                                sinr_val = np.mean(sinr)
                            else:
                                sinr_val = sinr
                            if not np.isnan(sinr_val):
                                total_sinr += sinr_val
        
        avg_cqi = total_cqi / cqi_count if cqi_count > 0 else 0
        avg_sinr = total_sinr / total_ues if total_ues > 0 else 0
        efficiency = (total_throughput / (total_power / 1000)) if total_power > 0 else 0
        
        return {
            'total_throughput': total_throughput,
            'total_power': total_power,
            'efficiency': efficiency,
            'n_ues_attached': total_ues,
            'avg_cqi': avg_cqi,
            'avg_sinr': avg_sinr
        }
    
    def loop(self):
        """Main logging loop."""
        while True:
            # Get current time
            current_time = float(self.sim.env.now)
            
            # Get network metrics
            net_metrics = self.get_network_metrics()
            
            # Get EXP3 state
            cells_off = []
            exp3_reward = 0.0
            exp3_arm_idx = -1
            exp3_best_arm = -1
            
            if self.ric:
                # Find currently off cells
                cells_off = [i for i, is_on in enumerate(self.ric.cells_on_state) if not is_on]
                
                # Get latest EXP3 data
                if self.ric.history:
                    latest = self.ric.history[-1]
                    exp3_reward = latest.get('reward', 0.0)
                    exp3_arm_idx = latest.get('arm_idx', -1)
                
                # Get best arm
                if hasattr(self.ric, 'weights'):
                    exp3_best_arm = np.argmax(self.ric.weights)
            
            # Build row data
            row = [
                f"{current_time:.2f}",
                str(sum(self.ric.cells_on_state) if self.ric else len(self.sim.cells)),
                ','.join(map(str, cells_off)) if cells_off else 'none',
                f"{net_metrics['total_throughput']:.4f}",
                f"{net_metrics['total_power']:.2f}",
                f"{net_metrics['efficiency']:.4f}",
                str(net_metrics['n_ues_attached']),
                f"{net_metrics['avg_cqi']:.2f}",
                f"{net_metrics['avg_sinr']:.2f}",
                f"{exp3_reward:.4f}",
                str(exp3_arm_idx),
                str(exp3_best_arm)
            ]
            
            # Add per-cell data
            for cell_idx in range(len(self.sim.cells)):
                metrics = self.get_cell_metrics(cell_idx)
                row.extend([
                    str(metrics['on']),
                    f"{metrics['power']:.2f}",
                    f"{metrics['throughput']:.4f}",
                    str(metrics['n_ues'])
                ])
            
            # Write row to TSV
            self.f.write('\t'.join(row) + '\n')
            self.f.flush()
            
            # Update statistics
            self.stats['total_energy'] += net_metrics['total_power'] * self.logging_interval
            self.stats['total_throughput'] += net_metrics['total_throughput'] * self.logging_interval
            self.stats['sample_count'] += 1
            
            # Wait for next interval
            yield self.sim.env.timeout(self.logging_interval)
    
    def finalize(self):
        """Called at end of simulation."""
        print(f"EXP3 Logger finalized. Results saved to: {self.logfile_path}")
        
        # Write summary statistics
        if self.stats['sample_count'] > 0:
            avg_power = self.stats['total_energy'] / self.stats['sample_count']
            avg_throughput = self.stats['total_throughput'] / self.stats['sample_count']
            
            summary = {
                'avg_power_watts': avg_power,
                'avg_throughput_mbps': avg_throughput,
                'total_energy_joules': self.stats['total_energy'],
                'simulation_time': float(self.sim.env.now)
            }
            
            # Save summary to JSON
            summary_path = self.logfile_path.with_suffix('.summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Summary statistics saved to: {summary_path}")
        
        # Close TSV file
        self.f.close()
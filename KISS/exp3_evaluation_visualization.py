# exp3_evaluation_visualization.py
"""
Evaluation and Visualization for EXP3 Cell On/Off Results
Generates comprehensive analysis plots and metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats
from typing import List, Dict, Tuple
import argparse


class EXP3Evaluator:
    """
    Comprehensive evaluation and visualization for EXP3 results
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize evaluator
        
        Parameters:
        -----------
        output_dir : str
            Directory containing EXP3 results
        """
        self.output_dir = Path(output_dir)
        self.seed_dirs = sorted([d for d in self.output_dir.iterdir() if d.is_dir() and 'seed_' in d.name])
        self.results = self.load_all_results()
        
        # Set plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def load_all_results(self) -> List[Dict]:
        """
        Load results from all seed directories
        """
        results = []
        
        for seed_dir in self.seed_dirs:
            # Load JSON results
            json_file = seed_dir / 'exp3_results.json'
            if json_file.exists():
                with open(json_file, 'r') as f:
                    seed_results = json.load(f)
                
                # Load TSV history
                history_file = seed_dir / 'exp3_history.tsv'
                if history_file.exists():
                    seed_results['history_df'] = pd.read_csv(history_file, sep='\t')
                
                # Load probability evolution
                prob_file = seed_dir / 'exp3_probabilities.tsv'
                if prob_file.exists():
                    seed_results['prob_df'] = pd.read_csv(prob_file, sep='\t')
                
                # Load metrics
                metrics_file = seed_dir / 'exp3_metrics.tsv'
                if metrics_file.exists():
                    seed_results['metrics_df'] = pd.read_csv(metrics_file, sep='\t')
                
                seed_results['seed'] = int(seed_dir.name.split('_')[1])
                results.append(seed_results)
        
        return results
    
    def calculate_evaluation_metrics(self) -> Dict:
        """
        Calculate all evaluation metrics across seeds
        """
        metrics = {}
        
        # 1. Cumulative Reward Analysis
        cumulative_rewards = [r['exp3']['cumulative_rewards'] for r in self.results]
        rewards_array = np.array([r + [r[-1]]*(max(map(len, cumulative_rewards)) - len(r)) 
                                 for r in cumulative_rewards])
        
        metrics['cumulative_reward_mean'] = np.mean(rewards_array, axis=0)
        metrics['cumulative_reward_std'] = np.std(rewards_array, axis=0)
        
        # 2. Cumulative and Average Regret
        if self.results[0]['exp3']['cumulative_regret']:
            cumulative_regrets = [r['exp3']['cumulative_regret'] for r in self.results]
            regrets_array = np.array([r + [r[-1]]*(max(map(len, cumulative_regrets)) - len(r)) 
                                     for r in cumulative_regrets])
            
            metrics['cumulative_regret_mean'] = np.mean(regrets_array, axis=0)
            metrics['cumulative_regret_std'] = np.std(regrets_array, axis=0)
            metrics['average_regret'] = metrics['cumulative_regret_mean'] / np.arange(1, len(metrics['cumulative_regret_mean']) + 1)
        
        # 3. Selection Probability Distribution
        final_probs = [r['exp3']['final_probabilities'] for r in self.results]
        metrics['selection_prob_mean'] = np.mean(final_probs, axis=0)
        metrics['selection_prob_std'] = np.std(final_probs, axis=0)
        
        # 4. Performance Variance
        efficiency_histories = [r['metrics']['efficiency'] for r in self.results]
        metrics['performance_variance'] = np.var([e[-100:] for e in efficiency_histories if len(e) >= 100])
        metrics['performance_std'] = np.sqrt(metrics['performance_variance'])
        
        # 5. Switching Cost Analysis
        switching_costs = [r['metrics']['switching_cost'] for r in self.results]
        metrics['avg_switching_cost'] = np.mean([np.mean(s) for s in switching_costs])
        metrics['total_switching_cost'] = np.mean([np.sum(s) for s in switching_costs])
        
        # 6. Energy Savings
        energy_savings = [r['metrics']['energy_savings'] for r in self.results]
        metrics['avg_energy_savings'] = np.mean([np.mean(s) for s in energy_savings])
        metrics['final_energy_savings'] = np.mean([s[-1] for s in energy_savings if s])
        
        # 7. Average Cell Throughput
        throughput_histories = [r['metrics']['throughput'] for r in self.results]
        metrics['avg_cell_throughput'] = np.mean([np.mean(t) for t in throughput_histories])
        
        # 8. Stabilization Time
        stabilization_episodes = [r['exp3'].get('stabilization_episode') for r in self.results]
        stabilization_episodes = [e for e in stabilization_episodes if e is not None]
        if stabilization_episodes:
            metrics['avg_stabilization_episode'] = np.mean(stabilization_episodes)
            metrics['std_stabilization_episode'] = np.std(stabilization_episodes)
        
        # 9. Confidence Intervals (95%)
        n_seeds = len(self.results)
        if n_seeds > 1:
            confidence_level = 0.95
            t_critical = stats.t.ppf((1 + confidence_level) / 2, n_seeds - 1)
            
            metrics['ci_reward_lower'] = metrics['cumulative_reward_mean'] - \
                                        t_critical * metrics['cumulative_reward_std'] / np.sqrt(n_seeds)
            metrics['ci_reward_upper'] = metrics['cumulative_reward_mean'] + \
                                        t_critical * metrics['cumulative_reward_std'] / np.sqrt(n_seeds)
        
        return metrics
    
    def plot_cumulative_rewards_regrets(self, metrics: Dict):
        """
        Plot cumulative rewards and regrets with confidence intervals
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        episodes = np.arange(len(metrics['cumulative_reward_mean']))
        
        # Cumulative Rewards
        ax1 = axes[0]
        ax1.plot(episodes, metrics['cumulative_reward_mean'], 'b-', label='Mean Reward', linewidth=2)
        ax1.fill_between(episodes,
                         metrics['cumulative_reward_mean'] - metrics['cumulative_reward_std'],
                         metrics['cumulative_reward_mean'] + metrics['cumulative_reward_std'],
                         alpha=0.3, color='blue', label='±1 Std Dev')
        
        if 'ci_reward_lower' in metrics:
            ax1.fill_between(episodes,
                            metrics['ci_reward_lower'],
                            metrics['ci_reward_upper'],
                            alpha=0.2, color='green', label='95% CI')
        
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Cumulative Reward', fontsize=12)
        ax1.set_title('Cumulative Reward over Episodes', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative Regrets (if available)
        ax2 = axes[1]
        if 'cumulative_regret_mean' in metrics:
            ax2.plot(episodes[:len(metrics['cumulative_regret_mean'])], 
                    metrics['cumulative_regret_mean'], 'r-', label='Cumulative Regret', linewidth=2)
            ax2.plot(episodes[:len(metrics['average_regret'])],
                    metrics['average_regret'], 'g--', label='Average Regret', linewidth=2)
            ax2.fill_between(episodes[:len(metrics['cumulative_regret_mean'])],
                            metrics['cumulative_regret_mean'] - metrics['cumulative_regret_std'],
                            metrics['cumulative_regret_mean'] + metrics['cumulative_regret_std'],
                            alpha=0.3, color='red')
        
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Regret', fontsize=12)
        ax2.set_title('Regret Analysis', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cumulative_rewards_regrets.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'cumulative_rewards_regrets.pdf', bbox_inches='tight')
        plt.close()
    
    def plot_selection_distribution(self, metrics: Dict):
        """
        Plot arm selection probability distribution
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Final probability distribution
        ax1 = axes[0]
        n_arms = len(metrics['selection_prob_mean'])
        arm_indices = np.arange(n_arms)
        
        bars = ax1.bar(arm_indices, metrics['selection_prob_mean'], 
                      yerr=metrics['selection_prob_std'], capsize=5,
                      color='skyblue', edgecolor='navy', linewidth=1.5)
        
        # Highlight top 5 arms
        top_5_indices = np.argsort(metrics['selection_prob_mean'])[-5:]
        for idx in top_5_indices:
            bars[idx].set_color('coral')
        
        ax1.set_xlabel('Arm Index', fontsize=12)
        ax1.set_ylabel('Selection Probability', fontsize=12)
        ax1.set_title('Final Arm Selection Probabilities', fontsize=14, fontweight='bold')
        ax1.set_xlim(-1, n_arms)
        
        # Probability evolution over time (for first seed as example)
        ax2 = axes[1]
        if self.results[0].get('prob_df') is not None:
            prob_df = self.results[0]['prob_df']
            episodes = np.arange(len(prob_df))
            
            # Plot top 5 arms evolution
            top_5_arms = np.argsort(metrics['selection_prob_mean'])[-5:]
            for arm_idx in top_5_arms:
                ax2.plot(episodes, prob_df.iloc[:, arm_idx], 
                        label=f'Arm {arm_idx}', linewidth=2)
        
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Selection Probability', fontsize=12)
        ax2.set_title('Probability Evolution (Top 5 Arms)', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', ncol=2)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'selection_distribution.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'selection_distribution.pdf', bbox_inches='tight')
        plt.close()
    
    def plot_energy_performance_metrics(self, metrics: Dict):
        """
        Plot energy savings and performance metrics
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Energy Savings Over Time
        ax1 = axes[0, 0]
        for result in self.results[:5]:  # Plot first 5 seeds
            if 'history_df' in result:
                df = result['history_df']
                ax1.plot(df['episode'], df['energy_savings'], 
                        alpha=0.5, linewidth=1)
        
        # Add mean line
        all_energy_savings = [r['metrics']['energy_savings'] for r in self.results]
        mean_savings = np.mean(all_energy_savings, axis=0)
        episodes = np.arange(len(mean_savings))
        ax1.plot(episodes, mean_savings, 'r-', linewidth=3, label='Mean')
        
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Energy Savings (%)', fontsize=12)
        ax1.set_title('Energy Savings Evolution', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Average Cell Throughput
        ax2 = axes[0, 1]
        for result in self.results[:5]:
            if 'history_df' in result:
                df = result['history_df']
                ax2.plot(df['episode'], df['throughput'],
                        alpha=0.5, linewidth=1)
        
        all_throughput = [r['metrics']['throughput'] for r in self.results]
        mean_throughput = np.mean(all_throughput, axis=0)
        ax2.plot(np.arange(len(mean_throughput)), mean_throughput, 
                'b-', linewidth=3, label='Mean')
        
        # Add baseline
        baseline_throughput = np.mean([r['baseline']['throughput'] for r in self.results])
        ax2.axhline(y=baseline_throughput, color='g', linestyle='--', 
                   label='Baseline', linewidth=2)
        
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Throughput (Mbps)', fontsize=12)
        ax2.set_title('Average Cell Throughput', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Network Efficiency
        ax3 = axes[1, 0]
        for result in self.results[:5]:
            if 'history_df' in result:
                df = result['history_df']
                ax3.plot(df['episode'], df['efficiency'],
                        alpha=0.5, linewidth=1)
        
        all_efficiency = [r['metrics']['efficiency'] for r in self.results]
        mean_efficiency = np.mean(all_efficiency, axis=0)
        ax3.plot(np.arange(len(mean_efficiency)), mean_efficiency,
                'g-', linewidth=3, label='Mean')
        
        # Add baseline
        baseline_efficiency = np.mean([r['baseline']['efficiency'] for r in self.results])
        ax3.axhline(y=baseline_efficiency, color='r', linestyle='--',
                   label='Baseline', linewidth=2)
        
        ax3.set_xlabel('Episode', fontsize=12)
        ax3.set_ylabel('Efficiency (Mbps/kW)', fontsize=12)
        ax3.set_title('Network Energy Efficiency', fontsize=14, fontweight='bold')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        # Switching Cost
        ax4 = axes[1, 1]
        all_switching = [r['metrics']['switching_cost'] for r in self.results]
        mean_switching = np.mean(all_switching, axis=0)
        episodes = np.arange(len(mean_switching))
        
        ax4.bar(episodes[::10], mean_switching[::10], width=5,
               color='orange', edgecolor='darkorange', alpha=0.7)
        
        # Add rolling average
        window = 20
        rolling_avg = pd.Series(mean_switching).rolling(window=window, center=True).mean()
        ax4.plot(episodes, rolling_avg, 'r-', linewidth=2,
                label=f'Rolling Avg (window={window})')
        
        ax4.set_xlabel('Episode', fontsize=12)
        ax4.set_ylabel('Switching Cost (# cells)', fontsize=12)
        ax4.set_title('Cell Switching Cost', fontsize=14, fontweight='bold')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'energy_performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'energy_performance_metrics.pdf', bbox_inches='tight')
        plt.close()
    
    def plot_convergence_analysis(self, metrics: Dict):
        """
        Plot convergence and stabilization analysis
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Weight Variance Over Time
        ax1 = axes[0]
        
        # Calculate weight variance for each seed
        for result in self.results[:5]:
            if result.get('prob_df') is not None:
                prob_df = result['prob_df']
                weight_variance = prob_df.var(axis=1)
                episodes = np.arange(len(weight_variance))
                ax1.plot(episodes, weight_variance, alpha=0.5, linewidth=1)
        
        # Mark stabilization points
        if 'avg_stabilization_episode' in metrics:
            ax1.axvline(x=metrics['avg_stabilization_episode'], 
                       color='r', linestyle='--', linewidth=2,
                       label=f"Avg Stabilization: {metrics['avg_stabilization_episode']:.0f}")
        
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Weight Variance', fontsize=12)
        ax1.set_title('EXP3 Weight Convergence', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Performance Stability
        ax2 = axes[1]
        
        # Calculate rolling variance of efficiency
        window = 50
        for result in self.results[:5]:
            if 'history_df' in result:
                df = result['history_df']
                rolling_var = df['efficiency'].rolling(window=window).var()
                ax2.plot(df['episode'], rolling_var, alpha=0.5, linewidth=1)
        
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Performance Variance', fontsize=12)
        ax2.set_title(f'Performance Stability (window={window})', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'convergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'convergence_analysis.pdf', bbox_inches='tight')
        plt.close()
    
    def create_summary_table(self, metrics: Dict):
        """
        Create a summary table of key metrics
        """
        summary_data = {
            'Metric': [],
            'Mean ± Std': [],
            'Min': [],
            'Max': []
        }
        
        # Define key metrics to summarize
        key_metrics = {
            'Final Energy Savings (%)': 'final_energy_savings',
            'Avg Cell Throughput (Mbps)': 'avg_cell_throughput',
            'Avg Switching Cost': 'avg_switching_cost',
            'Performance Variance': 'performance_variance',
            'Stabilization Episode': 'avg_stabilization_episode'
        }
        
        for display_name, metric_key in key_metrics.items():
            if metric_key in metrics:
                value = metrics[metric_key]
                
                # Get std if available
                std_key = metric_key.replace('avg_', 'std_')
                if std_key in metrics:
                    std = metrics[std_key]
                    summary_data['Mean ± Std'].append(f"{value:.2f} ± {std:.2f}")
                else:
                    summary_data['Mean ± Std'].append(f"{value:.2f}")
                
                summary_data['Metric'].append(display_name)
                
                # Calculate min/max from raw data
                if 'energy_savings' in metric_key:
                    all_values = [r['metrics']['energy_savings'][-1] for r in self.results 
                                 if r['metrics']['energy_savings']]
                elif 'throughput' in metric_key:
                    all_values = [np.mean(r['metrics']['throughput']) for r in self.results]
                elif 'switching' in metric_key:
                    all_values = [np.mean(r['metrics']['switching_cost']) for r in self.results]
                else:
                    all_values = [value]
                
                if all_values:
                    summary_data['Min'].append(f"{min(all_values):.2f}")
                    summary_data['Max'].append(f"{max(all_values):.2f}")
                else:
                    summary_data['Min'].append('N/A')
                    summary_data['Max'].append('N/A')
        
        # Create DataFrame and save
        summary_df = pd.DataFrame(summary_data)
        
        # Save as TSV
        summary_df.to_csv(self.output_dir / 'summary_table.tsv', sep='\t', index=False)
        
        # Save as LaTeX
        latex_table = summary_df.to_latex(index=False, escape=False,
                                         column_format='l' + 'r'*(len(summary_df.columns)-1))
        with open(self.output_dir / 'summary_table.tex', 'w') as f:
            f.write(latex_table)
        
        # Print to console
        print("\n" + "="*60)
        print("SUMMARY TABLE")
        print("="*60)
        print(summary_df.to_string(index=False))
        print("="*60 + "\n")
        
        return summary_df
    
    def generate_full_report(self):
        """
        Generate complete evaluation report with all plots and metrics
        """
        print("Generating EXP3 Cell On/Off Evaluation Report...")
        
        # Calculate all metrics
        metrics = self.calculate_evaluation_metrics()
        
        # Save metrics to JSON
        metrics_serializable = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                               for k, v in metrics.items()}
        with open(self.output_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        # Generate all plots
        print("Creating visualizations...")
        self.plot_cumulative_rewards_regrets(metrics)
        self.plot_selection_distribution(metrics)
        self.plot_energy_performance_metrics(metrics)
        self.plot_convergence_analysis(metrics)
        
        # Create summary table
        print("Creating summary table...")
        summary_df = self.create_summary_table(metrics)
        
        # Print final summary
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        print(f"Number of seeds analyzed: {len(self.results)}")
        print(f"Average energy savings: {metrics.get('avg_energy_savings', 0):.2f}%")
        print(f"Final energy savings: {metrics.get('final_energy_savings', 0):.2f}%")
        print(f"Average cell throughput: {metrics.get('avg_cell_throughput', 0):.2f} Mbps")
        
        if 'avg_stabilization_episode' in metrics:
            print(f"Average stabilization episode: {metrics['avg_stabilization_episode']:.0f}")
        
        print(f"\nResults saved to: {self.output_dir}")
        print("="*60)


def main():
    """
    Main function to run evaluation
    """
    parser = argparse.ArgumentParser(description='Evaluate EXP3 Cell On/Off Results')
    parser.add_argument('--output-dir', type=str, 
                       default='data/output/exp3_cell_onoff',
                       help='Directory containing EXP3 results')
    
    args = parser.parse_args()
    
    # Create evaluator and generate report
    evaluator = EXP3Evaluator(args.output_dir)
    evaluator.generate_full_report()


if __name__ == '__main__':
    main()
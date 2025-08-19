#!/usr/bin/env python3
"""
Multi-seed EXP3 analysis and visualization script.
Aggregates results from multiple seed experiments and generates comprehensive plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import argparse
from typing import List, Dict, Tuple
import re

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class EXP3MultiSeedAnalyzer:
    """Analyze and visualize multi-seed EXP3 experiment results."""
    
    def __init__(self, output_dir: str, seeds: List[int] = None):
        self.output_dir = Path(output_dir)
        
        # seeds가 지정되지 않으면 폴더에서 자동 감지
        if seeds is None:
            self.seeds = self._detect_seeds()
        else:
            self.seeds = seeds
            
        self.metrics_dfs = {}
        self.episode_dfs = {}
        self.convergence_dfs = {}
    
    def _detect_seeds(self) -> List[int]:
        """자동으로 사용 가능한 시드 감지"""
        detected_seeds = set()
        
        # 하위 폴더명에서 시드 감지
        for subdir in self.output_dir.iterdir():
            if subdir.is_dir():
                match = re.search(r'_s(\d+)_', subdir.name)
                if match:
                    detected_seeds.add(int(match.group(1)))
        
        # 파일명에서 시드 감지
        for file_path in self.output_dir.glob("*seed*.tsv"):
            match = re.search(r'seed_(\d+)', file_path.name)
            if match:
                detected_seeds.add(int(match.group(1)))
        
        if detected_seeds:
            seeds = sorted(list(detected_seeds))
            print(f"Auto-detected seeds: {seeds}")
            return seeds
        else:
            # 기본값: 0-9 또는 1-10 시도
            print("No seeds detected, using default range 0-9")
            return list(range(0, 10))
        
    def load_data(self):
        """Load all metrics files for specified seeds."""
        print("Loading data from multiple seeds...")
        
        # 먼저 하위 폴더에서 파일 찾기 (eco_sN_p0_0 형식의 폴더들)
        subdirs = [d for d in self.output_dir.iterdir() if d.is_dir()]
        
        if subdirs:
            print(f"Found {len(subdirs)} subdirectories. Searching for metrics files...")
            
            for subdir in subdirs:
                # 하위 폴더명에서 시드 번호 추출 (예: eco_s0_p0_0)
                match = re.search(r'_s(\d+)_', subdir.name)
                if match:
                    seed = int(match.group(1))
                    
                    # exp3_metrics_seed_N.tsv 패턴 확인
                    metrics_path = subdir / f"exp3_metrics_seed_{seed}.tsv"
                    if metrics_path.exists():
                        self.metrics_dfs[seed] = pd.read_csv(metrics_path, sep='\t')
                        print(f"  Loaded metrics for seed {seed} from {subdir.name}/")
                    
                    episode_path = subdir / f"episode_metrics_seed_{seed}.tsv"
                    if episode_path.exists():
                        self.episode_dfs[seed] = pd.read_csv(episode_path, sep='\t')
                    
                    conv_path = subdir / f"convergence_metrics_seed_{seed}.tsv"
                    if conv_path.exists():
                        self.convergence_dfs[seed] = pd.read_csv(conv_path, sep='\t')
                    
                    # 대체 패턴: *_exp3_metrics.tsv
                    if seed not in self.metrics_dfs:
                        for file_path in subdir.glob("*_exp3_metrics.tsv"):
                            self.metrics_dfs[seed] = pd.read_csv(file_path, sep='\t')
                            print(f"  Loaded metrics for seed {seed} from {file_path.name}")
                            break
                        
                        for file_path in subdir.glob("*_episode_metrics.tsv"):
                            self.episode_dfs[seed] = pd.read_csv(file_path, sep='\t')
                            break
                        
                        for file_path in subdir.glob("*_convergence_metrics.tsv"):
                            self.convergence_dfs[seed] = pd.read_csv(file_path, sep='\t')
                            break
        
        # 현재 디렉토리에서도 찾기 (하위 폴더가 없는 경우)
        if not self.metrics_dfs:
            print("Searching in main directory...")
            
            # 파일 패턴 찾기 - 실제 KISS 시뮬레이터의 파일명 패턴에 맞춤
            for file_path in self.output_dir.glob("*_exp3_metrics.tsv"):
                # 파일명에서 시드 번호 추출 (예: exp3_s1_p24_0_exp3_metrics.tsv)
                match = re.search(r'_s(\d+)_', file_path.stem)
                if match:
                    seed = int(match.group(1))
                    if seed in self.seeds:
                        self.metrics_dfs[seed] = pd.read_csv(file_path, sep='\t')
                        print(f"  Loaded metrics for seed {seed} from {file_path.name}")
            
            # Episode metrics 로드
            for file_path in self.output_dir.glob("*_episode_metrics.tsv"):
                match = re.search(r'_s(\d+)_', file_path.stem)
                if match:
                    seed = int(match.group(1))
                    if seed in self.seeds:
                        self.episode_dfs[seed] = pd.read_csv(file_path, sep='\t')
            
            # Convergence metrics 로드
            for file_path in self.output_dir.glob("*_convergence_metrics.tsv"):
                match = re.search(r'_s(\d+)_', file_path.stem)
                if match:
                    seed = int(match.group(1))
                    if seed in self.seeds:
                        self.convergence_dfs[seed] = pd.read_csv(file_path, sep='\t')
            
            # 대체 패턴: exp3_metrics_seed_N.tsv
            if not self.metrics_dfs:
                for seed in self.seeds:
                    metrics_path = self.output_dir / f"exp3_metrics_seed_{seed}.tsv"
                    if metrics_path.exists():
                        self.metrics_dfs[seed] = pd.read_csv(metrics_path, sep='\t')
                        print(f"  Loaded metrics for seed {seed}")
                    
                    episode_path = self.output_dir / f"episode_metrics_seed_{seed}.tsv"
                    if episode_path.exists():
                        self.episode_dfs[seed] = pd.read_csv(episode_path, sep='\t')
                    
                    conv_path = self.output_dir / f"convergence_metrics_seed_{seed}.tsv"
                    if conv_path.exists():
                        self.convergence_dfs[seed] = pd.read_csv(conv_path, sep='\t')
        
        if self.metrics_dfs:
            print(f"Loaded data for {len(self.metrics_dfs)} seeds: {sorted(self.metrics_dfs.keys())}")
        else:
            print("No metrics files found.")
            print(f"Searched in: {self.output_dir}")
            if subdirs:
                print("Subdirectories found:")
                for d in subdirs[:5]:  # 처음 5개만 표시
                    print(f"  - {d.name}")
                    # 각 하위 폴더의 파일 몇 개 표시
                    files = list(d.glob("*.tsv"))[:3]
                    for f in files:
                        print(f"    - {f.name}")
            else:
                print("Files in directory:")
                for f in list(self.output_dir.glob("*.tsv"))[:10]:
                    print(f"  - {f.name}")
    
    def calculate_confidence_intervals(self, data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
        """Calculate mean and confidence interval."""
        mean = np.mean(data)
        std_err = stats.sem(data)
        ci = std_err * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
        return mean, mean - ci, mean + ci
    
    def plot_cumulative_metrics(self):
        """Plot cumulative reward and regret curves with confidence intervals."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Prepare data
        max_episodes = max(df['episode'].max() for df in self.episode_dfs.values())
        episodes = np.arange(1, max_episodes + 1)
        
        # Initialize arrays for each metric
        rewards = np.zeros((len(self.seeds), max_episodes))
        regrets = np.zeros((len(self.seeds), max_episodes))
        efficiencies = np.zeros((len(self.seeds), max_episodes))
        powers = np.zeros((len(self.seeds), max_episodes))
        
        for i, (seed, df) in enumerate(self.episode_dfs.items()):
            # Interpolate to common episode grid
            n_eps = len(df)
            if n_eps > 0:
                # Cumulative reward
                cum_reward = np.cumsum(df['instant_reward'].values)
                rewards[i, :n_eps] = cum_reward
                if n_eps < max_episodes:
                    rewards[i, n_eps:] = cum_reward[-1]
                
                # Cumulative regret (approximate)
                best_reward = df['instant_reward'].max()
                instant_regret = best_reward - df['instant_reward'].values
                cum_regret = np.cumsum(instant_regret)
                regrets[i, :n_eps] = cum_regret
                if n_eps < max_episodes:
                    regrets[i, n_eps:] = cum_regret[-1]
                
                # Efficiency
                efficiencies[i, :n_eps] = df['instant_efficiency'].values
                if n_eps < max_episodes:
                    efficiencies[i, n_eps:] = df['instant_efficiency'].values[-1]
                
                # Power
                powers[i, :n_eps] = df['instant_power'].values
                if n_eps < max_episodes:
                    powers[i, n_eps:] = df['instant_power'].values[-1]
        
        # Plot 1: Cumulative Reward
        ax = axes[0, 0]
        mean_rewards = np.mean(rewards, axis=0)
        std_rewards = np.std(rewards, axis=0)
        ax.plot(episodes, mean_rewards, 'b-', linewidth=2, label='Mean')
        ax.fill_between(episodes, 
                        mean_rewards - std_rewards, 
                        mean_rewards + std_rewards, 
                        alpha=0.3, label='±1 Std Dev')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Reward')
        ax.set_title('Cumulative Reward Over Episodes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative Regret
        ax = axes[0, 1]
        mean_regrets = np.mean(regrets, axis=0)
        std_regrets = np.std(regrets, axis=0)
        ax.plot(episodes, mean_regrets, 'r-', linewidth=2, label='Mean')
        ax.fill_between(episodes, 
                        mean_regrets - std_regrets, 
                        mean_regrets + std_regrets, 
                        alpha=0.3, label='±1 Std Dev')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Regret')
        ax.set_title('Cumulative Regret Over Episodes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Network Efficiency
        ax = axes[1, 0]
        mean_eff = np.mean(efficiencies, axis=0)
        ci_upper = np.percentile(efficiencies, 97.5, axis=0)
        ci_lower = np.percentile(efficiencies, 2.5, axis=0)
        ax.plot(episodes, mean_eff, 'g-', linewidth=2, label='Mean')
        ax.fill_between(episodes, ci_lower, ci_upper, alpha=0.3, label='95% CI')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Efficiency (bits/J)')
        ax.set_title('Network Efficiency Over Episodes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Power Consumption
        ax = axes[1, 1]
        mean_power = np.mean(powers, axis=0)
        std_power = np.std(powers, axis=0)
        ax.plot(episodes, mean_power, 'm-', linewidth=2, label='Mean')
        ax.fill_between(episodes, 
                        mean_power - std_power, 
                        mean_power + std_power, 
                        alpha=0.3, label='±1 Std Dev')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Power (kW)')
        ax.set_title('Power Consumption Over Episodes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cumulative_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved cumulative metrics plot")
    
    def plot_convergence_analysis(self):
        """Plot convergence analysis including stability metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Convergence episodes across seeds
        convergence_episodes = []
        for seed, df in self.metrics_dfs.items():
            conv_ep = df['convergence_episode'].iloc[-1]
            if conv_ep > 0:
                convergence_episodes.append(conv_ep)
        
        # Plot 1: Convergence Episode Distribution
        ax = axes[0, 0]
        if convergence_episodes:
            ax.hist(convergence_episodes, bins=20, edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(convergence_episodes), color='red', 
                      linestyle='--', linewidth=2, label=f'Mean: {np.mean(convergence_episodes):.1f}')
            ax.set_xlabel('Convergence Episode')
            ax.set_ylabel('Number of Seeds')
            ax.set_title('Distribution of Convergence Episodes')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No convergence detected', 
                   ha='center', va='center', transform=ax.transAxes)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Probability Variance Over Time
        ax = axes[0, 1]
        for seed, df in self.convergence_dfs.items():
            if not df.empty:
                ax.plot(df['episode'], df['prob_variance'], alpha=0.5, linewidth=1)
        
        # Add mean line
        if self.convergence_dfs:
            all_episodes = []
            all_variances = []
            for df in self.convergence_dfs.values():
                all_episodes.extend(df['episode'].values)
                all_variances.extend(df['prob_variance'].values)
            
            # Bin and average
            bins = np.linspace(min(all_episodes), max(all_episodes), 50)
            digitized = np.digitize(all_episodes, bins)
            mean_variances = [np.mean([v for i, v in enumerate(all_variances) if digitized[i] == j]) 
                             for j in range(1, len(bins))]
            ax.plot(bins[:-1], mean_variances, 'r-', linewidth=2, label='Mean')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Probability Variance')
        ax.set_title('Probability Distribution Stability')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Average Regret Over Seeds
        ax = axes[1, 0]
        avg_regrets = []
        episodes_list = []
        
        for seed, df in self.metrics_dfs.items():
            avg_regrets.append(df['avg_regret'].values)
            episodes_list.append(df['episode'].values)
        
        # Find common episode range
        min_len = min(len(r) for r in avg_regrets)
        avg_regrets_aligned = np.array([r[:min_len] for r in avg_regrets])
        episodes_common = episodes_list[0][:min_len]
        
        mean_avg_regret = np.mean(avg_regrets_aligned, axis=0)
        std_avg_regret = np.std(avg_regrets_aligned, axis=0)
        
        ax.plot(episodes_common, mean_avg_regret, 'b-', linewidth=2, label='Mean')
        ax.fill_between(episodes_common,
                        mean_avg_regret - std_avg_regret,
                        mean_avg_regret + std_avg_regret,
                        alpha=0.3, label='±1 Std Dev')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Regret')
        ax.set_title('Average Regret Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Switching Cost
        ax = axes[1, 1]
        switching_costs = []
        for seed, df in self.episode_dfs.items():
            if 'switching_cost' in df.columns:
                switching_costs.append(df['switching_cost'].values)
        
        if switching_costs:
            min_len = min(len(s) for s in switching_costs)
            switching_aligned = np.array([s[:min_len] for s in switching_costs])
            episodes_switch = np.arange(1, min_len + 1)
            
            mean_switch = np.mean(switching_aligned, axis=0)
            std_switch = np.std(switching_aligned, axis=0)
            
            ax.plot(episodes_switch, mean_switch, 'g-', linewidth=2, label='Mean')
            ax.fill_between(episodes_switch,
                           mean_switch - std_switch,
                           mean_switch + std_switch,
                           alpha=0.3, label='±1 Std Dev')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Switching Cost')
        ax.set_title('Configuration Switching Cost')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'convergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved convergence analysis plot")
    
    def plot_energy_savings(self):
        """Plot energy savings analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Collect energy savings data
        savings_all_on = []
        savings_random = []
        
        for seed, df in self.metrics_dfs.items():
            if 'energy_savings_vs_all_on' in df.columns:
                savings_all_on.append(df['energy_savings_vs_all_on'].iloc[-1])
            if 'energy_savings_vs_random' in df.columns:
                savings_random.append(df['energy_savings_vs_random'].iloc[-1])
        
        # Plot 1: Energy Savings Distribution
        ax = axes[0]
        data_to_plot = []
        labels = []
        
        if savings_all_on:
            data_to_plot.append(savings_all_on)
            labels.append('vs All-On')
        if savings_random:
            data_to_plot.append(savings_random)
            labels.append('vs Random')
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            colors = ['lightblue', 'lightgreen']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_ylabel('Energy Savings (%)')
            ax.set_title('Energy Savings Distribution Across Seeds')
            ax.grid(True, alpha=0.3)
            
            # Add mean values
            for i, data in enumerate(data_to_plot):
                ax.text(i+1, np.mean(data), f'μ={np.mean(data):.1f}%', 
                       ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Energy Savings Over Time
        ax = axes[1]
        for seed, df in self.metrics_dfs.items():
            if 'energy_savings_vs_all_on' in df.columns:
                ax.plot(df['episode'], df['energy_savings_vs_all_on'], 
                       alpha=0.5, linewidth=1)
        
        # Add mean trajectory
        if self.metrics_dfs:
            all_episodes = []
            all_savings = []
            for df in self.metrics_dfs.values():
                if 'energy_savings_vs_all_on' in df.columns:
                    all_episodes.extend(df['episode'].values)
                    all_savings.extend(df['energy_savings_vs_all_on'].values)
            
            if all_episodes:
                # Bin and average
                bins = np.linspace(min(all_episodes), max(all_episodes), 50)
                digitized = np.digitize(all_episodes, bins)
                mean_savings = [np.mean([s for i, s in enumerate(all_savings) if digitized[i] == j]) 
                               for j in range(1, len(bins))]
                ax.plot(bins[:-1], mean_savings, 'r-', linewidth=2, label='Mean')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Energy Savings vs All-On (%)')
        ax.set_title('Energy Savings Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'energy_savings.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved energy savings plot")
    
    def generate_summary_table(self):
        """Generate summary statistics table."""
        summary_data = []
        
        for seed in self.seeds:
            if seed not in self.metrics_dfs:
                continue
            
            df = self.metrics_dfs[seed]
            last_row = df.iloc[-1]
            
            summary_data.append({
                'Seed': seed,
                'Episodes': last_row['episode'],
                'Convergence': last_row['convergence_episode'] if last_row['convergence_episode'] > 0 else 'No',
                'Cum. Reward': f"{last_row['cumulative_reward']:.2f}",
                'Cum. Regret': f"{last_row['cumulative_regret']:.2f}",
                'Avg. Efficiency': f"{last_row['avg_efficiency']:.2f} ± {last_row.get('efficiency_std', 0):.2f}",
                'Avg. Throughput': f"{last_row['avg_throughput']:.2f} ± {last_row.get('throughput_std', 0):.2f}",
                'Avg. Power': f"{last_row['avg_power']:.2f} ± {last_row.get('power_std', 0):.2f}",
                'Energy Savings': f"{last_row.get('energy_savings_vs_all_on', 0):.1f}%",
                'Avg. Switch Cost': f"{last_row.get('avg_switching_cost', 0):.3f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Add overall statistics
        overall = {
            'Seed': 'Overall',
            'Episodes': '-',
            'Convergence': f"{sum(1 for d in summary_data if d['Convergence'] != 'No')}/{len(summary_data)}",
            'Cum. Reward': f"{np.mean([float(d['Cum. Reward']) for d in summary_data]):.2f}",
            'Cum. Regret': f"{np.mean([float(d['Cum. Regret']) for d in summary_data]):.2f}",
            'Avg. Efficiency': f"{np.mean([float(d['Avg. Efficiency'].split(' ±')[0]) for d in summary_data]):.2f}",
            'Avg. Throughput': f"{np.mean([float(d['Avg. Throughput'].split(' ±')[0]) for d in summary_data]):.2f}",
            'Avg. Power': f"{np.mean([float(d['Avg. Power'].split(' ±')[0]) for d in summary_data]):.2f}",
            'Energy Savings': f"{np.mean([float(d['Energy Savings'].rstrip('%')) for d in summary_data]):.1f}%",
            'Avg. Switch Cost': f"{np.mean([float(d['Avg. Switch Cost']) for d in summary_data]):.3f}"
        }
        
        summary_df = pd.concat([summary_df, pd.DataFrame([overall])], ignore_index=True)
        
        # Save to file
        summary_path = self.output_dir / 'summary_statistics.tsv'
        summary_df.to_csv(summary_path, sep='\t', index=False)
        print(f"\nSummary statistics saved to: {summary_path}")
        print("\nSummary Table:")
        print(summary_df.to_string(index=False))
        
        return summary_df
    
    def run_analysis(self):
        """Run complete analysis pipeline."""
        print("="*80)
        print("EXP3 MULTI-SEED ANALYSIS")
        print("="*80)
        
        # Load data
        self.load_data()
        
        if not self.metrics_dfs:
            print("No data found. Please run experiments first.")
            return
        
        # Generate plots
        print("\nGenerating visualizations...")
        self.plot_cumulative_metrics()
        self.plot_convergence_analysis()
        self.plot_energy_savings()
        
        # Generate summary table
        print("\nGenerating summary statistics...")
        self.generate_summary_table()
        
        print("\n" + "="*80)
        print("Analysis complete! Check output directory for results.")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Analyze multi-seed EXP3 experiments')
    parser.add_argument('-d', '--data-dir', type=str, required=True,
                       help='Directory containing experiment outputs')
    parser.add_argument('-s', '--seeds', type=int, nargs='+', default=None,
                       help='List of seed values to analyze (default: 1-10)')
    
    args = parser.parse_args()
    
    # Create analyzer and run
    analyzer = EXP3MultiSeedAnalyzer(args.data_dir, args.seeds)
    analyzer.run_analysis()


if __name__ == '__main__':
    main()
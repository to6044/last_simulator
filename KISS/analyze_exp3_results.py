#!/usr/bin/env python3
"""
Multi-seed EXP3 analysis and visualization script with post-simulation regret calculation.
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
    """Analyze and visualize multi-seed EXP3 experiment results with post-simulation regret calculation."""
    
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
            print(f"Found {len(subdirs)} subdirectories.")
            for subdir in subdirs:
                for seed in self.seeds:
                    # Try different file naming patterns
                    patterns = [
                        f"*_s{seed}_*_exp3_metrics.tsv",
                        f"*seed_{seed}_*_exp3_metrics.tsv",
                        f"*seed{seed}_*_exp3_metrics.tsv"
                    ]
                    
                    for pattern in patterns:
                        metrics_files = list(subdir.glob(pattern))
                        if metrics_files:
                            self.metrics_dfs[seed] = pd.read_csv(metrics_files[0], sep='\t')
                            print(f"  Loaded seed {seed} metrics from {subdir.name}")
                            
                            # Load episode metrics
                            episode_file = str(metrics_files[0]).replace('_exp3_metrics.tsv', '_episode_metrics.tsv')
                            if Path(episode_file).exists():
                                self.episode_dfs[seed] = pd.read_csv(episode_file, sep='\t')
                            
                            # Load convergence metrics
                            conv_file = str(metrics_files[0]).replace('_exp3_metrics.tsv', '_convergence_metrics.tsv')
                            if Path(conv_file).exists():
                                self.convergence_dfs[seed] = pd.read_csv(conv_file, sep='\t')
                            break
        
        # 직접 output_dir에서도 찾기
        else:
            for seed in self.seeds:
                patterns = [
                    f"*_s{seed}_*_exp3_metrics.tsv",
                    f"*seed_{seed}_*_exp3_metrics.tsv"
                ]
                
                for pattern in patterns:
                    metrics_files = list(self.output_dir.glob(pattern))
                    if metrics_files:
                        self.metrics_dfs[seed] = pd.read_csv(metrics_files[0], sep='\t')
                        print(f"  Loaded seed {seed} metrics")
                        
                        # Load episode metrics
                        episode_file = str(metrics_files[0]).replace('_exp3_metrics.tsv', '_episode_metrics.tsv')
                        if Path(episode_file).exists():
                            self.episode_dfs[seed] = pd.read_csv(episode_file, sep='\t')
                        
                        # Load convergence metrics
                        conv_file = str(metrics_files[0]).replace('_exp3_metrics.tsv', '_convergence_metrics.tsv')
                        if Path(conv_file).exists():
                            self.convergence_dfs[seed] = pd.read_csv(conv_file, sep='\t')
                        break
        
        print(f"Successfully loaded data for {len(self.metrics_dfs)} seeds: {list(self.metrics_dfs.keys())}")
    
    def calculate_regret(self, rewards: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calculate regret from reward history after simulation.
        
        Returns:
            instant_regret: Array of instant regret values
            cumulative_regret: Array of cumulative regret
            best_arm_reward: Estimated best arm reward
        """
        if len(rewards) == 0:
            return np.array([]), np.array([]), 0.0
        
        # Estimate best arm reward using different strategies
        # Strategy 1: Use maximum observed reward
        max_reward = np.max(rewards)
        
        # Strategy 2: Use high percentile (e.g., 95th) to be more robust to outliers
        percentile_reward = np.percentile(rewards, 95)
        
        # Strategy 3: Use rolling maximum with smoothing
        window_size = min(50, len(rewards) // 10)
        if window_size > 0:
            rolling_max = pd.Series(rewards).rolling(window=window_size, min_periods=1).max().values
            smoothed_best = np.maximum.accumulate(rolling_max)
        else:
            smoothed_best = np.maximum.accumulate(rewards)
        
        # Use a combination of strategies for robust estimation
        best_arm_reward = 0.9 * max_reward + 0.1 * percentile_reward
        
        # Calculate instant regret
        instant_regret = best_arm_reward - rewards
        
        # Calculate cumulative regret
        cumulative_regret = np.cumsum(instant_regret)
        
        return instant_regret, cumulative_regret, best_arm_reward
    
    def calculate_confidence_interval(self, data, confidence=0.95):
        """Calculate confidence interval for given data."""
        mean = np.mean(data)
        std_err = stats.sem(data)
        ci = std_err * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
        return mean, mean - ci, mean + ci
    
    def plot_cumulative_metrics(self):
        """Plot cumulative reward and other metrics with confidence intervals."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Prepare data
        max_episodes = max(df['episode'].max() for df in self.episode_dfs.values() if not df.empty)
        episodes = np.arange(1, max_episodes + 1)
        
        # Initialize arrays for each metric
        rewards = np.zeros((len(self.seeds), max_episodes))
        efficiencies = np.zeros((len(self.seeds), max_episodes))
        powers = np.zeros((len(self.seeds), max_episodes))
        throughputs = np.zeros((len(self.seeds), max_episodes))
        
        for i, (seed, df) in enumerate(self.episode_dfs.items()):
            # Interpolate to common episode grid
            n_eps = len(df)
            if n_eps > 0:
                # Cumulative reward
                cum_reward = np.cumsum(df['instant_reward'].values)
                rewards[i, :n_eps] = cum_reward
                if n_eps < max_episodes:
                    rewards[i, n_eps:] = cum_reward[-1]
                
                # Efficiency
                efficiencies[i, :n_eps] = df['instant_efficiency'].values
                if n_eps < max_episodes:
                    efficiencies[i, n_eps:] = df['instant_efficiency'].values[-1]
                
                # Power
                powers[i, :n_eps] = df['instant_power'].values
                if n_eps < max_episodes:
                    powers[i, n_eps:] = df['instant_power'].values[-1]
                
                # Throughput
                if 'instant_throughput' in df.columns:
                    throughputs[i, :n_eps] = df['instant_throughput'].values
                    if n_eps < max_episodes:
                        throughputs[i, n_eps:] = df['instant_throughput'].values[-1]
        
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
        
        # Plot 2: Efficiency
        ax = axes[0, 1]
        mean_eff = np.mean(efficiencies, axis=0)
        std_eff = np.std(efficiencies, axis=0)
        ax.plot(episodes, mean_eff, 'g-', linewidth=2, label='Mean')
        ax.fill_between(episodes,
                        mean_eff - std_eff,
                        mean_eff + std_eff,
                        alpha=0.3, label='±1 Std Dev')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Efficiency (bits/J)')
        ax.set_title('Energy Efficiency Over Episodes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Power Consumption
        ax = axes[1, 0]
        mean_power = np.mean(powers, axis=0)
        std_power = np.std(powers, axis=0)
        ax.plot(episodes, mean_power, 'r-', linewidth=2, label='Mean')
        ax.fill_between(episodes,
                        mean_power - std_power,
                        mean_power + std_power,
                        alpha=0.3, label='±1 Std Dev')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Power (kW)')
        ax.set_title('Power Consumption Over Episodes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Throughput
        ax = axes[1, 1]
        mean_tp = np.mean(throughputs, axis=0)
        std_tp = np.std(throughputs, axis=0)
        ax.plot(episodes, mean_tp, 'm-', linewidth=2, label='Mean')
        ax.fill_between(episodes,
                        mean_tp - std_tp,
                        mean_tp + std_tp,
                        alpha=0.3, label='±1 Std Dev')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Throughput (Mb/s)')
        ax.set_title('Network Throughput Over Episodes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cumulative_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved cumulative metrics plot")
    
    def plot_regret_analysis(self):
        """Plot comprehensive regret analysis with post-simulation calculation."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Initialize arrays for regret metrics
        all_instant_regrets = []
        all_cumulative_regrets = []
        all_avg_regrets = []
        all_best_estimates = []
        
        for seed, df in self.episode_dfs.items():
            if 'instant_reward' in df.columns:
                rewards = df['instant_reward'].values
                instant_regret, cumulative_regret, best_estimate = self.calculate_regret(rewards)
                
                all_instant_regrets.append(instant_regret)
                all_cumulative_regrets.append(cumulative_regret)
                all_avg_regrets.append(cumulative_regret / np.arange(1, len(cumulative_regret) + 1))
                all_best_estimates.append(best_estimate)
        
        # Plot 1: Instant Regret Over Time
        ax = axes[0, 0]
        has_instant_regret = False
        for i, instant_regret in enumerate(all_instant_regrets):
            ax.plot(np.arange(1, len(instant_regret) + 1), instant_regret, 
                   alpha=0.3, linewidth=0.5)
            has_instant_regret = True
        
        # Add mean line
        if all_instant_regrets:
            min_len = min(len(r) for r in all_instant_regrets)
            aligned_regrets = np.array([r[:min_len] for r in all_instant_regrets])
            mean_instant = np.mean(aligned_regrets, axis=0)
            ax.plot(np.arange(1, min_len + 1), mean_instant, 
                   'r-', linewidth=2, label='Mean')
            ax.legend()
        elif not has_instant_regret:
            ax.text(0.5, 0.5, 'No instant regret data available', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Instant Regret')
        ax.set_title('Instant Regret Per Episode')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative Regret
        ax = axes[0, 1]
        has_cum_regret = False
        for i, cum_regret in enumerate(all_cumulative_regrets):
            ax.plot(np.arange(1, len(cum_regret) + 1), cum_regret, 
                   alpha=0.3, linewidth=0.5)
            has_cum_regret = True
        
        # Add mean and confidence interval
        if all_cumulative_regrets:
            min_len = min(len(r) for r in all_cumulative_regrets)
            aligned_cum_regrets = np.array([r[:min_len] for r in all_cumulative_regrets])
            mean_cum = np.mean(aligned_cum_regrets, axis=0)
            std_cum = np.std(aligned_cum_regrets, axis=0)
            
            episodes_aligned = np.arange(1, min_len + 1)
            ax.plot(episodes_aligned, mean_cum, 'b-', linewidth=2, label='Mean')
            ax.fill_between(episodes_aligned,
                           mean_cum - std_cum,
                           mean_cum + std_cum,
                           alpha=0.3, label='±1 Std Dev')
            ax.legend()
        elif not has_cum_regret:
            ax.text(0.5, 0.5, 'No cumulative regret data available', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Regret')
        ax.set_title('Cumulative Regret Over Episodes')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Average Regret
        ax = axes[0, 2]
        has_avg_regret = False
        for avg_regret in all_avg_regrets:
            ax.plot(np.arange(1, len(avg_regret) + 1), avg_regret, 
                   alpha=0.3, linewidth=0.5)
            has_avg_regret = True
        
        if all_avg_regrets:
            min_len = min(len(r) for r in all_avg_regrets)
            aligned_avg_regrets = np.array([r[:min_len] for r in all_avg_regrets])
            mean_avg = np.mean(aligned_avg_regrets, axis=0)
            std_avg = np.std(aligned_avg_regrets, axis=0)
            
            episodes_aligned = np.arange(1, min_len + 1)
            ax.plot(episodes_aligned, mean_avg, 'g-', linewidth=2, label='Mean')
            ax.fill_between(episodes_aligned,
                           mean_avg - std_avg,
                           mean_avg + std_avg,
                           alpha=0.3, label='±1 Std Dev')
            ax.legend()
        elif not has_avg_regret:
            ax.text(0.5, 0.5, 'No average regret data available', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Regret')
        ax.set_title('Average Regret (Cumulative/Episode)')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Regret Distribution
        ax = axes[1, 0]
        final_regrets = [r[-1] if len(r) > 0 else 0 for r in all_cumulative_regrets]
        if final_regrets:
            ax.hist(final_regrets, bins=20, edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(final_regrets), color='r', linestyle='--', 
                      label=f'Mean: {np.mean(final_regrets):.2f}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No final regret data available', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set_xlabel('Final Cumulative Regret')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Final Regret Across Seeds')
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Regret vs Reward Correlation
        ax = axes[1, 1]
        for seed, df in self.episode_dfs.items():
            if 'instant_reward' in df.columns:
                rewards = df['instant_reward'].values
                _, cum_regret, _ = self.calculate_regret(rewards)
                cum_reward = np.cumsum(rewards)
                
                if len(cum_regret) > 0 and len(cum_reward) > 0:
                    # Sample points to avoid overcrowding
                    sample_indices = np.linspace(0, len(cum_regret)-1, 
                                               min(100, len(cum_regret)), dtype=int)
                    ax.scatter(cum_reward[sample_indices], cum_regret[sample_indices], 
                             alpha=0.5, s=10)
        
        ax.set_xlabel('Cumulative Reward')
        ax.set_ylabel('Cumulative Regret')
        ax.set_title('Regret vs Reward Trade-off')
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Regret Rate Over Time
        ax = axes[1, 2]
        window_size = 50
        has_regret_rate = False
        for seed, df in self.episode_dfs.items():
            if 'instant_reward' in df.columns:
                rewards = df['instant_reward'].values
                instant_regret, _, _ = self.calculate_regret(rewards)
                
                if len(instant_regret) >= window_size:
                    # Calculate rolling average of regret
                    regret_rate = pd.Series(instant_regret).rolling(
                        window=window_size, min_periods=1).mean().values
                    ax.plot(np.arange(1, len(regret_rate) + 1), regret_rate, 
                           alpha=0.3, linewidth=0.5)
                    has_regret_rate = True
        
        if not has_regret_rate:
            ax.text(0.5, 0.5, f'Insufficient data for {window_size}-episode moving average', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Regret Rate (Moving Average)')
        ax.set_title(f'Regret Rate ({window_size}-Episode Moving Average)')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Post-Simulation Regret Analysis', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'regret_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved regret analysis plot")
    
    def plot_convergence_analysis(self):
        """Plot convergence analysis including probability variance and switching cost."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Probability Variance Over Time
        ax = axes[0, 0]
        has_prob_variance = False
        for seed, df in self.convergence_dfs.items():
            if 'probability_variance' in df.columns:
                ax.plot(df['episode'], df['probability_variance'], 
                       alpha=0.5, linewidth=1, label=f'Seed {seed}')
                has_prob_variance = True
        
        if has_prob_variance and len(self.convergence_dfs) <= 10:
            ax.legend()
        elif not has_prob_variance:
            ax.text(0.5, 0.5, 'No probability variance data available', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Probability Variance')
        ax.set_title('Arm Selection Probability Variance')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Convergence Detection
        ax = axes[0, 1]
        convergence_episodes = []
        for seed, df in self.metrics_dfs.items():
            if 'convergence_episode' in df.columns:
                conv_ep = df['convergence_episode'].iloc[-1]
                if conv_ep > 0:
                    convergence_episodes.append(conv_ep)
        
        if convergence_episodes:
            ax.hist(convergence_episodes, bins=20, edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(convergence_episodes), color='r', linestyle='--',
                      label=f'Mean: {np.mean(convergence_episodes):.1f}')
            ax.set_xlabel('Convergence Episode')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Convergence Episodes')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No convergence detected', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Convergence Detection')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Switching Cost
        ax = axes[1, 0]
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
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No switching cost data available', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Switching Cost')
        ax.set_title('Configuration Switching Cost')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Arm Selection Distribution (Final)
        ax = axes[1, 1]
        final_arm_selections = []
        for seed, df in self.metrics_dfs.items():
            # Get top 5 arms for each seed
            for i in range(1, 6):
                if f'top{i}_selections' in df.columns:
                    selections = df[f'top{i}_selections'].iloc[-1]
                    final_arm_selections.append(selections)
        
        if final_arm_selections:
            ax.hist(final_arm_selections, bins=20, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Number of Selections')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Arm Selection Counts (Top 5 Arms)')
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
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
            
            ax.set_ylabel('Energy Savings (%)')
            ax.set_title('Energy Savings Distribution')
        else:
            ax.text(0.5, 0.5, 'No energy savings data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Energy Savings Distribution')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Energy Savings Over Time
        ax = axes[1]
        has_savings_data = False
        for seed, df in self.metrics_dfs.items():
            if 'energy_savings_vs_all_on' in df.columns:
                ax.plot(df['episode'], df['energy_savings_vs_all_on'], 
                       alpha=0.5, linewidth=1)
                has_savings_data = True
        
        if not has_savings_data:
            ax.text(0.5, 0.5, 'No energy savings data available', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Energy Savings vs All-On (%)')
        ax.set_title('Energy Savings Evolution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'energy_savings.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved energy savings plot")
    
    def generate_summary_table(self):
        """Generate comprehensive summary table including regret statistics."""
        summary_data = []
        
        for seed in sorted(self.metrics_dfs.keys()):
            df_metrics = self.metrics_dfs[seed]
            df_episodes = self.episode_dfs.get(seed, pd.DataFrame())
            
            last_row = df_metrics.iloc[-1]
            
            # Calculate regret if episode data available
            regret_info = {
                'Cum. Regret': 'N/A',
                'Avg. Regret': 'N/A',
                'Max Instant Regret': 'N/A'
            }
            
            if not df_episodes.empty and 'instant_reward' in df_episodes.columns:
                rewards = df_episodes['instant_reward'].values
                instant_regret, cumulative_regret, best_estimate = self.calculate_regret(rewards)
                
                if len(cumulative_regret) > 0:
                    regret_info['Cum. Regret'] = f"{cumulative_regret[-1]:.2f}"
                    regret_info['Avg. Regret'] = f"{cumulative_regret[-1] / len(rewards):.4f}"
                    regret_info['Max Instant Regret'] = f"{np.max(instant_regret):.4f}"
            
            # Get convergence info
            conv_episode = last_row.get('convergence_episode', -1)
            
            summary_data.append({
                'Seed': seed,
                'Episodes': last_row['episode'],
                'Convergence': f"Episode {int(conv_episode)}" if conv_episode > 0 else "No",
                'Cum. Reward': f"{last_row['cumulative_reward']:.2f}",
                'Cum. Regret': regret_info['Cum. Regret'],
                'Avg. Regret': regret_info['Avg. Regret'],
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
        }
        
        # Handle regret statistics safely
        valid_regrets = [float(d['Cum. Regret']) for d in summary_data if d['Cum. Regret'] != 'N/A']
        if valid_regrets:
            overall['Cum. Regret'] = f"{np.mean(valid_regrets):.2f}"
            overall['Avg. Regret'] = f"{np.mean([float(d['Avg. Regret']) for d in summary_data if d['Avg. Regret'] != 'N/A']):.4f}"
        else:
            overall['Cum. Regret'] = 'N/A'
            overall['Avg. Regret'] = 'N/A'
        
        overall.update({
            'Avg. Efficiency': f"{np.mean([float(d['Avg. Efficiency'].split(' ±')[0]) for d in summary_data]):.2f}",
            'Avg. Throughput': f"{np.mean([float(d['Avg. Throughput'].split(' ±')[0]) for d in summary_data]):.2f}",
            'Avg. Power': f"{np.mean([float(d['Avg. Power'].split(' ±')[0]) for d in summary_data]):.2f}",
            'Energy Savings': f"{np.mean([float(d['Energy Savings'].rstrip('%')) for d in summary_data]):.1f}%",
            'Avg. Switch Cost': f"{np.mean([float(d['Avg. Switch Cost']) for d in summary_data]):.3f}"
        })
        
        summary_df = pd.concat([summary_df, pd.DataFrame([overall])], ignore_index=True)
        
        # Save to file
        summary_path = self.output_dir / 'summary_statistics.tsv'
        summary_df.to_csv(summary_path, sep='\t', index=False)
        print(f"\nSummary statistics saved to: {summary_path}")
        print("\nSummary Table:")
        print(summary_df.to_string(index=False))
        
        return summary_df
    
    def run_analysis(self):
        """Run complete analysis pipeline including all visualizations and regret calculation."""
        print("="*80)
        print("EXP3 MULTI-SEED ANALYSIS WITH POST-SIMULATION REGRET CALCULATION")
        print("="*80)
        
        # Load data
        self.load_data()
        
        if not self.metrics_dfs:
            print("No data found. Please run experiments first.")
            return
        
        # Generate all plots
        print("\nGenerating visualizations...")
        
        # Original plots
        print("  - Cumulative metrics...")
        self.plot_cumulative_metrics()
        
        print("  - Convergence analysis...")
        self.plot_convergence_analysis()
        
        print("  - Energy savings...")
        self.plot_energy_savings()
        
        # New regret analysis
        print("  - Regret analysis (post-simulation calculation)...")
        self.plot_regret_analysis()
        
        # Generate summary table
        print("\nGenerating summary statistics...")
        self.generate_summary_table()
        
        print("\n" + "="*80)
        print("Analysis complete! Check output directory for results:")
        print(f"  - {self.output_dir}/cumulative_metrics.png")
        print(f"  - {self.output_dir}/convergence_analysis.png")
        print(f"  - {self.output_dir}/energy_savings.png")
        print(f"  - {self.output_dir}/regret_analysis.png")
        print(f"  - {self.output_dir}/summary_statistics.tsv")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Analyze multi-seed EXP3 experiments with post-simulation regret calculation')
    parser.add_argument('-d', '--data-dir', type=str, required=True,
                       help='Directory containing experiment outputs')
    parser.add_argument('-s', '--seeds', type=int, nargs='+', default=None,
                       help='List of seed values to analyze (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Create analyzer and run
    analyzer = EXP3MultiSeedAnalyzer(args.data_dir, args.seeds)
    analyzer.run_analysis()


if __name__ == '__main__':
    main()
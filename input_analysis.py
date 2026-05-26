import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

def parse_csv_fast(csv_path: str):
    """Load CSV and parse times into seconds since 06:00 efficiently."""
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print("Parsing times...")
    # Fast manual parsing of HH:MM:SS.SSS
    time_parts = df['time'].str.split(':')
    hours = time_parts.str[0].astype(int)
    minutes = time_parts.str[1].astype(int)
    seconds = time_parts.str[2].astype(float)
    df['sec'] = (hours - 6) * 3600 + minutes * 60 + seconds
    return df

# Period definitions (seconds since 06:00)
PERIODS = {
    'am_peak': (3600, 10800),     # 07:00 - 09:00
    'midday': (10800, 39600),    # 09:00 - 17:00
    'pm_peak': (39600, 46800),    # 17:00 - 19:00
    'low_am': (0, 3600),          # 06:00 - 07:00
    'low_pm': (46800, 57600)       # 19:00 - 22:00
}

def analyze_terminal(csv_path: str, terminal_name: str, out_dir: str):
    df = parse_csv_fast(csv_path)
    os.makedirs(out_dir, exist_ok=True)
    
    # Map periods
    df['period'] = 'unknown'
    for p_name, (start, end) in PERIODS.items():
        mask = (df['sec'] >= start) & (df['sec'] < end)
        df.loc[mask, 'period'] = p_name
        
    # Destination splits
    print(f"Calculating destination splits for {terminal_name}...")
    splits = {}
    df_split = df.copy()
    df_split.loc[df_split['period'].isin(['low_am', 'low_pm']), 'period'] = 'low'
    for p_name in ['am_peak', 'midday', 'pm_peak', 'low']:
        sub = df_split[df_split['period'] == p_name]
        counts = sub['destination'].value_counts()
        total = counts.sum()
        splits[p_name] = {dest: count / total for dest, count in counts.items()}
        print(f"  {p_name}: {splits[p_name]}")

    results = []

    # Compute inter-arrival times per period and fit distribution
    for p_name in ['am_peak', 'midday', 'pm_peak', 'low']:
        print(f"Analyzing period: {p_name} for {terminal_name}...")
        
        if p_name == 'low':
            # For 'low', compute differences within low_am and low_pm separately, then pool them
            sub_am = df[df['period'] == 'low_am'].copy().sort_values(by=['date', 'sec'])
            sub_am['prev_sec'] = sub_am.groupby('date')['sec'].shift(1)
            diff_am = (sub_am['sec'] - sub_am['prev_sec']).dropna().to_numpy()
            
            sub_pm = df[df['period'] == 'low_pm'].copy().sort_values(by=['date', 'sec'])
            sub_pm['prev_sec'] = sub_pm.groupby('date')['sec'].shift(1)
            diff_pm = (sub_pm['sec'] - sub_pm['prev_sec']).dropna().to_numpy()
            
            inter_arrivals = np.concatenate([diff_am, diff_pm])
        else:
            sub = df[df['period'] == p_name].copy().sort_values(by=['date', 'sec'])
            sub['prev_sec'] = sub.groupby('date')['sec'].shift(1)
            inter_arrivals = (sub['sec'] - sub['prev_sec']).dropna().to_numpy()
        
        if len(inter_arrivals) == 0:
            print(f"  No inter-arrival data for {p_name}")
            continue
            
        # Fit exponential distribution (scale is the mean inter-arrival time in seconds)
        mean_ia = float(np.mean(inter_arrivals))
        fitted_lambda_sec = 1.0 / mean_ia
        fitted_lambda_min = 60.0 / mean_ia
        
        # Kolmogorov-Smirnov test (H0: Exponential distribution)
        # scipy expon has loc=0 and scale=mean_ia
        ks_stat, p_value = stats.kstest(inter_arrivals, 'expon', args=(0, mean_ia))
        
        print(f"  Mean inter-arrival time: {mean_ia:.4f} s")
        print(f"  Estimated lambda: {fitted_lambda_min:.4f} pax/min")
        print(f"  KS statistic: {ks_stat:.6f}, p-value: {p_value:.6e}")
        
        results.append({
            'terminal': terminal_name,
            'period': p_name,
            'mean_ia_sec': mean_ia,
            'lambda_min': fitted_lambda_min,
            'ks_stat': ks_stat,
            'p_value': p_value,
            'num_samples': len(inter_arrivals)
        })
        
        # Plot Histogram & Q-Q Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Histogram
        n_bins = min(100, int(np.sqrt(len(inter_arrivals))))
        axes[0].hist(inter_arrivals, bins=n_bins, density=True, alpha=0.6, color='#2F6690', label='Empirical')
        x_eval = np.linspace(0, np.percentile(inter_arrivals, 99), 1000)
        axes[0].plot(x_eval, stats.expon.pdf(x_eval, scale=mean_ia), 'r-', lw=2, label='Fitted Expon')
        axes[0].set_title(f'Inter-Arrival Time Histogram: {terminal_name} - {p_name}')
        axes[0].set_xlabel('Inter-Arrival Time (seconds)')
        axes[0].set_ylabel('Density')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Q-Q Plot
        # Sample points to make Q-Q plotting fast for large datasets
        qq_samples = inter_arrivals
        if len(qq_samples) > 10000:
            qq_samples = np.random.choice(qq_samples, size=10000, replace=False)
        qq_samples_sorted = np.sort(qq_samples)
        
        # Theoretical quantiles
        q_empirical = (np.arange(1, len(qq_samples_sorted) + 1) - 0.5) / len(qq_samples_sorted)
        q_theoretical = stats.expon.ppf(q_empirical, scale=mean_ia)
        
        axes[1].scatter(q_theoretical, qq_samples_sorted, alpha=0.5, color='#00798C', s=10, label='Data points')
        max_val = max(q_theoretical.max(), qq_samples_sorted.max())
        axes[1].plot([0, max_val], [0, max_val], 'r--', lw=2, label='45-degree Reference')
        axes[1].set_title(f'Exponential Q-Q Plot: {terminal_name} - {p_name}')
        axes[1].set_xlabel('Theoretical Quantiles (seconds)')
        axes[1].set_ylabel('Empirical Quantiles (seconds)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{terminal_name.lower()}_{p_name}_fit.png'), dpi=200)
        plt.close()
        
    return results, splits

def main():
    print("Starting Input Analysis...")
    out_dir = os.path.join('plots', 'input_analysis')
    
    k_res, k_splits = analyze_terminal('arrivals_kadikoy.csv', 'Kadikoy (A1)', out_dir)
    e_res, e_splits = analyze_terminal('arrivals_eminonu.csv', 'Eminonu (E1)', out_dir)
    
    # Save results to a CSV
    res_df = pd.DataFrame(k_res + e_res)
    res_df.to_csv('input_analysis_results.csv', index=False)
    print("\nInput Analysis Summary saved to input_analysis_results.csv")
    
    # Print formatted markdown table
    print("\n=== INPUT ANALYSIS SUMMARY TABLE (MARKDOWN) ===")
    print("| Terminal | Period | Sample Size | Mean Inter-arrival (s) | Rate (pax/min) | KS Stat | KS p-value | Splits |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in k_res + e_res:
        t_name = r['terminal']
        p_name = r['period']
        s_size = r['num_samples']
        mean_ia = r['mean_ia_sec']
        lam_min = r['lambda_min']
        ks_s = r['ks_stat']
        p_val = r['p_value']
        
        # Get splits formatted
        split_dict = k_splits[p_name] if 'Kadikoy' in t_name else e_splits[p_name]
        split_str = ", ".join([f"{d}: {v*100:.1f}%" for d, v in split_dict.items()])
        
        print(f"| {t_name} | {p_name} | {s_size:,} | {mean_ia:.2f} | {lam_min:.2f} | {ks_s:.4f} | {p_val:.2e} | {split_str} |")

if __name__ == '__main__':
    main()

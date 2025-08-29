#!/usr/bin/env python3
"""
Example usage of the workflow construction statistics analyzer.

This script demonstrates how to use the const_stats module programmatically
to analyze workflow construction metrics.
"""

from const_stats import analyze_workflow_metrics, generate_summary_table, generate_speedup_analysis_table, generate_values_summary_table
import json


def main():
    """Example of programmatic usage."""
    
    # Example 1: Analyze a specific JSON file
    print("Example 1: Analyzing workflow construction metrics")
    print("=" * 60)
    
    # You can specify custom metrics if you want
    custom_metrics = [
        'write_remote_per_event_mb',
        'event_throughput',
        'memory_per_event_mb',
        'network_transfer_per_event_mb'
    ]
    
    # Analyze the metrics
    results, metrics = analyze_workflow_metrics(
        'output/others/5tasks_fullsim/construction_metrics.json',
        custom_metrics
    )
    
    # Generate and display summary table
    print(generate_summary_table(results))
    
    # Generate and display speedup analysis
    print(generate_speedup_analysis_table(results))
    
    # Generate and display values summary
    print(generate_values_summary_table(results, metrics))
    
    # Example 2: Access specific statistics programmatically
    print("\nExample 2: Accessing specific statistics")
    print("=" * 60)
    
    if 'event_throughput' in results:
        throughput_stats = results['event_throughput']
        print(f"Event Throughput Analysis:")
        print(f"  - Mean: {throughput_stats['mean']:.6f}")
        print(f"  - Median: {throughput_stats['median']:.6f}")
        print(f"  - Standard Deviation: {throughput_stats['std_dev']:.6f}")
        print(f"  - Coefficient of Variation: {throughput_stats['cv']:.3f}")
        
        # Calculate efficiency metric (higher throughput is better)
        if throughput_stats['mean'] > 0:
            efficiency = throughput_stats['mean'] / throughput_stats['std_dev']
            print(f"  - Efficiency (mean/std_dev): {efficiency:.3f}")
        
        # Speedup analysis
        if 'max_speedup' in throughput_stats:
            print(f"  - Maximum Speedup: {throughput_stats['max_speedup']:.2f}x")
            if 'mean_to_min_ratio' in throughput_stats:
                print(f"  - Mean/Min Ratio: {throughput_stats['mean_to_min_ratio']:.2f}x")
    
    # Example 3: Compare metrics
    print("\nExample 3: Comparing metrics")
    print("=" * 60)
    
    if 'memory_per_event_mb' in results and 'network_transfer_per_event_mb' in results:
        mem_stats = results['memory_per_event_mb']
        net_stats = results['network_transfer_per_event_mb']
        
        print(f"Memory vs Network Transfer Comparison:")
        print(f"  - Memory CV: {mem_stats['cv']:.3f} (lower is more consistent)")
        print(f"  - Network CV: {net_stats['cv']:.3f} (lower is more consistent)")
        
        if mem_stats['cv'] < net_stats['cv']:
            print(f"  - Memory usage is more consistent than network transfer")
        else:
            print(f"  - Network transfer is more consistent than memory usage")
    
    # Example 4: Save custom analysis
    print("\nExample 4: Saving custom analysis")
    print("=" * 60)
    
    # Create a custom summary with only key metrics
    custom_summary = {}
    for metric_name, stats in results.items():
        custom_summary[metric_name] = {
            'mean': stats['mean'],
            'std_dev': stats['std_dev'],
            'cv': stats['cv'],
            'min': stats['min'],
            'max': stats['max'],
            'max_speedup': stats.get('max_speedup', 0),
            'mean_to_min_ratio': stats.get('mean_to_min_ratio', 0)
        }
    
    # Save to a custom file
    with open('custom_analysis.json', 'w') as f:
        json.dump(custom_summary, f, indent=2)
    
    print("Custom analysis saved to 'custom_analysis.json'")
    
    # Example 5: Performance insights
    print("\nExample 5: Performance insights")
    print("=" * 60)
    
    # Find the most variable metric
    most_variable = max(results.items(), key=lambda x: x[1]['cv'])
    least_variable = min(results.items(), key=lambda x: x[1]['cv'])
    
    print(f"Most variable metric: {most_variable[0]} (CV: {most_variable[1]['cv']:.3f})")
    print(f"Least variable metric: {least_variable[0]} (CV: {least_variable[1]['cv']:.3f})")
    
    # Check for potential outliers using IQR method
    for metric_name, stats in results.items():
        q1, q3 = stats['q1'], stats['q3']
        iqr = stats['iqr']
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = [v for v in [stats['min'], stats['max']] 
                   if v < lower_bound or v > upper_bound]
        
        if outliers:
            print(f"  - {metric_name}: Potential outliers detected")
    
    # Example 6: Speedup analysis insights
    print("\nExample 6: Speedup analysis insights")
    print("=" * 60)
    
    # Find metrics with highest speedup potential
    speedup_metrics = [(name, stats.get('max_speedup', 0)) 
                       for name, stats in results.items() 
                       if 'max_speedup' in stats and stats['max_speedup'] != float('inf')]
    
    if speedup_metrics:
        # Sort by speedup potential (descending)
        speedup_metrics.sort(key=lambda x: x[1], reverse=True)
        
        print("Speedup Potential Ranking (highest to lowest):")
        for i, (metric_name, speedup) in enumerate(speedup_metrics, 1):
            print(f"  {i}. {metric_name}: {speedup:.2f}x speedup potential")
        
        # Identify the most promising metric for optimization
        best_metric = speedup_metrics[0]
        print(f"\nMost promising for optimization: {best_metric[0]} ({best_metric[1]:.2f}x speedup)")
        
        # Calculate potential improvement from current mean to best case
        if best_metric[0] in results:
            stats = results[best_metric[0]]
            if 'mean_to_min_ratio' in stats:
                improvement = stats['mean_to_min_ratio'] - 1
                print(f"  - Moving from mean to best case: {improvement:.1%} improvement")
    
    # Example 7: Speedup vs variability analysis
    print("\nExample 7: Speedup vs variability analysis")
    print("=" * 60)
    
    for metric_name, stats in results.items():
        if 'max_speedup' in stats and 'cv' in stats:
            speedup = stats['max_speedup']
            cv = stats['cv']
            
            print(f"{metric_name}:")
            print(f"  - Speedup potential: {speedup:.2f}x")
            print(f"  - Variability (CV): {cv:.3f}")
            
            # Categorize based on speedup and variability
            if speedup >= 2 and cv >= 0.3:
                print(f"  - Category: High potential, high variability (good optimization target)")
            elif speedup >= 2 and cv < 0.3:
                print(f"  - Category: High potential, low variability (consistent but improvable)")
            elif speedup < 2 and cv >= 0.3:
                print(f"  - Category: Low potential, high variability (unstable)")
            else:
                print(f"  - Category: Low potential, low variability (stable)")


if __name__ == "__main__":
    main()

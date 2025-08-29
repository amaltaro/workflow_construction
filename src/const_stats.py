#!/usr/bin/env python3
"""
Workflow Construction Statistics Analyzer

This script parses JSON files containing workflow construction metrics and calculates
various statistical measures for key performance parameters.
"""

import json
import statistics
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import argparse


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Load and parse JSON data from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")


def extract_metrics(data: List[Dict[str, Any]], 
                   metric_names: List[str]) -> Dict[str, List[float]]:
    """Extract specified metrics from the data."""
    metrics = {name: [] for name in metric_names}
    
    for entry in data:
        for metric_name in metric_names:
            if metric_name in entry:
                metrics[metric_name].append(entry[metric_name])
    
    return metrics


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate comprehensive statistics for a list of values."""
    if not values:
        return {}
    
    # Basic statistics
    stats = {
        'count': len(values),
        'min': min(values),
        'max': max(values),
        'mean': statistics.mean(values),
        'median': statistics.median(values)
    }
    
    # Standard deviation
    if len(values) > 1:
        stats['std_dev'] = statistics.stdev(values)
        stats['variance'] = statistics.variance(values)
    else:
        stats['std_dev'] = 0.0
        stats['variance'] = 0.0
    
    # Percentiles
    if len(values) > 0:
        stats['q1'] = np.percentile(values, 25)
        stats['q3'] = np.percentile(values, 75)
        stats['iqr'] = stats['q3'] - stats['q1']
    
    # Range
    stats['range'] = stats['max'] - stats['min']
    
    # Coefficient of variation (relative standard deviation)
    if stats['mean'] != 0:
        stats['cv'] = stats['std_dev'] / abs(stats['mean'])
    else:
        stats['cv'] = 0.0
    
    # Speedup calculations
    if stats['min'] > 0:  # Avoid division by zero
        stats['max_speedup'] = stats['max'] / stats['min']
        stats['min_speedup'] = stats['min'] / stats['max']  # Inverse speedup
        stats['speedup_factor'] = stats['max_speedup']  # Alias for clarity
    else:
        stats['max_speedup'] = float('inf') if stats['max'] > 0 else 0.0
        stats['min_speedup'] = 0.0
        stats['speedup_factor'] = stats['max_speedup']
    
    # Additional speedup insights
    if stats['mean'] > 0:
        stats['mean_to_min_ratio'] = stats['mean'] / stats['min']
        stats['max_to_mean_ratio'] = stats['max'] / stats['mean']
    
    return stats


def generate_values_summary_table(results: Dict[str, Dict[str, float]], metrics: Dict[str, List[float]]) -> str:
    """Generate a summary table showing all values for each metric."""
    if not results:
        return "No data available for values summary"
    
    lines = ["\nALL VALUES SUMMARY TABLE"]
    lines.append("=" * 120)
    
    # Create header
    header = f"{'Metric':<35}"
    for i in range(10):  # Show first 10 values
        header += f"{'V' + str(i+1):>12}"
    header += f"{'...':>8}{'Total':>8}"
    lines.append(header)
    lines.append("-" * 120)
    
    # Create data rows
    for metric_name, stats in results.items():
        if metric_name in metrics:
            values = sorted(metrics[metric_name])
            
            # Format the row
            row = f"{metric_name:<35}"
            
            # Show first 10 values
            for i in range(10):
                if i < len(values):
                    # Determine appropriate precision
                    if stats['max'] >= 1000:
                        precision = 2
                    elif stats['max'] >= 100:
                        precision = 3
                    elif stats['max'] >= 10:
                        precision = 4
                    else:
                        precision = 6
                    row += f"{values[i]:>12.{precision}f}"
                else:
                    row += f"{'':>12}"
            
            # Add ellipsis and total count
            if len(values) > 10:
                row += f"{'...':>8}"
            else:
                row += f"{'':>8}"
            
            row += f"{len(values):>8}"
            lines.append(row)
    
    lines.append("-" * 120)
    lines.append("Note: V1, V2, etc. represent values in ascending order")
    
    return "\n".join(lines)


def generate_speedup_analysis_table(results: Dict[str, Dict[str, float]]) -> str:
    """Generate a dedicated speedup analysis table."""
    if not results:
        return "No data available for speedup analysis"
    
    lines = ["\nSPEEDUP ANALYSIS TABLE"]
    lines.append("=" * 80)
    
    # Create header
    header = f"{'Metric':<35}{'Max Speedup':>15}{'Mean/Min':>12}{'Max/Mean':>12}{'Insight':>20}"
    lines.append(header)
    lines.append("-" * 80)
    
    # Create data rows
    for metric_name, stats in results.items():
        if stats and 'max_speedup' in stats:
            # Calculate speedup insights
            max_speedup = stats['max_speedup']
            mean_to_min = stats.get('mean_to_min_ratio', 0)
            max_to_mean = stats.get('max_to_mean_ratio', 0)
            
            # Determine insight based on speedup values
            if max_speedup == float('inf'):
                insight = "Infinite potential"
            elif max_speedup >= 10:
                insight = "Very high potential"
            elif max_speedup >= 5:
                insight = "High potential"
            elif max_speedup >= 2:
                insight = "Moderate potential"
            elif max_speedup >= 1.5:
                insight = "Low potential"
            else:
                insight = "Minimal potential"
            
            # Format the row
            if max_speedup == float('inf'):
                speedup_str = "∞"
            else:
                speedup_str = f"{max_speedup:.2f}x"
            
            row = f"{metric_name:<35}{speedup_str:>15}{mean_to_min:>12.2f}x{max_to_mean:>12.2f}x{insight:>20}"
            lines.append(row)
    
    lines.append("-" * 80)
    return "\n".join(lines)


def generate_summary_table(results: Dict[str, Dict[str, float]]) -> str:
    """Generate a summary table comparing all metrics."""
    if not results:
        return "No data available for summary table"
    
    # Define the metrics to show in the summary table
    summary_metrics = ['count', 'min', 'max', 'mean', 'median', 'std_dev', 'cv', 'max_speedup']
    
    # Create header
    lines = ["\nSUMMARY COMPARISON TABLE"]
    lines.append("=" * 100)
    
    # Create column headers
    header = f"{'Metric':<35}"
    for metric in summary_metrics:
        if metric == 'max_speedup':
            header += f"{'SPEEDUP':>12}"
        else:
            header += f"{metric.upper():>12}"
    lines.append(header)
    lines.append("-" * 100)
    
    # Create data rows
    for metric_name, stats in results.items():
        if stats:
            row = f"{metric_name:<35}"
            for metric in summary_metrics:
                if metric in stats:
                    if metric == 'count':
                        row += f"{stats[metric]:>12.0f}"
                    elif metric in ['cv', 'max_speedup']:
                        if stats[metric] == float('inf'):
                            row += f"{'∞':>12}"
                        else:
                            row += f"{stats[metric]:>12.3f}"
                    else:
                        row += f"{stats[metric]:>12.6f}"
                else:
                    row += f"{'N/A':>12}"
            lines.append(row)
    
    lines.append("-" * 100)
    return "\n".join(lines)


def format_statistics(stats: Dict[str, float], metric_name: str, values: List[float] = None) -> str:
    """Format statistics for display."""
    if not stats:
        return f"No data available for {metric_name}"
    
    lines = [f"\n{metric_name.upper()} Statistics:"]
    lines.append("=" * (len(metric_name) + 11))
    
    # Basic statistics
    lines.append(f"Count: {stats['count']}")
    lines.append(f"Min: {stats['min']:.6f}")
    lines.append(f"Max: {stats['max']:.6f}")
    lines.append(f"Range: {stats['range']:.6f}")
    lines.append(f"Mean: {stats['mean']:.6f}")
    lines.append(f"Median: {stats['median']:.6f}")
    
    # Variability measures
    lines.append(f"Standard Deviation: {stats['std_dev']:.6f}")
    lines.append(f"Variance: {stats['variance']:.6f}")
    lines.append(f"Coefficient of Variation: {stats['cv']:.6f}")
    
    # Quartiles
    lines.append(f"Q1 (25th percentile): {stats['q1']:.6f}")
    lines.append(f"Q3 (75th percentile): {stats['q3']:.6f}")
    lines.append(f"Interquartile Range: {stats['iqr']:.6f}")
    
    # Speedup metrics
    lines.append("")
    lines.append("Speedup Analysis:")
    lines.append("-" * 20)
    
    if 'max_speedup' in stats:
        if stats['max_speedup'] == float('inf'):
            lines.append(f"Maximum Speedup: ∞ (infinite)")
        else:
            lines.append(f"Maximum Speedup: {stats['max_speedup']:.3f}x")
            lines.append(f"Speedup Factor: {stats['speedup_factor']:.3f}x")
        
        if 'mean_to_min_ratio' in stats:
            lines.append(f"Mean/Min Ratio: {stats['mean_to_min_ratio']:.3f}x")
            lines.append(f"Max/Mean Ratio: {stats['max_to_mean_ratio']:.3f}x")
    
    # Values list (if provided)
    if values:
        lines.append("")
        lines.append("All Values (Ascending Order):")
        lines.append("-" * 30)
        
        # Format values with consistent precision
        if stats['max'] >= 1000:
            precision = 2
        elif stats['max'] >= 100:
            precision = 3
        elif stats['max'] >= 10:
            precision = 4
        else:
            precision = 6
        
        # Group values in rows for better readability
        sorted_values = sorted(values)
        values_per_line = 6
        
        for i in range(0, len(sorted_values), values_per_line):
            batch = sorted_values[i:i + values_per_line]
            formatted_batch = [f"{v:.{precision}f}" for v in batch]
            lines.append("  " + "  ".join(f"{v:>12}" for v in formatted_batch))
        
        lines.append(f"\nTotal: {len(sorted_values)} values")
    
    return "\n".join(lines)


def analyze_workflow_metrics(file_path: str, 
                           metric_names: List[str] = None) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[float]]]:
    """Main function to analyze workflow metrics."""
    if metric_names is None:
        metric_names = [
            'write_remote_per_event_mb',
            'event_throughput',
            'memory_per_event_mb',
            'network_transfer_per_event_mb'
        ]
    
    # Load data
    print(f"Loading data from: {file_path}")
    data = load_json_data(file_path)
    print(f"Loaded {len(data)} workflow construction entries")
    
    # Extract metrics
    metrics = extract_metrics(data, metric_names)
    
    # Calculate statistics for each metric
    results = {}
    for metric_name, values in metrics.items():
        if values:
            print(f"\nProcessing {metric_name}: {len(values)} values")
            results[metric_name] = calculate_statistics(values)
        else:
            print(f"\nWarning: No data found for {metric_name}")
            results[metric_name] = {}
    
    return results, metrics


def save_results_to_file(results: Dict[str, Dict[str, float]], 
                        output_file: str):
    """Save results to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Analyze workflow construction metrics from JSON files"
    )
    parser.add_argument(
        'input_file',
        help='Path to the JSON file containing workflow construction data'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file path for results (default: input_file_stats.json)'
    )
    parser.add_argument(
        '-m', '--metrics',
        nargs='+',
        help='Specific metrics to analyze (default: all four main metrics)'
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Show only summary table and speedup analysis, not detailed statistics'
    )
    parser.add_argument(
        '--speedup-only',
        action='store_true',
        help='Show only speedup analysis table'
    )
    parser.add_argument(
        '--values-only',
        action='store_true',
        help='Show only values summary table'
    )
    
    args = parser.parse_args()
    
    # Set default output file if not specified
    if not args.output:
        input_path = Path(args.input_file)
        args.output = str(input_path).replace('.json', '_stats.json')
    
    try:
        # Analyze metrics
        results, metrics = analyze_workflow_metrics(args.input_file, args.metrics)
        
        # Display results
        print("\n" + "="*60)
        print("WORKFLOW CONSTRUCTION METRICS ANALYSIS")
        print("="*60)
        
        if args.values_only:
            # Show only values summary table
            print(generate_values_summary_table(results, metrics))
        elif args.speedup_only:
            # Show only speedup analysis table
            print(generate_speedup_analysis_table(results))
        elif args.summary_only:
            # Show only summary table, speedup analysis, and values summary
            print(generate_summary_table(results))
            print(generate_speedup_analysis_table(results))
            print(generate_values_summary_table(results, metrics))
        else:
            # Show detailed statistics for each metric
            for metric_name, stats in results.items():
                # Get the original values for this metric
                metric_values = metrics.get(metric_name, [])
                print(format_statistics(stats, metric_name, metric_values))
            
            # Show summary table, speedup analysis, and values summary at the end
            print(generate_summary_table(results))
            print(generate_speedup_analysis_table(results))
            print(generate_values_summary_table(results, metrics))
        
        # Save results
        save_results_to_file(results, args.output)
        
        print(f"\nAnalysis complete! Processed {len(results)} metrics.")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

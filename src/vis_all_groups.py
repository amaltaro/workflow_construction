import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict
import json
import os
from pathlib import Path

def plot_resource_utilization(groups: List[Dict], output_dir: str = "plots"):
    """Plot resource utilization metrics for all groups"""
    print(f"\nPlotting resource utilization for {len(groups)} groups")
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract metrics
    group_ids = []
    cpu_efficiencies = []
    memory_occupancies = []
    resource_utilizations = []
    group_sizes = []
    
    for group in groups:
        group_ids.append(group["group_id"])
        cpu_efficiencies.append(group["resource_metrics"]["cpu"]["efficiency"])
        memory_occupancies.append(group["resource_metrics"]["memory"]["occupancy"])
        resource_utilizations.append(group["utilization_metrics"]["resource_utilization"])
        group_sizes.append(len(group["task_ids"]))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Resource Utilization Analysis", fontsize=16)
    
    # Plot 1: CPU Efficiency vs Group Size
    sns.scatterplot(x=group_sizes, y=cpu_efficiencies, ax=axes[0, 0])
    axes[0, 0].set_title("CPU Efficiency vs Group Size")
    axes[0, 0].set_xlabel("Number of Tasks in Group")
    axes[0, 0].set_ylabel("CPU Efficiency")
    axes[0, 0].grid(True)
    
    # Plot 2: Memory Occupancy vs Group Size
    sns.scatterplot(x=group_sizes, y=memory_occupancies, ax=axes[0, 1])
    axes[0, 1].set_title("Memory Occupancy vs Group Size")
    axes[0, 1].set_xlabel("Number of Tasks in Group")
    axes[0, 1].set_ylabel("Memory Occupancy")
    axes[0, 1].grid(True)
    
    # Plot 3: Overall Resource Utilization vs Group Size
    sns.scatterplot(x=group_sizes, y=resource_utilizations, ax=axes[1, 0])
    axes[1, 0].set_title("Overall Resource Utilization vs Group Size")
    axes[1, 0].set_xlabel("Number of Tasks in Group")
    axes[1, 0].set_ylabel("Resource Utilization")
    axes[1, 0].grid(True)
    
    # Plot 4: CPU Efficiency vs Memory Occupancy
    sns.scatterplot(x=cpu_efficiencies, y=memory_occupancies, 
                   size=group_sizes, sizes=(50, 200), ax=axes[1, 1])
    axes[1, 1].set_title("CPU Efficiency vs Memory Occupancy")
    axes[1, 1].set_xlabel("CPU Efficiency")
    axes[1, 1].set_ylabel("Memory Occupancy")
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "resource_utilization.png"))
    plt.close()

def plot_throughput_analysis(groups: List[Dict], output_dir: str = "plots"):
    """Plot throughput and I/O metrics for all groups"""
    print(f"Plotting throughput analysis for {len(groups)} groups")
    # Extract metrics
    group_sizes = []
    total_throughputs = []
    max_throughputs = []
    min_throughputs = []
    total_output_sizes = []
    max_output_sizes = []
    
    for group in groups:
        group_sizes.append(len(group["task_ids"]))
        total_throughputs.append(group["resource_metrics"]["throughput"]["total_eps"])
        max_throughputs.append(group["resource_metrics"]["throughput"]["max_eps"])
        min_throughputs.append(group["resource_metrics"]["throughput"]["min_eps"])
        total_output_sizes.append(group["resource_metrics"]["io"]["total_output_mb"])
        max_output_sizes.append(group["resource_metrics"]["io"]["max_output_mb"])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Throughput and I/O Analysis", fontsize=16)
    
    # Plot 1: Throughput vs Group Size
    sns.scatterplot(x=group_sizes, y=total_throughputs, ax=axes[0, 0])
    axes[0, 0].set_title("Total Throughput vs Group Size")
    axes[0, 0].set_xlabel("Number of Tasks in Group")
    axes[0, 0].set_ylabel("Total Events per Second")
    axes[0, 0].grid(True)
    
    # Plot 2: Throughput Range vs Group Size
    axes[0, 1].scatter(group_sizes, max_throughputs, label="Max Throughput", alpha=0.6)
    axes[0, 1].scatter(group_sizes, min_throughputs, label="Min Throughput", alpha=0.6)
    axes[0, 1].set_title("Throughput Range vs Group Size")
    axes[0, 1].set_xlabel("Number of Tasks in Group")
    axes[0, 1].set_ylabel("Events per Second")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: Output Size vs Group Size
    sns.scatterplot(x=group_sizes, y=total_output_sizes, ax=axes[1, 0])
    axes[1, 0].set_title("Total Output Size vs Group Size")
    axes[1, 0].set_xlabel("Number of Tasks in Group")
    axes[1, 0].set_ylabel("Total Output Size (MB)")
    axes[1, 0].grid(True)
    
    # Plot 4: Throughput vs Output Size
    sns.scatterplot(x=total_throughputs, y=total_output_sizes, 
                   size=group_sizes, sizes=(50, 200), ax=axes[1, 1])
    axes[1, 1].set_title("Throughput vs Output Size")
    axes[1, 1].set_xlabel("Total Events per Second")
    axes[1, 1].set_ylabel("Total Output Size (MB)")
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "throughput_analysis.png"))
    plt.close()

def plot_dependency_analysis(groups: List[Dict], output_dir: str = "plots"):
    """Plot dependency-related metrics for all groups"""
    print(f"Plotting dependency analysis for {len(groups)} groups")
    # Extract metrics
    group_sizes = []
    dependency_paths_counts = []
    avg_path_lengths = []
    
    for group in groups:
        group_sizes.append(len(group["task_ids"]))
        paths = group["dependency_paths"]
        dependency_paths_counts.append(len(paths))
        if paths:
            avg_path_lengths.append(sum(len(path) for path in paths) / len(paths))
        else:
            avg_path_lengths.append(0)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Dependency Analysis", fontsize=16)
    
    # Plot 1: Number of Dependency Paths vs Group Size
    sns.scatterplot(x=group_sizes, y=dependency_paths_counts, ax=axes[0])
    axes[0].set_title("Number of Dependency Paths vs Group Size")
    axes[0].set_xlabel("Number of Tasks in Group")
    axes[0].set_ylabel("Number of Dependency Paths")
    axes[0].grid(True)
    
    # Plot 2: Average Path Length vs Group Size
    sns.scatterplot(x=group_sizes, y=avg_path_lengths, ax=axes[1])
    axes[1].set_title("Average Path Length vs Group Size")
    axes[1].set_xlabel("Number of Tasks in Group")
    axes[1].set_ylabel("Average Path Length")
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dependency_analysis.png"))
    plt.close()

def plot_comparison_heatmap(groups: List[Dict], output_dir: str = "plots"):
    """Create a heatmap comparing different metrics across groups"""
    print(f"Plotting comparison heatmap for {len(groups)} groups")
    # Extract metrics
    metrics = {
        "Group Size": [],
        "CPU Efficiency": [],
        "Memory Occupancy": [],
        "Resource Utilization": [],
        "Total Throughput": [],
        "Total Output Size": [],
        "Dependency Paths": []
    }
    
    for group in groups:
        metrics["Group Size"].append(len(group["task_ids"]))
        metrics["CPU Efficiency"].append(group["resource_metrics"]["cpu"]["efficiency"])
        metrics["Memory Occupancy"].append(group["resource_metrics"]["memory"]["occupancy"])
        metrics["Resource Utilization"].append(group["utilization_metrics"]["resource_utilization"])
        metrics["Total Throughput"].append(group["resource_metrics"]["throughput"]["total_eps"])
        metrics["Total Output Size"].append(group["resource_metrics"]["io"]["total_output_mb"])
        metrics["Dependency Paths"].append(len(group["dependency_paths"]))
    
    # Create correlation matrix
    df = pd.DataFrame(metrics)
    corr_matrix = df.corr()
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
    plt.title("Metric Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metric_correlation.png"))
    plt.close()

def visualize_groups(groups: List[Dict], output_dir: str = "plots"):
    """Generate all visualizations for the task groups"""
    # Create plots
    plot_resource_utilization(groups, output_dir)
    plot_throughput_analysis(groups, output_dir)
    plot_dependency_analysis(groups, output_dir)
    plot_comparison_heatmap(groups, output_dir)
    
    # Save raw data for further analysis
    print(f"Saving raw data for {len(groups)} groups")
    with open(os.path.join(output_dir, "group_metrics.json"), "w") as f:
        json.dump(groups, f, indent=2)

if __name__ == "__main__":
    import pandas as pd
    from find_all_groups import create_workflow_from_json
    
    # Example usage
    template_name = "1group_perfect.json"
    print(f"Parsing workflow data for template {template_name}")
    with open(f"tests/sequential/{template_name}") as f:
        workflow_data = json.load(f)
    
    groups, tasks = create_workflow_from_json(workflow_data)
    visualize_groups(groups) 
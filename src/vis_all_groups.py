import matplotlib.pyplot as plt
import seaborn as sns
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
    cpu_utilization_ratios = []
    memory_occupancies = []
    resource_utilizations = []
    group_sizes = []
    events_per_job = []

    for group in groups:
        group_ids.append(group["group_id"])
        cpu_utilization_ratios.append(group["resource_metrics"]["cpu"]["utilization_ratio"])
        memory_occupancies.append(group["resource_metrics"]["memory"]["occupancy"])
        resource_utilizations.append(group["utilization_metrics"]["resource_utilization"])
        group_sizes.append(len(group["task_ids"]))
        events_per_job.append(group["events_per_job"])

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Resource Utilization Analysis", fontsize=16)

    # Plot 1: CPU Utilization Ratio vs Group Size
    sns.scatterplot(x=group_sizes, y=cpu_utilization_ratios, ax=axes[0, 0])
    axes[0, 0].set_title("CPU Utilization Ratio vs Group Size")
    axes[0, 0].set_xlabel("Number of Tasks in Group")
    axes[0, 0].set_ylabel("CPU Utilization Ratio")
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

    # Plot 4: CPU Utilization Ratio vs Memory Occupancy
    sns.scatterplot(x=cpu_utilization_ratios, y=memory_occupancies, 
                   size=events_per_job, sizes=(50, 200), ax=axes[1, 1])
    axes[1, 1].set_title("CPU Utilization Ratio vs Memory Occupancy\n(size indicates events per job)")
    axes[1, 1].set_xlabel("CPU Utilization Ratio")
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
    input_data_sizes = []
    output_data_sizes = []
    stored_data_sizes = []

    for group in groups:
        group_sizes.append(len(group["task_ids"]))
        total_throughputs.append(group["resource_metrics"]["throughput"]["total_eps"])
        max_throughputs.append(group["resource_metrics"]["throughput"]["max_eps"])
        min_throughputs.append(group["resource_metrics"]["throughput"]["min_eps"])
        input_data_sizes.append(group["resource_metrics"]["io"]["input_data_mb"])
        output_data_sizes.append(group["resource_metrics"]["io"]["output_data_mb"])
        stored_data_sizes.append(group["resource_metrics"]["io"]["stored_data_mb"])

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

    # Plot 3: Data Size vs Group Size
    axes[1, 0].scatter(group_sizes, input_data_sizes, label="Input Data", alpha=0.6)
    axes[1, 0].scatter(group_sizes, output_data_sizes, label="Output Data", alpha=0.6)
    axes[1, 0].scatter(group_sizes, stored_data_sizes, label="Stored Data", alpha=0.6)
    axes[1, 0].set_title("Data Size vs Group Size")
    axes[1, 0].set_xlabel("Number of Tasks in Group")
    axes[1, 0].set_ylabel("Data Size (MB)")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot 4: Throughput vs Data Size
    sns.scatterplot(x=total_throughputs, y=stored_data_sizes,
                   size=group_sizes, sizes=(50, 200), ax=axes[1, 1])
    axes[1, 1].set_title("Throughput vs Stored Data Size")
    axes[1, 1].set_xlabel("Total Events per Second")
    axes[1, 1].set_ylabel("Stored Data Size (MB)")
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
        "CPU Utilization Ratio": [],
        "Memory Occupancy": [],
        "Resource Utilization": [],
        "Total Throughput": [],
        "Input Data Size": [],
        "Output Data Size": [],
        "Stored Data Size": [],
        "Events per Job": [],
        "Dependency Paths": []
    }

    for group in groups:
        metrics["Group Size"].append(len(group["task_ids"]))
        metrics["CPU Utilization Ratio"].append(group["resource_metrics"]["cpu"]["utilization_ratio"])
        metrics["Memory Occupancy"].append(group["resource_metrics"]["memory"]["occupancy"])
        metrics["Resource Utilization"].append(group["utilization_metrics"]["resource_utilization"])
        metrics["Total Throughput"].append(group["resource_metrics"]["throughput"]["total_eps"])
        metrics["Input Data Size"].append(group["resource_metrics"]["io"]["input_data_mb"])
        metrics["Output Data Size"].append(group["resource_metrics"]["io"]["output_data_mb"])
        metrics["Stored Data Size"].append(group["resource_metrics"]["io"]["stored_data_mb"])
        metrics["Dependency Paths"].append(len(group["dependency_paths"]))
        metrics["Events per Job"].append(group["events_per_job"])

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

def plot_workflow_constructions(construction_metrics: List[Dict], output_dir: str = "plots"):
    """Plot metrics for different workflow constructions"""
    print(f"Plotting workflow construction analysis for {len(construction_metrics)} constructions")

    # Extract metrics
    num_groups = []
    event_throughputs = []
    total_events = []
    total_cpu_times = []
    group_combinations = []

    for metrics in construction_metrics:
        num_groups.append(metrics["num_groups"])
        event_throughputs.append(metrics["event_throughput"])
        total_events.append(metrics["total_events"])
        total_cpu_times.append(metrics["total_cpu_time"])
        group_combinations.append(" + ".join(metrics["groups"]))

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Workflow Construction Analysis", fontsize=16)

    # Plot 1: Event Throughput vs Number of Groups
    sns.scatterplot(x=num_groups, y=event_throughputs, ax=axes[0, 0])
    axes[0, 0].set_title("Event Throughput vs Number of Groups")
    axes[0, 0].set_xlabel("Number of Groups")
    axes[0, 0].set_ylabel("Events per Second")
    axes[0, 0].grid(True)

    # Plot 2: Total Events vs Total CPU Time
    sns.scatterplot(x=total_cpu_times, y=total_events,
                   size=num_groups, sizes=(50, 200), ax=axes[0, 1])
    axes[0, 1].set_title("Total Events vs Total CPU Time\n(size indicates number of groups)")
    axes[0, 1].set_xlabel("Total CPU Time (seconds)")
    axes[0, 1].set_ylabel("Total Events")
    axes[0, 1].grid(True)

    # Plot 3: Event Throughput Distribution
    sns.histplot(event_throughputs, bins=20, ax=axes[1, 0])
    axes[1, 0].set_title("Event Throughput Distribution")
    axes[1, 0].set_xlabel("Events per Second")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].grid(True)

    # Plot 4: Group Size Distribution
    sns.histplot(num_groups, bins=range(min(num_groups), max(num_groups) + 2), ax=axes[1, 1])
    axes[1, 1].set_title("Number of Groups Distribution")
    axes[1, 1].set_xlabel("Number of Groups")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "workflow_constructions.png"))
    plt.close()

    # Create a table of the top 5 constructions by throughput
    top_constructions = sorted(construction_metrics,
                             key=lambda x: x["event_throughput"],
                             reverse=True)[:5]

    # Save top constructions to a text file
    with open(os.path.join(output_dir, "top_constructions.txt"), "w") as f:
        f.write("Top 5 Workflow Constructions by Event Throughput:\n\n")
        for i, metrics in enumerate(top_constructions, 1):
            f.write(f"Construction {i}:\n")
            f.write(f"  Groups: {metrics['groups']}\n")
            f.write(f"  Number of Groups: {metrics['num_groups']}\n")
            f.write(f"  Total Events: {metrics['total_events']}\n")
            f.write(f"  Total CPU Time: {metrics['total_cpu_time']:.2f} seconds\n")
            f.write(f"  Event Throughput: {metrics['event_throughput']:.2f} events/second\n\n")

def visualize_groups(groups: List[Dict], construction_metrics: List[Dict], output_dir: str = "plots"):
    """Generate all visualizations for the task groups and workflow constructions"""
    # Create plots for individual groups
    plot_resource_utilization(groups, output_dir)
    plot_throughput_analysis(groups, output_dir)
    plot_dependency_analysis(groups, output_dir)
    plot_comparison_heatmap(groups, output_dir)

    # Create plots for workflow constructions
    plot_workflow_constructions(construction_metrics, output_dir)

    # Save raw data for further analysis
    print(f"Saving raw data for {len(groups)} groups and {len(construction_metrics)} constructions")
    with open(os.path.join(output_dir, "group_metrics.json"), "w") as f:
        json.dump(groups, f, indent=2)
    with open(os.path.join(output_dir, "construction_metrics.json"), "w") as f:
        json.dump(construction_metrics, f, indent=2)

if __name__ == "__main__":
    import pandas as pd
    from find_all_groups import create_workflow_from_json

    # Example usage
    template_name = "1group_perfect.json"
    print(f"Parsing workflow data for template {template_name}")
    with open(f"tests/sequential/{template_name}") as f:
        workflow_data = json.load(f)

    groups, tasks, construction_metrics = create_workflow_from_json(workflow_data)
    visualize_groups(groups, construction_metrics)

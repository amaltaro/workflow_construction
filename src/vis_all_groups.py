import argparse
import json
import os
from typing import List, Dict
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from vis_constructions import plot_workflow_topology
from find_all_groups import create_workflow_from_json
import networkx as nx


def plot_resource_utilization(groups: List[Dict], output_dir: str = "plots"):
    """Plot resource utilization metrics for all groups"""
    print(f"Plotting resource utilization for {len(groups)} groups")
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

    # Plot 4: CPU Utilization Ratio vs Memory Occupancy with color-coded events per job
    # Create discrete color bins for events per job
    events_array = np.array(events_per_job)
    unique_events = np.unique(events_array)

    # Create a discrete colormap
    n_events = len(unique_events)
    cmap = plt.colormaps['viridis'].resampled(n_events)

    # Create scatter plot with discrete colors
    scatter = axes[1, 1].scatter(cpu_utilization_ratios, memory_occupancies,
                                c=events_array, cmap=cmap, s=100, alpha=0.7)

    # Add colorbar with discrete ticks
    cbar = plt.colorbar(scatter, ax=axes[1, 1], label="Events per Job")
    cbar.set_ticks(unique_events)
    cbar.set_ticklabels([f"{int(x)}" for x in unique_events])

    axes[1, 1].set_title("CPU Utilization Ratio vs Memory Occupancy\n(color indicates events per job)")
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
    fig.suptitle("Group Throughput and I/O Analysis", fontsize=16)

    # Plot 1: Throughput vs Group Size
    sns.scatterplot(x=group_sizes, y=total_throughputs, ax=axes[0, 0])
    axes[0, 0].set_title("Total Throughput vs Group Size")
    axes[0, 0].set_xlabel("Number of Tasks in Group")
    axes[0, 0].set_ylabel("Total Events per Second")
    axes[0, 0].grid(True)

    # Plot 2: Throughput Range vs Group Size
    # Aggregate data by group size to get actual min/max across all groups of same size
    group_size_data = defaultdict(lambda: {'max_throughputs': [], 'min_throughputs': []})

    for i, size in enumerate(group_sizes):
        group_size_data[size]['max_throughputs'].append(max_throughputs[i])
        group_size_data[size]['min_throughputs'].append(min_throughputs[i])

    # Calculate actual min and max for each group size
    aggregated_sizes = []
    aggregated_max_throughputs = []
    aggregated_min_throughputs = []

    for size in sorted(group_size_data.keys()):
        aggregated_sizes.append(size)
        aggregated_max_throughputs.append(max(group_size_data[size]['max_throughputs']))
        aggregated_min_throughputs.append(min(group_size_data[size]['min_throughputs']))

    axes[0, 1].scatter(aggregated_sizes, aggregated_max_throughputs, label="Max Throughput", alpha=0.6, s=100)
    axes[0, 1].scatter(aggregated_sizes, aggregated_min_throughputs, label="Min Throughput", alpha=0.6, s=100)
    axes[0, 1].set_title("Throughput Range vs Group Size\n(Min/Max across all groups of same size)")
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
    # Create discrete color bins for group sizes
    group_sizes_array = np.array(group_sizes)
    unique_group_sizes = np.unique(group_sizes_array)

    # Create a discrete colormap
    n_sizes = len(unique_group_sizes)
    cmap = plt.colormaps['plasma'].resampled(n_sizes)

    # Create scatter plot with discrete colors
    scatter = axes[1, 1].scatter(total_throughputs, stored_data_sizes,
                                c=group_sizes_array, cmap=cmap, s=100, alpha=0.7)

    # Add colorbar with discrete ticks
    cbar = plt.colorbar(scatter, ax=axes[1, 1], label="Group Size")
    cbar.set_ticks(unique_group_sizes)
    cbar.set_ticklabels([f"{int(x)}" for x in unique_group_sizes])

    axes[1, 1].set_title("Throughput vs Stored Data Size\n(color indicates group size)")
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

def plot_storage_efficiency(construction_metrics: List[Dict], output_dir: str = "plots"):
    """Plot storage efficiency metrics for workflow constructions"""
    print(f"Plotting storage efficiency analysis for {len(construction_metrics)} constructions")

    # Extract metrics
    num_groups = []
    event_throughputs = []
    stored_data_per_event = []
    total_stored_data = []
    total_events = []

    for metrics in construction_metrics:
        num_groups.append(metrics["num_groups"])
        event_throughputs.append(metrics["event_throughput"])
        stored_data_per_event.append(metrics["stored_data_per_event_mb"])
        total_stored_data.append(metrics["total_stored_data_mb"])
        total_events.append(metrics["total_events"])

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Storage Efficiency Analysis for Workflow Constructions", fontsize=16)

    # Plot 1: Stored Data per Event vs Number of Groups
    sns.scatterplot(x=num_groups, y=stored_data_per_event, ax=axes[0, 0])
    axes[0, 0].set_title("Storage Efficiency vs Number of Groups")
    axes[0, 0].set_xlabel("Number of Groups")
    axes[0, 0].set_ylabel("Stored Data per Event (MB)")
    axes[0, 0].grid(True)

    # Plot 2: Stored Data per Event vs Event Throughput
    sns.scatterplot(x=event_throughputs, y=stored_data_per_event, ax=axes[0, 1])
    axes[0, 1].set_title("Storage Efficiency vs Event Throughput")
    axes[0, 1].set_xlabel("Event Throughput (events/second)")
    axes[0, 1].set_ylabel("Stored Data per Event (MB)")
    axes[0, 1].grid(True)

    # Plot 3: Total Stored Data vs Total Events
    # Create discrete color bins for number of groups
    num_groups_array = np.array(num_groups)
    unique_num_groups = np.unique(num_groups_array)

    # Create a discrete colormap
    n_groups = len(unique_num_groups)
    cmap = plt.colormaps['cool'].resampled(n_groups)

    # Create scatter plot with discrete colors
    scatter = axes[1, 0].scatter(total_events, total_stored_data,
                                c=num_groups_array, cmap=cmap, s=100, alpha=0.7)

    # Add colorbar with discrete ticks
    cbar = plt.colorbar(scatter, ax=axes[1, 0], label="Number of Groups")
    cbar.set_ticks(unique_num_groups)
    cbar.set_ticklabels([f"{int(x)}" for x in unique_num_groups])

    axes[1, 0].set_title("Total Stored Data vs Total Events\n(color indicates number of groups)")
    axes[1, 0].set_xlabel("Total Events")
    axes[1, 0].set_ylabel("Total Stored Data (MB)")
    axes[1, 0].grid(True)

    # Plot 4: Stored Data per Event Distribution
    sns.histplot(stored_data_per_event, bins=20, ax=axes[1, 1])
    axes[1, 1].set_title("Stored Data per Event Distribution")
    axes[1, 1].set_xlabel("Stored Data per Event (MB)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "storage_efficiency.png"))
    plt.close()

def plot_workflow_constructions(construction_metrics: List[Dict], output_dir: str = "plots"):
    """Plot metrics for different workflow constructions"""
    print(f"Plotting workflow construction analysis for {len(construction_metrics)} constructions")

    # Extract metrics
    num_groups = []
    event_throughputs = []
    total_events = []
    total_cpu_times = []
    stored_data_per_event = []
    group_combinations = []

    for metrics in construction_metrics:
        num_groups.append(metrics["num_groups"])
        event_throughputs.append(metrics["event_throughput"])
        total_events.append(metrics["total_events"])
        total_cpu_times.append(metrics["total_cpu_time"])
        stored_data_per_event.append(metrics["stored_data_per_event_mb"])
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
    # Create discrete color bins for stored data per event
    stored_data_array = np.array(stored_data_per_event)
    unique_stored_data = np.unique(stored_data_array)

    # Create a discrete colormap
    n_stored = len(unique_stored_data)
    cmap = plt.colormaps['magma'].resampled(n_stored)

    # Create scatter plot with discrete colors
    scatter = axes[0, 1].scatter(total_cpu_times, total_events,
                                c=stored_data_array, cmap=cmap, s=100, alpha=0.7)

    # Add colorbar with discrete ticks
    cbar = plt.colorbar(scatter, ax=axes[0, 1], label="Stored Data per Event (MB)")
    cbar.set_ticks(unique_stored_data)
    cbar.set_ticklabels([f"{x:.3f}" for x in unique_stored_data])

    axes[0, 1].set_title("Total Events vs Total CPU Time\n(color indicates stored data per event)")
    axes[0, 1].set_xlabel("Total CPU Time (seconds)")
    axes[0, 1].set_ylabel("Total Events")
    axes[0, 1].grid(True)

    # Plot 3: Event Throughput Distribution
    sns.histplot(event_throughputs, bins=20, ax=axes[1, 0])
    axes[1, 0].set_title("Event Throughput Distribution")
    axes[1, 0].set_xlabel("Events per Second")
    axes[1, 0].set_ylabel("Number of Workflow Constructions")
    axes[1, 0].grid(True)

    # Plot 4: Stored Data per Event vs Event Throughput
    # Create discrete color bins for number of groups
    num_groups_array = np.array(num_groups)
    unique_num_groups = np.unique(num_groups_array)

    # Create a discrete colormap
    n_groups = len(unique_num_groups)
    cmap = plt.colormaps['spring'].resampled(n_groups)

    # Create scatter plot with discrete colors
    scatter = axes[1, 1].scatter(event_throughputs, stored_data_per_event,
                                c=num_groups_array, cmap=cmap, s=100, alpha=0.7)

    # Add colorbar with discrete ticks
    cbar = plt.colorbar(scatter, ax=axes[1, 1], label="Number of Groups")
    cbar.set_ticks(unique_num_groups)
    cbar.set_ticklabels([f"{int(x)}" for x in unique_num_groups])

    axes[1, 1].set_title("Storage Efficiency vs Event Throughput\n(color indicates number of groups)")
    axes[1, 1].set_xlabel("Event Throughput (events/second)")
    axes[1, 1].set_ylabel("Stored Data per Event (MB)")
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
            f.write(f"  Event Throughput: {metrics['event_throughput']:.4f} events/second\n")
            f.write(f"  Stored Data per Event: {metrics['stored_data_per_event_mb']:.3f} MB/event\n\n")

def plot_group_data_volume_analysis(construction_metrics: List[Dict], output_dir: str = "plots"):
    """Create a dedicated horizontal bar plot for group-level data volume analysis.

    This plot shows the data volumes (input, output, stored) for each group
    across all workflow constructions in a horizontal format for better readability.
    """
    # Calculate total number of groups for sizing
    total_groups = sum(len(metrics["group_details"]) for metrics in construction_metrics)
    print(f"Creating group-level analysis for total groups: {total_groups}, total constructions: {len(construction_metrics)}")

    # Calculate optimal figure height based on number of groups, bar height and spacing between constructions
    bar_height = 0.2
    fig_height = max(3, (total_groups + len(construction_metrics)) * bar_height)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    # Calculate positions for each group in each construction
    group_positions = []
    group_labels = []
    current_pos = 0

    for i, metrics in enumerate(construction_metrics):
        groups_in_construction = len(metrics["group_details"])
        # Create exactly the right number of positions for this construction's groups
        # Each group gets one position, spaced by bar_height
        # np.arange() is not good for floating point numbers
        positions = np.linspace(current_pos, current_pos + (groups_in_construction - 1) * bar_height, groups_in_construction)
        # print(f"Construction {i}: current_pos {current_pos}, groups_in_construction: {groups_in_construction}, positions: {positions}")
        group_positions.extend(positions)
        # Create labels for each group with more descriptive names
        for j, group in enumerate(metrics["group_details"]):
            group_id = group["group_id"]
            tasks_str = ", ".join(group["tasks"])
            group_labels.append(f"Const {i+1} - {group_id} ({tasks_str})")
        # Move to next construction position (add spacing only between constructions)
        current_pos += groups_in_construction * bar_height + bar_height

    left = np.zeros(len(group_positions))
    # Plot input data
    input_data = [group["input_data_mb"] for metrics in construction_metrics
                 for group in metrics["group_details"]]
    ax.barh(group_positions, input_data, bar_height, label='Input Data', left=left, alpha=0.8,
            edgecolor='black', linewidth=0.4)
    left += input_data

    # Plot output data
    output_data = [group["output_data_mb"] for metrics in construction_metrics
                  for group in metrics["group_details"]]
    ax.barh(group_positions, output_data, bar_height, label='Output Data', left=left, alpha=0.8,
            edgecolor='black', linewidth=0.4)
    left += output_data

    # Plot stored data
    stored_data = [group["stored_data_mb"] for metrics in construction_metrics
                  for group in metrics["group_details"]]
    ax.barh(group_positions, stored_data, bar_height, label='Stored Data', left=left, alpha=0.8,
            edgecolor='black', linewidth=0.4)

    for i, (pos, input_val, output_val, stored_val) in enumerate(zip(group_positions, input_data, output_data, stored_data)):
        total = input_val + output_val + stored_val
        ax.text(total + 0.1, pos, f'{total:.1f}', va='center', fontsize=8, alpha=0.8)

    ax.set_ylabel("Workflow Construction and Groups")
    ax.set_xlabel("Total Data Volume (MB)")
    ax.set_title(f"Group-level Data Volume Analysis\n(Data Volumes per Group - {len(construction_metrics)} Constructions)", fontsize=14, fontweight='bold')
    ax.set_yticks(group_positions)
    ax.set_yticklabels(group_labels, fontsize=8)  # Reduced font size for better fit
    ax.legend(loc='upper right', fontsize=10)  # or "best" for dynamic placement
    ax.grid(True, which='major', alpha=0.2)
    ax.set_xlim(left=0)  # Start x-axis at 0

    # Adjust y-axis limits to ensure all groups are visible with proper spacing
    ax.set_ylim(bottom=(-1 * bar_height), top=fig_height)

    # Add some styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adjust layout to ensure all elements are visible
    plt.tight_layout(pad=2.0)  # Increased padding

    # Save with high DPI and tight bbox to ensure all content is captured
    plt.savefig(os.path.join(output_dir, "group_data_volume_analysis.png"),
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def plot_workflow_comparison(construction_metrics: List[Dict], output_dir: str = "plots"):
    """Create a comprehensive comparison of workflow constructions.

    This function creates multiple visualizations to help identify trade-offs
    between different workflow constructions.
    """
    print(f"Creating comprehensive workflow construction comparison for {len(construction_metrics)} constructions")

    # Extract metrics for comparison
    num_groups = []
    event_throughputs = []
    total_cpu_times = []
    stored_data_per_event = []
    total_stored_data = []
    input_data_per_event = []
    output_data_per_event = []
    group_combinations = []

    for metrics in construction_metrics:
        num_groups.append(metrics["num_groups"])
        event_throughputs.append(metrics["event_throughput"])
        total_cpu_times.append(metrics["total_cpu_time"])
        stored_data_per_event.append(metrics["stored_data_per_event_mb"])
        total_stored_data.append(metrics["total_stored_data_mb"])
        input_data_per_event.append(metrics["input_data_per_event_mb"])
        output_data_per_event.append(metrics["output_data_per_event_mb"])
        group_combinations.append(" + ".join(metrics["groups"]))

    # Convert lists to numpy arrays for numerical operations
    num_groups = np.array(num_groups)
    event_throughputs = np.array(event_throughputs)
    total_cpu_times = np.array(total_cpu_times)
    stored_data_per_event = np.array(stored_data_per_event)
    total_stored_data = np.array(total_stored_data)
    input_data_per_event = np.array(input_data_per_event)
    output_data_per_event = np.array(output_data_per_event)

    # Create a figure with multiple subplots - now with fixed, professional proportions
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1, 1])  # Equal height ratios for all rows

    # 1. Group Size Distribution
    ax3 = fig.add_subplot(gs[0, 0])
    group_sizes = []
    for metrics in construction_metrics:
        sizes = [len(group["tasks"]) for group in metrics["group_details"]]
        group_sizes.append(sizes)

    # Create a box plot for group sizes
    ax3.boxplot(group_sizes, tick_labels=[f"Const {i+1}" for i in range(len(construction_metrics))])
    ax3.set_xlabel("Workflow Construction")
    ax3.set_ylabel("Number of Tasks per Group")
    ax3.set_title("Group Size Distribution")
    ax3.set_xticklabels([f"Const {i+1}" for i in range(len(construction_metrics))], rotation=45)
    ax3.grid(True)
    ax3.set_ylim(bottom=0)  # Set y-axis to start at 0

    # 2. Data Flow Analysis (Updated to use per-event metrics)
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(construction_metrics))
    width = 0.25
    ax2.bar(x - width, input_data_per_event, width, label='Input Data/Event')
    ax2.bar(x, output_data_per_event, width, label='Output Data/Event')
    ax2.bar(x + width, stored_data_per_event, width, label='Stored Data/Event')
    ax2.set_xlabel("Workflow Construction")
    ax2.set_ylabel("Data Volume per Event (MB)")
    ax2.set_title("Data Flow Analysis\n(Per-Event Data Volumes)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Const {i+1}" for i in range(len(construction_metrics))], rotation=45)
    ax2.legend()
    ax2.grid(True)

    # 3. Total Data Volume Analysis (Stacked Bar)
    ax10 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(construction_metrics))
    width = 0.6
    bottom = np.zeros(len(construction_metrics))

    # Plot each data type as a layer in the stack
    ax10.bar(x, [m["total_input_data_mb"] for m in construction_metrics], width,
            label='Input Data', bottom=bottom)
    bottom += [m["total_input_data_mb"] for m in construction_metrics]

    ax10.bar(x, [m["total_output_data_mb"] for m in construction_metrics], width,
            label='Output Data', bottom=bottom)
    bottom += [m["total_output_data_mb"] for m in construction_metrics]

    ax10.bar(x, [m["total_stored_data_mb"] for m in construction_metrics], width,
            label='Stored Data', bottom=bottom)

    # Add total value labels on top of each bar
    totals = [m["total_input_data_mb"] + m["total_output_data_mb"] + m["total_stored_data_mb"]
             for m in construction_metrics]
    for i, total in enumerate(totals):
        ax10.text(i, total, f'{total:.1f}', ha='center', va='bottom')

    ax10.set_xlabel("Workflow Construction")
    ax10.set_ylabel("Total Data Volume (MB)")
    ax10.set_title("Total Data Volume Analysis\n(Aggregated Data Volumes for one job of each group)")
    ax10.set_xticks(x)
    ax10.set_xticklabels([f"Const {i+1}" for i in range(len(construction_metrics))], rotation=45)
    ax10.legend()
    ax10.grid(True)

    # 4. Performance vs Storage Efficiency
    ax1 = fig.add_subplot(gs[1, 1])

    # Create a discrete colormap for number of groups
    unique_groups = np.unique(num_groups)
    n_groups = len(unique_groups)
    cmap = plt.colormaps['viridis'].resampled(n_groups)

    # Create scatter plot with discrete colors
    scatter = ax1.scatter(event_throughputs, stored_data_per_event,
                         c=num_groups,  # This is already a numpy array of the right size
                         cmap=cmap,
                         s=total_cpu_times/1000, alpha=0.6)

    # Add colorbar with discrete ticks
    cbar = plt.colorbar(scatter, ax=ax1, label="Number of Groups")
    cbar.set_ticks(unique_groups)
    cbar.set_ticklabels([f"{int(x)}" for x in unique_groups])

    ax1.set_xlabel("Event Throughput (events/second)")
    ax1.set_ylabel("Stored Data per Event (MB)")
    ax1.set_title("Performance vs Storage Efficiency\n(size=CPU time, color=num groups)")
    ax1.grid(True)

    # set x-axis to start at 0 and add 10% padding to the right
    ax1.set_xlim(left=0, right=np.max(event_throughputs) * 1.1)

    # 5. Network Transfer Analysis
    ax7 = fig.add_subplot(gs[2, 0])
    network_transfer = []
    for metrics in construction_metrics:
        # Calculate network transfer as sum of input and output data
        transfer = metrics["input_data_per_event_mb"] + metrics["output_data_per_event_mb"]
        network_transfer.append(transfer)

    ax7.bar(range(len(construction_metrics)), network_transfer)
    ax7.set_xlabel("Workflow Construction")
    ax7.set_ylabel("Network Transfer (MB)")
    ax7.set_title("Network Transfer Analysis")
    ax7.set_xticks(range(len(construction_metrics)))
    ax7.set_xticklabels([f"Const {i+1}" for i in range(len(construction_metrics))], rotation=45)
    ax7.grid(True)

    # 6. CPU Utilization Analysis
    ax4 = fig.add_subplot(gs[2, 1])
    cpu_utilization = []
    for metrics in construction_metrics:
        # Get CPU utilization ratio for each group from the original groups data
        util = []
        for group_id in metrics["groups"]:
            # Find the corresponding group in the original groups data
            group_data = next((g for g in groups if g["group_id"] == group_id), None)
            if group_data:
                util.append(group_data["resource_metrics"]["cpu"]["utilization_ratio"])
        cpu_utilization.append(util)

    ax4.boxplot(cpu_utilization, tick_labels=[f"Const {i+1}" for i in range(len(construction_metrics))])
    ax4.set_xlabel("Workflow Construction")
    ax4.set_ylabel("CPU Utilization Ratio")
    ax4.set_title("CPU Utilization Analysis\n(Actual CPU Usage / Allocated CPU)")
    ax4.set_xticklabels([f"Const {i+1}" for i in range(len(construction_metrics))], rotation=45)
    ax4.grid(True)

    # 7. Memory Utilization Analysis
    ax6 = fig.add_subplot(gs[3, 0])
    memory_utilization = []
    memory_std = []  # Store standard deviations
    for metrics in construction_metrics:
        # Get memory occupancy for each group from the original groups data
        occupancies = []
        for group_id in metrics["groups"]:
            # Find the corresponding group in the original groups data
            group_data = next((g for g in groups if g["group_id"] == group_id), None)
            if group_data:
                occupancies.append(group_data["resource_metrics"]["memory"]["occupancy"])
        # Calculate average and standard deviation of memory occupancy
        if occupancies:
            avg_occupancy = sum(occupancies) / len(occupancies)
            std_occupancy = np.std(occupancies)
            memory_utilization.append(avg_occupancy)
            memory_std.append(std_occupancy)
        else:
            memory_utilization.append(0)
            memory_std.append(0)

    # Create bar plot with error bars
    x = range(len(construction_metrics))
    ax6.bar(x, memory_utilization, yerr=memory_std, capsize=5)
    ax6.set_xlabel("Workflow Construction")
    ax6.set_ylabel("Memory Utilization Ratio")
    ax6.set_title("Memory Utilization Analysis\n(Average Memory Occupancy ± Std Dev)")
    ax6.set_xticks(x)
    ax6.set_xticklabels([f"Const {i+1}" for i in range(len(construction_metrics))], rotation=45)
    ax6.grid(True)

    # 8. Event Processing Analysis
    ax5 = fig.add_subplot(gs[3, 1])
    events_per_group = []
    for metrics in construction_metrics:
        events = [group["total_events"] for group in metrics["group_details"]]
        events_per_group.append(events)

    ax5.boxplot(events_per_group, tick_labels=[f"Const {i+1}" for i in range(len(construction_metrics))])
    ax5.set_xlabel("Workflow Construction")
    ax5.set_ylabel("Events per Group")
    ax5.set_title("Event Processing Distribution")
    ax5.set_xticklabels([f"Const {i+1}" for i in range(len(construction_metrics))], rotation=45)
    ax5.grid(True)

    # 9. Parallelism Analysis
    ax9 = fig.add_subplot(gs[4, 0])
    parallelism_metrics = []

    for metrics in construction_metrics:
        # Calculate metrics for parallel execution analysis
        group_details = metrics["group_details"]

        # Calculate sequential execution time (sum of all CPU times)
        sequential_time = sum(group["cpu_seconds"] for group in group_details)

        # Calculate parallel execution time by following the DAG dependencies
        parallel_time = 0

        # Create a mapping of group dependencies
        group_deps = {}
        for i, group in enumerate(group_details):
            deps = []
            for j, prev_group in enumerate(group_details[:i]):
                # If this group needs input from a previous group
                if group["input_data_mb"] > 0 and prev_group["output_data_mb"] > 0:
                    deps.append(j)
            group_deps[i] = deps

        # Calculate the longest path through the DAG
        def get_longest_path(group_idx, visited=None):
            if visited is None:
                visited = set()
            if group_idx in visited:
                return 0
            visited.add(group_idx)

            # Get all dependent groups
            next_groups = [i for i, deps in group_deps.items() if group_idx in deps]
            if not next_groups:
                return group_details[group_idx]["cpu_seconds"]

            # Recursively find the longest path
            max_path = 0
            for next_group in next_groups:
                path_length = get_longest_path(next_group, visited.copy())
                max_path = max(max_path, path_length)

            return group_details[group_idx]["cpu_seconds"] + max_path

        # Find the longest path starting from each group
        parallel_time = max(get_longest_path(i) for i in range(len(group_details)))

        # Calculate parallel efficiency as ratio of sequential to parallel time
        # Higher ratio means better parallelization
        parallel_efficiency = sequential_time / parallel_time if parallel_time > 0 else 1.0

        # Store metrics
        parallelism_metrics.append({
            "sequential_time": sequential_time,
            "parallel_time": parallel_time,
            "parallel_efficiency": parallel_efficiency
        })

    # Plot parallel efficiency
    parallel_efficiency = [m["parallel_efficiency"] for m in parallelism_metrics]
    ax9.bar(range(len(construction_metrics)), parallel_efficiency)
    ax9.set_xlabel("Workflow Construction")
    ax9.set_ylabel("Parallel Efficiency")
    ax9.set_title("Parallel Execution Analysis\n(Efficiency = Sequential Time / Parallel Time)")
    ax9.set_xticks(range(len(construction_metrics)))
    ax9.set_xticklabels([f"Const {i+1}" for i in range(len(construction_metrics))], rotation=45)
    ax9.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "workflow_comparison.png"))
    plt.close()

    # Create a detailed comparison table
    with open(os.path.join(output_dir, "workflow_comparison.txt"), "w") as f:
        f.write("Workflow Construction Comparison\n")
        f.write("==============================\n\n")

        for i, metrics in enumerate(construction_metrics, 1):
            f.write(f"Construction {i}:\n")
            f.write(f"  Groups: {metrics['groups']}\n")
            f.write(f"  Number of Groups: {metrics['num_groups']}\n")
            f.write(f"  Event Throughput: {metrics['event_throughput']:.4f} events/second\n")
            f.write(f"  Total CPU Time: {metrics['total_cpu_time']:.2f} seconds\n")
            f.write("  Total Data Volumes for one job of each group:\n")
            f.write(f"    Input Data: {metrics['total_input_data_mb']:.2f} MB\n")
            f.write(f"    Output Data: {metrics['total_output_data_mb']:.2f} MB\n")
            f.write(f"    Stored Data: {metrics['total_stored_data_mb']:.2f} MB\n")
            f.write("  Data Flow Metrics (per event):\n")
            f.write(f"    Input Data: {metrics['input_data_per_event_mb']:.3f} MB/event\n")
            f.write(f"    Output Data: {metrics['output_data_per_event_mb']:.3f} MB/event\n")
            f.write(f"    Stored Data: {metrics['stored_data_per_event_mb']:.3f} MB/event\n")
            f.write(f"  Memory Utilization: {memory_utilization[i-1]:.2f}\n")
            f.write(f"  Network Transfer: {network_transfer[i-1]:.2f} MB\n")
           # f.write(f"  Estimated Cost: ${costs[i-1]:.2f}\n")
            f.write("  Parallel Execution Metrics:\n")
            f.write(f"    Sequential Time: {parallelism_metrics[i-1]['sequential_time']:.2f} seconds\n")
            f.write(f"    Parallel Time: {parallelism_metrics[i-1]['parallel_time']:.2f} seconds\n")
            f.write(f"    Parallel Efficiency: {parallelism_metrics[i-1]['parallel_efficiency']:.3f}\n")
            f.write("  Group Details:\n")
            for group in metrics["group_details"]:
                f.write(f"    {group['group_id']}:\n")
                f.write(f"      Tasks: {group['tasks']}\n")
                f.write(f"      Events per Task: {group['events_per_task']}\n")
                f.write(f"      CPU Time: {group['cpu_seconds']:.2f} seconds\n")
                f.write("      Data Flow (per event):\n")
                f.write(f"        Input: {group['input_data_per_event_mb']:.3f} MB/event\n")
                f.write(f"        Output: {group['output_data_per_event_mb']:.3f} MB/event\n")
                f.write(f"        Stored: {group['stored_data_per_event_mb']:.3f} MB/event\n")
            f.write("\n")

def filter_toy_model_constructions(construction_metrics: List[Dict]) -> List[Dict]:
    """Filter construction_metrics to include only the two extreme cases for toy model.

    Returns:
        List containing only two constructions:
        1. Single group with all tasks
        2. One group per task (maximum number of groups)
    """
    if not construction_metrics:
        return []

    # Find the construction with the minimum number of groups (single group)
    single_group_construction = min(construction_metrics, key=lambda x: x["num_groups"])

    # Find the construction with the maximum number of groups (one group per task)
    max_groups_construction = max(construction_metrics, key=lambda x: x["num_groups"])

    # Verify we have different constructions
    if single_group_construction == max_groups_construction:
        print("Warning: Only one construction type found. Returning all constructions.")
        return construction_metrics

    print(f"Toy model: Single group construction has {single_group_construction['num_groups']} groups")
    print(f"Toy model: Max groups construction has {max_groups_construction['num_groups']} groups")

    return [single_group_construction, max_groups_construction]


def visualize_toy_model(groups: List[Dict],
                       construction_metrics: List[Dict],
                       template_path: str,
                       output_dir: str = "output",
                       dag: nx.DiGraph = None):
    """Create visualizations for the toy model with only two extreme constructions.

    This function creates the same visualizations as visualize_groups but only
    for the two extreme cases: single group and one group per task.
    """
    print("Creating toy model visualizations with two extreme constructions...")

    # Filter to only the two extreme constructions
    toy_constructions = filter_toy_model_constructions(construction_metrics)

    if len(toy_constructions) != 2:
        print(f"Warning: Expected 2 constructions for toy model, got {len(toy_constructions)}")
        return

    # Filter groups to only include those used in the toy constructions
    toy_group_ids = set()
    for construction in toy_constructions:
        toy_group_ids.update(construction["groups"])

    # Filter groups list to only include groups used in toy constructions
    toy_groups = [group for group in groups if group["group_id"] in toy_group_ids]

    print(f"Toy model: Using {len(toy_groups)} groups out of {len(groups)} total groups")
    print(f"Toy model: Group IDs used: {sorted(toy_group_ids)}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate all the same visualizations but with filtered data
    print(f"Creating toy model visualizations for {len(toy_constructions)} constructions")

    # 1. Group-level data volume analysis (the main plot you're working on)
    plot_group_data_volume_analysis(toy_constructions, str(output_path))

    # 2. Workflow construction comparison
    plot_workflow_comparison(toy_constructions, str(output_path))

    # 3. Storage efficiency analysis
    plot_storage_efficiency(toy_constructions, str(output_path))

    # 4. Workflow constructions overview
    plot_workflow_constructions(toy_constructions, str(output_path))

    # 5. Resource utilization analysis (using filtered groups)
    plot_resource_utilization(toy_groups, str(output_path))

    # 6. Throughput analysis (using filtered groups)
    plot_throughput_analysis(toy_groups, str(output_path))

    # 7. Dependency analysis (using filtered groups)
    plot_dependency_analysis(toy_groups, str(output_path))

    # 8. Workflow topology visualization
    plot_workflow_topology(toy_constructions, str(output_path),
                          Path(template_path).name, dag)

    print(f"Toy model visualizations saved to {output_path}")


def visualize_groups(groups: List[Dict],
                     construction_metrics: List[Dict],
                     template_path: str,
                     output_dir: str = "output",
                     dag: nx.DiGraph = None):
    """Generate all visualizations for the task groups and workflow constructions

    Args:
        groups: List of group metrics dictionaries
        construction_metrics: List of construction metrics dictionaries
        template_path: path to the JSON template
        output_dir: Base output directory (default: "output")
        dag: The directed acyclic graph representing task dependencies
    """
    # Extract template directory name from the path
    # This will get the parent directory name (e.g., "sequential" or "fork")
    tmpl_dir = template_path.parent.name
    # extract file name without the extension
    tmpl_name = template_path.stem
    # Create output directory with template subdirectory
    output_path = os.path.join(output_dir, tmpl_dir, tmpl_name)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    print(f"\n*** Saving output data at: {output_path}")

    # Create plots for individual groups
    plot_resource_utilization(groups, output_path)
    plot_throughput_analysis(groups, output_path)
    plot_dependency_analysis(groups, output_path)
    plot_comparison_heatmap(groups, output_path)

    # Create plots for workflow constructions
    plot_workflow_constructions(construction_metrics, output_path)
    plot_storage_efficiency(construction_metrics, output_path)
    plot_workflow_comparison(construction_metrics, output_path)
    plot_group_data_volume_analysis(construction_metrics, output_path)  # New dedicated plot
    plot_workflow_topology(construction_metrics, output_path, template_path, dag)

    # Save raw data for further analysis
    print(f"Saving raw data for {len(groups)} groups and {len(construction_metrics)} constructions")
    with open(os.path.join(output_path, "group_metrics.json"), "w") as f:
        json.dump(groups, f, indent=2)
    with open(os.path.join(output_path, "construction_metrics.json"), "w") as f:
        json.dump(construction_metrics, f, indent=2)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Visualize workflow groups and constructions from a template file.')
    parser.add_argument('template_file', type=str, help='Path to the template JSON file')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Base output directory (default: output)')
    parser.add_argument('--toy-model', action='store_true',
                       help='Create toy model with only two extreme constructions (single group vs one group per task)')
    args = parser.parse_args()

    # Convert template file path to Path object for easier manipulation
    template_path = Path(args.template_file)

    print(f"Parsing workflow data from template: {template_path}")
    with open(template_path) as f:
        workflow_data = json.load(f)

    groups, tasks, construction_metrics, dag = create_workflow_from_json(workflow_data)

    if args.toy_model:
        print("\nRunning in toy model mode - visualizing only two extreme constructions")
        visualize_toy_model(groups, construction_metrics, template_path, args.output_dir, dag)
    else:
        print("Running in full mode - visualizing all constructions")
        visualize_groups(groups, construction_metrics, template_path, args.output_dir, dag)

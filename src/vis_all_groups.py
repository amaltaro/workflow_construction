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
from find_all_groups import create_workflow_from_json, TARGET_WALLCLOCK_TIME_HOURS
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
        input_data_sizes.append(group["resource_metrics"]["io"]["read_remote_mb"])
        output_data_sizes.append(group["resource_metrics"]["io"]["write_local_mb"])
        stored_data_sizes.append(group["resource_metrics"]["io"]["write_remote_mb"])

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
    axes[1, 0].scatter(group_sizes, input_data_sizes, label="Remote Read Data", alpha=0.6)
    axes[1, 0].scatter(group_sizes, output_data_sizes, label="Local Write Data", alpha=0.6)
    axes[1, 0].scatter(group_sizes, stored_data_sizes, label="Remote Write Data", alpha=0.6)
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

    axes[1, 1].set_title("Throughput vs Remote Write Data Size\n(color indicates group size)")
    axes[1, 1].set_xlabel("Total Events per Second")
    axes[1, 1].set_ylabel("Remote Write Data Size (MB)")
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
        "Remote Read Data Size": [],
        "Local Write Data Size": [],
        "Remote Write Data Size": [],
        "Events per Job": [],
        "Dependency Paths": []
    }

    for group in groups:
        metrics["Group Size"].append(len(group["task_ids"]))
        metrics["CPU Utilization Ratio"].append(group["resource_metrics"]["cpu"]["utilization_ratio"])
        metrics["Memory Occupancy"].append(group["resource_metrics"]["memory"]["occupancy"])
        metrics["Resource Utilization"].append(group["utilization_metrics"]["resource_utilization"])
        metrics["Total Throughput"].append(group["resource_metrics"]["throughput"]["total_eps"])
        metrics["Remote Read Data Size"].append(group["resource_metrics"]["io"]["read_remote_mb"])
        metrics["Local Write Data Size"].append(group["resource_metrics"]["io"]["write_local_mb"])
        metrics["Remote Write Data Size"].append(group["resource_metrics"]["io"]["write_remote_mb"])
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
        stored_data_per_event.append(metrics["write_remote_per_event_mb"])
        total_stored_data.append(metrics["total_write_remote_mb"])
        total_events.append(metrics["total_events"])

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Storage Efficiency Analysis for Workflow Constructions", fontsize=16)

    # Plot 1: Remote Write Data per Event vs Number of Groups
    sns.scatterplot(x=num_groups, y=stored_data_per_event, ax=axes[0, 0])
    axes[0, 0].set_title("Remote Write Efficiency vs Number of Groups")
    axes[0, 0].set_xlabel("Number of Groups")
    axes[0, 0].set_ylabel("Remote Write Data per Event (MB)")
    axes[0, 0].grid(True)

    # Plot 2: Remote Write Data per Event vs Event Throughput
    sns.scatterplot(x=event_throughputs, y=stored_data_per_event, ax=axes[0, 1])
    axes[0, 1].set_title("Remote Write Efficiency vs Event Throughput")
    axes[0, 1].set_xlabel("Event Throughput (events/second)")
    axes[0, 1].set_ylabel("Remote Write Data per Event (MB)")
    axes[0, 1].grid(True)

    # Plot 3: Total Remote Write Data vs Total Events
    # Create discrete color bins for number of groups
    num_groups_array = np.array(num_groups)
    unique_num_groups = np.unique(num_groups_array)

    # Create a discrete colormap
    n_groups = len(unique_num_groups)
    cmap = plt.colormaps['cool'].resampled(n_groups)

    # Convert MB to GB for better readability
    total_stored_data_gb = [data / 1024.0 for data in total_stored_data]

    # Create scatter plot with discrete colors
    scatter = axes[1, 0].scatter(total_events, total_stored_data_gb,
                                c=num_groups_array, cmap=cmap, s=100, alpha=0.7)

    # Add colorbar with discrete ticks
    cbar = plt.colorbar(scatter, ax=axes[1, 0], label="Number of Groups")
    cbar.set_ticks(unique_num_groups)
    cbar.set_ticklabels([f"{int(x)}" for x in unique_num_groups])

    axes[1, 0].set_title("Total Remote Write Data vs Total Events\n(color indicates number of groups)")
    axes[1, 0].set_xlabel("Total Events")
    axes[1, 0].set_ylabel("Total Remote Write Data (GB)")
    axes[1, 0].grid(True)

    # Plot 4: Remote Write Data per Event Distribution
    sns.histplot(stored_data_per_event, bins=20, ax=axes[1, 1])
    axes[1, 1].set_title("Remote Write Data per Event Distribution")
    axes[1, 1].set_xlabel("Remote Write Data per Event (MB)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "storage_efficiency.png"))
    plt.close()

def plot_workflow_constructions(construction_metrics: List[Dict], output_dir: str = "plots", custom_labels: List[str] = None):
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
        stored_data_per_event.append(metrics["write_remote_per_event_mb"])
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
    # Create discrete color bins for remote write data per event
    stored_data_array = np.array(stored_data_per_event)
    unique_stored_data = np.unique(stored_data_array)

    # Create a discrete colormap
    n_stored = len(unique_stored_data)
    cmap = plt.colormaps['magma'].resampled(n_stored)

    # Create scatter plot with discrete colors
    scatter = axes[0, 1].scatter(total_cpu_times, total_events,
                                c=stored_data_array, cmap=cmap, s=100, alpha=0.7)

    # Add colorbar with discrete ticks
    cbar = plt.colorbar(scatter, ax=axes[0, 1], label="Remote Write Data per Event (MB)")
    cbar.set_ticks(unique_stored_data)
    cbar.set_ticklabels([f"{x:.3f}" for x in unique_stored_data])

    axes[0, 1].set_title("Total Events vs Total CPU Time\n(color indicates remote write data per event)")
    axes[0, 1].set_xlabel("Total CPU Time (seconds)")
    axes[0, 1].set_ylabel("Total Events")
    axes[0, 1].grid(True)

    # Plot 3: Event Throughput Distribution
    sns.histplot(event_throughputs, bins=20, ax=axes[1, 0])
    axes[1, 0].set_title("Event Throughput Distribution")
    axes[1, 0].set_xlabel("Events per Second")
    axes[1, 0].set_ylabel("Number of Workflow Constructions")
    axes[1, 0].grid(True)

    # Plot 4: Remote Write Data per Event vs Event Throughput
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

    axes[1, 1].set_title("Remote Write Efficiency vs Event Throughput\n(color indicates number of groups)")
    axes[1, 1].set_xlabel("Event Throughput (events/second)")
    axes[1, 1].set_ylabel("Remote Write Data per Event (MB)")
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
            # Use custom label if provided and if this construction is in the toy model
            if custom_labels and len(construction_metrics) == 2:
                # For toy model, find the index of this construction in the original list
                try:
                    original_index = construction_metrics.index(metrics)
                    if original_index < len(custom_labels):
                        construction_label = custom_labels[original_index]
                    else:
                        construction_label = f"Construction {i}"
                except ValueError:
                    construction_label = f"Construction {i}"
            else:
                construction_label = f"Construction {i}"
            f.write(f"{construction_label}:\n")
            f.write(f"  Groups: {metrics['groups']}\n")
            f.write(f"  Number of Groups: {metrics['num_groups']}\n")
            f.write(f"  Total Events: {metrics['total_events']}\n")
            f.write(f"  Total CPU Time: {metrics['total_cpu_time']:.2f} seconds\n")
            f.write(f"  Event Throughput: {metrics['event_throughput']:.4f} events/second\n")
            f.write(f"  Remote Write Data per Event: {metrics['write_remote_per_event_mb']:.3f} MB/event\n\n")

def plot_group_data_volume_analysis(construction_metrics: List[Dict], output_dir: str = "plots", custom_labels: List[str] = None):
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
            # Use custom label if provided, otherwise use default "Const" label
            if custom_labels and i < len(custom_labels):
                construction_label = custom_labels[i]
            else:
                construction_label = f"Const {i+1}"
            group_labels.append(f"{construction_label} - {group_id} ({tasks_str})")
        # Move to next construction position (add spacing only between constructions)
        current_pos += groups_in_construction * bar_height + bar_height

    left = np.zeros(len(group_positions))
    # Plot remote read data
    input_data = [group["read_remote_mb"] for metrics in construction_metrics
                 for group in metrics["group_details"]]
    ax.barh(group_positions, input_data, bar_height, label='Remote Read', left=left, alpha=0.8,
            edgecolor='black', linewidth=0.4)
    left += input_data

    # Plot local write data
    output_data = [group["write_local_mb"] for metrics in construction_metrics
                  for group in metrics["group_details"]]
    ax.barh(group_positions, output_data, bar_height, label='Local Write', left=left, alpha=0.8,
            edgecolor='black', linewidth=0.4)
    left += output_data

    # Plot remote write data
    stored_data = [group["write_remote_mb"] for metrics in construction_metrics
                  for group in metrics["group_details"]]
    ax.barh(group_positions, stored_data, bar_height, label='Remote Write', left=left, alpha=0.8,
            edgecolor='black', linewidth=0.4)

    for i, (pos, input_val, output_val, stored_val) in enumerate(zip(group_positions, input_data, output_data, stored_data)):
        total = input_val + output_val + stored_val
        ax.text(total + 0.1, pos, f'{total:.1f}', va='center', fontsize=8, alpha=0.8)

    ax.set_ylabel("Workflow Construction and Groups")
    ax.set_xlabel("Total Data Volume (MB)")
    ax.set_title(f"Group-level Data Volume Analysis\n(One job per Group - {len(construction_metrics)} Constructions)", fontsize=14, fontweight='bold')
    ax.set_yticks(group_positions)
    ax.set_yticklabels(group_labels, fontsize=8)  # Reduced font size for better fit
    ax.legend(loc='best', fontsize=10)  # or "upper right", etc
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


def plot_job_scaling_analysis(construction_metrics: List[Dict], output_dir: str = "plots", custom_labels: List[str] = None):
    """Create a visualization showing job scaling for different workflow constructions.

    This plot shows how many jobs each group needs to run to process the requested
    number of events, helping to understand the scaling behavior of different
    workflow constructions.
    """
    print(f"Creating job scaling analysis for {len(construction_metrics)} constructions")

    # Helper function to get construction labels
    if custom_labels:
        construction_labels = custom_labels
    else:
        construction_labels = [f"Const {i+1}" for i in range(len(construction_metrics))]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: Total jobs needed per construction
    total_jobs = []
    for metrics in construction_metrics:
        total_jobs.append(sum(metrics["group_jobs_needed"].values()))

    bars = ax1.bar(range(len(construction_metrics)), total_jobs, alpha=0.7)
    ax1.set_xlabel("Workflow Construction")
    ax1.set_ylabel("Total Jobs Needed")
    ax1.set_title("Total Jobs Required per Workflow Construction")
    ax1.set_xticks(range(len(construction_metrics)))
    ax1.set_xticklabels(construction_labels, rotation=45)
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (bar, total) in enumerate(zip(bars, total_jobs)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(total_jobs)*0.01,
                f'{total:.0f}', ha='center', va='bottom', fontweight='bold')

    # Plot 2: Jobs per group (stacked bar chart)
    # Get all unique group IDs across all constructions
    all_group_ids = set()
    for metrics in construction_metrics:
        all_group_ids.update(metrics["group_jobs_needed"].keys())
    all_group_ids = sorted(all_group_ids)

    # Create stacked bar chart
    bottom = np.zeros(len(construction_metrics))
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_group_ids)))

    for i, group_id in enumerate(all_group_ids):
        jobs_data = []
        for metrics in construction_metrics:
            jobs_data.append(metrics["group_jobs_needed"].get(group_id, 0))

        ax2.bar(range(len(construction_metrics)), jobs_data, bottom=bottom,
                label=group_id, color=colors[i], alpha=0.8)
        bottom += np.array(jobs_data)

    ax2.set_xlabel("Workflow Construction")
    ax2.set_ylabel("Jobs per Group")
    ax2.set_title("Job Distribution Across Groups")
    ax2.set_xticks(range(len(construction_metrics)))
    ax2.set_xticklabels(construction_labels, rotation=45)
    ax2.legend(title="Group ID", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Add total value labels on stacked bars
    for i, total in enumerate(total_jobs):
        ax2.text(i, total + max(total_jobs)*0.01, f'{total:.0f}',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "job_scaling_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Create a detailed text report
    with open(os.path.join(output_dir, "job_scaling_report.txt"), "w") as f:
        f.write("Job Scaling Analysis Report\n")
        f.write("==========================\n\n")
        f.write(f"Requested Events: {construction_metrics[0]['request_num_events']:,}\n\n")

        for i, metrics in enumerate(construction_metrics, 1):
            construction_label = construction_labels[i-1] if i <= len(construction_labels) else f"Construction {i}"
            f.write(f"{construction_label}:\n")
            f.write(f"  Groups: {metrics['groups']}\n")
            f.write(f"  Total Jobs: {sum(metrics['group_jobs_needed'].values()):.0f}\n")
            f.write(f"  CPU Time per Event: {metrics['cpu_time_per_event']:.4f} seconds/event\n")
            f.write("  Jobs per Group:\n")
            for group_id, jobs_needed in metrics['group_jobs_needed'].items():
                f.write(f"    {group_id}: {jobs_needed:.1f} jobs\n")
            f.write("\n")


def plot_time_analysis(construction_metrics: List[Dict], output_dir: str = "plots", custom_labels: List[str] = None):
    """Create dedicated time analysis plots for workflow constructions.

    This function creates a comprehensive time analysis considering:
    1. Baseline (infinite resources) - ideal execution time
    2. Realistic grid slot constraints (100 and 1000 slots)
    3. Impact of task dependencies on parallelization
    """
    print(f"Creating dedicated time analysis for {len(construction_metrics)} constructions")

    # Helper function to get construction labels
    if custom_labels:
        construction_labels = custom_labels
    else:
        construction_labels = [f"Const {i+1}" for i in range(len(construction_metrics))]

    # Calculate time metrics for different scenarios
    baseline_times = []  # Infinite resources
    grid_100_times = []  # 100 grid slots
    grid_1000_times = []  # 1000 grid slots

    for metrics in construction_metrics:
        group_details = metrics["group_details"]
        group_jobs_needed = metrics["group_jobs_needed"]

        # Calculate baseline time (infinite resources)
        # This considers task dependencies and group structure
        baseline_time = calculate_baseline_time(group_details, group_jobs_needed)
        baseline_times.append(baseline_time)

        # Calculate time with 100 grid slots
        grid_100_time = calculate_constrained_time(group_details, group_jobs_needed, 100)
        grid_100_times.append(grid_100_time)

        # Calculate time with 1000 grid slots
        grid_1000_time = calculate_constrained_time(group_details, group_jobs_needed, 1000)
        grid_1000_times.append(grid_1000_time)

    # Convert seconds to hours for plotting
    baseline_hours = [t / 3600 for t in baseline_times]
    grid_100_hours = [t / 3600 for t in grid_100_times]
    grid_1000_hours = [t / 3600 for t in grid_1000_times]

    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # Set up bar positions
    x = np.arange(len(construction_metrics))
    width = 0.25

    # Plot three bars for each construction (using hours)
    bars1 = ax.bar(x - width, baseline_hours, width, label='Baseline (Infinite)', color='lightgreen', alpha=0.8)
    bars2 = ax.bar(x, grid_100_hours, width, label='100 Grid Slots', color='red', alpha=0.8)
    bars3 = ax.bar(x + width, grid_1000_hours, width, label='1000 Grid Slots', color='orange', alpha=0.8)

    ax.set_xlabel("Workflow Construction")
    ax.set_ylabel("Execution Time (hours)")
    ax.set_title("Workflow Execution Time Analysis\n(Considering Task Dependencies and Grid Constraints)")
    ax.set_xticks(x)
    ax.set_xticklabels(construction_labels, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set y-axis to hours
    ax.set_ylabel("Execution Time (hours)")

    # Add value labels on bars
    for i, (bar1, bar2, bar3, h1, h2, h3) in enumerate(zip(bars1, bars2, bars3, baseline_hours, grid_100_hours, grid_1000_hours)):
        ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + max(baseline_hours)*0.01,
                f'{h1:.1f}h', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + max(grid_100_hours)*0.01,
                f'{h2:.1f}h', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.text(bar3.get_x() + bar3.get_width()/2, bar3.get_height() + max(grid_1000_hours)*0.01,
                f'{h3:.1f}h', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_analysis.png"))
    plt.close()

    # Create a detailed time analysis report
    with open(os.path.join(output_dir, "time_analysis.txt"), "w") as f:
        f.write("Workflow Time Analysis Report\n")
        f.write("===========================\n\n")

        for i, metrics in enumerate(construction_metrics, 1):
            # Use custom label if provided, otherwise use default "Construction" label
            if custom_labels and i <= len(custom_labels):
                construction_label = custom_labels[i-1]
            else:
                construction_label = f"Construction {i}"
            f.write(f"{construction_label}:\n")
            f.write(f"  Groups: {metrics['groups']}\n")
            f.write(f"  Number of Groups: {metrics['num_groups']}\n")
            f.write(f"  Baseline Time (Infinite Resources): {baseline_hours[i-1]:.1f} hours\n")
            f.write(f"  Time with 100 Grid Slots: {grid_100_hours[i-1]:.1f} hours\n")
            f.write(f"  Time with 1000 Grid Slots: {grid_1000_hours[i-1]:.1f} hours\n")
            f.write("  Group Jobs Needed:\n")
            for group_id, jobs_needed in metrics['group_jobs_needed'].items():
                f.write(f"    {group_id}: {jobs_needed:.1f} jobs\n")
            f.write("\n")


def calculate_baseline_time(group_details: List[Dict], group_jobs_needed: Dict[str, float]) -> float:
    """Calculate baseline execution time with infinite resources.

    This considers task dependencies and group structure to determine
    the minimum time needed to complete the workflow.
    """
    if len(group_details) == 1:
        # Single group: all jobs can run in parallel
        # Time = wallclock time for one job
        return TARGET_WALLCLOCK_TIME_HOURS * 3600
    else:
        # Multiple groups: must consider dependencies
        # For now, assume sequential execution of groups
        # TODO: Implement proper DAG-based dependency analysis
        total_time = 0
        for group in group_details:
            group_id = group["group_id"]
            jobs_needed = group_jobs_needed[group_id]
            # Each group takes wallclock time to complete all its jobs
            group_time = TARGET_WALLCLOCK_TIME_HOURS * 3600
            total_time += group_time
        return total_time


def calculate_constrained_time(group_details: List[Dict], group_jobs_needed: Dict[str, float], grid_slots: int) -> float:
    """Calculate execution time with limited grid slots.

    This simulates realistic grid constraints and job scheduling,
    accounting for event dependencies between groups.
    """
    if len(group_details) == 1:
        # Single group: all jobs can run in parallel up to grid slot limit
        group_id = group_details[0]["group_id"]
        jobs_needed = group_jobs_needed[group_id]

        # Calculate how many job batches we need (ceiling division)
        batches_needed = max(1, int((jobs_needed + grid_slots - 1) / grid_slots))
        return batches_needed * TARGET_WALLCLOCK_TIME_HOURS * 3600
    else:
        # Multiple groups: must account for event dependencies
        total_time = 0
        cumulative_events_processed = 0

        for i, group in enumerate(group_details):
            group_id = group["group_id"]
            jobs_needed = group_jobs_needed[group_id]
            events_per_job = group["events_per_task"]  # events per job for this group

            if i == 0:
                # First group: can start immediately
                # Calculate batches needed for this group
                batches_needed = max(1, int((jobs_needed + grid_slots - 1) / grid_slots))
                group_time = batches_needed * TARGET_WALLCLOCK_TIME_HOURS * 3600
                total_time += group_time
                cumulative_events_processed = jobs_needed * events_per_job
            else:
                # Subsequent groups: must wait for enough events from previous group
                # Find the previous group that feeds into this one
                prev_group = group_details[i-1]
                prev_events_per_job = prev_group["events_per_task"]

                # Calculate how many events this group needs to start
                # This is typically the events_per_job of the current group
                events_needed_to_start = events_per_job

                # Calculate how many jobs from previous group must complete
                jobs_from_prev_needed = max(1, int((events_needed_to_start + prev_events_per_job - 1) / prev_events_per_job))

                # Calculate time to get enough events from previous group
                # We need to calculate how long it takes for the previous group to produce enough events
                # This is the time it takes for the required number of jobs from the previous group to complete
                time_for_prev_events = (jobs_from_prev_needed / grid_slots) * TARGET_WALLCLOCK_TIME_HOURS * 3600

                # Time for current group to complete
                current_batches_needed = max(1, int((jobs_needed + grid_slots - 1) / grid_slots))
                time_for_current_group = current_batches_needed * TARGET_WALLCLOCK_TIME_HOURS * 3600

                # Total time is the maximum of:
                # 1. Time to get enough events from previous group
                # 2. Time for current group to complete
                group_total_time = max(time_for_prev_events, time_for_current_group)
                total_time += group_total_time

                cumulative_events_processed += jobs_needed * events_per_job

        return total_time


def plot_workflow_comparison(construction_metrics: List[Dict], output_dir: str = "plots", custom_labels: List[str] = None):
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
        stored_data_per_event.append(metrics["write_remote_per_event_mb"])
        total_stored_data.append(metrics["total_write_remote_mb"])
        input_data_per_event.append(metrics["read_remote_per_event_mb"])
        output_data_per_event.append(metrics["write_local_per_event_mb"])
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
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1])  # Equal height ratios for all rows

    # Helper function to get construction labels
    if custom_labels:
        construction_labels = custom_labels
    else:
        construction_labels = [f"Const {i+1}" for i in range(len(construction_metrics))]

    # 1. Group Size Distribution
    ax3 = fig.add_subplot(gs[0, 0])
    group_sizes = []
    for metrics in construction_metrics:
        sizes = [len(group["tasks"]) for group in metrics["group_details"]]
        group_sizes.append(sizes)

    # Create a box plot for group sizes
    ax3.boxplot(group_sizes, tick_labels=construction_labels)
    ax3.set_xlabel("Workflow Construction")
    ax3.set_ylabel("Number of Tasks per Group")
    ax3.set_title("Group Size Distribution")
    ax3.set_xticklabels(construction_labels, rotation=45)
    ax3.grid(True)
    ax3.set_ylim(bottom=0)  # Set y-axis to start at 0

    # 2. Data Flow Analysis (Updated to use per-event metrics)
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(construction_metrics))
    width = 0.25
    ax2.bar(x - width, input_data_per_event, width, label='Remote Read')
    ax2.bar(x, output_data_per_event, width, label='Local Write')
    ax2.bar(x + width, stored_data_per_event, width, label='Remote Write')
    ax2.set_xlabel("Workflow Construction")
    ax2.set_ylabel("Data Volume per Event (MB)")
    ax2.set_title("Data Volume Analysis Per Event")
    ax2.set_xticks(x)
    ax2.set_xticklabels(construction_labels, rotation=45)
    ax2.legend()
    ax2.grid(True)

    # 3. Total Data Volume Analysis (Stacked Bar)
    ax10 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(construction_metrics))
    width = 0.6
    bottom = np.zeros(len(construction_metrics))

    # Convert MB to GB for better readability
    remote_read_gb = [m["total_read_remote_mb"] / 1024.0 for m in construction_metrics]
    local_write_gb = [m["total_write_local_mb"] / 1024.0 for m in construction_metrics]
    remote_write_gb = [m["total_write_remote_mb"] / 1024.0 for m in construction_metrics]

    # Plot each data type as a layer in the stack
    ax10.bar(x, remote_read_gb, width, label='Remote Read', bottom=bottom)
    bottom += remote_read_gb

    ax10.bar(x, local_write_gb, width, label='Local Write', bottom=bottom)
    bottom += local_write_gb

    ax10.bar(x, remote_write_gb, width, label='Remote Write', bottom=bottom)

    # Add total value labels on top of each bar
    totals_gb = [rr + lw + rw for rr, lw, rw in zip(remote_read_gb, local_write_gb, remote_write_gb)]
    for i, total in enumerate(totals_gb):
        ax10.text(i, total, f'{total:.1f}', ha='center', va='bottom')

    ax10.set_xlabel("Workflow Construction")
    ax10.set_ylabel("Total Data Volume (GB)")
    ax10.set_title("Total Workflow Data Volume Analysis")
    ax10.set_xticks(x)
    ax10.set_xticklabels(construction_labels, rotation=45)
    ax10.legend()
    ax10.grid(True)

    # 4. Performance vs Remote Write Efficiency
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
    ax1.set_ylabel("Remote Write Data per Event (MB)")
    ax1.set_title("Performance vs Remote Write Efficiency\n(size=CPU time, color=num groups)")
    ax1.grid(True)

    # set x-axis to start at 0 and add 10% padding to the right
    ax1.set_xlim(left=0, right=np.max(event_throughputs) * 1.1)

    # 5. Network Transfer Analysis
    ax7 = fig.add_subplot(gs[2, 0])
    network_transfer = []
    for metrics in construction_metrics:
        # Calculate network transfer as sum of remote read and local write data
        transfer = metrics["read_remote_per_event_mb"] + metrics["write_local_per_event_mb"]
        network_transfer.append(transfer)

    ax7.bar(range(len(construction_metrics)), network_transfer)
    ax7.set_xlabel("Workflow Construction")
    ax7.set_ylabel("Network Transfer (MB)")
    ax7.set_title("Network Transfer Analysis")
    ax7.set_xticks(range(len(construction_metrics)))
    ax7.set_xticklabels(construction_labels, rotation=45)
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

    ax4.boxplot(cpu_utilization, tick_labels=construction_labels)
    ax4.set_xlabel("Workflow Construction")
    ax4.set_ylabel("CPU Utilization Ratio")
    ax4.set_title("CPU Utilization Analysis\n(Actual CPU Usage / Allocated CPU)")
    ax4.set_xticklabels(construction_labels, rotation=45)
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
    ax6.set_xticklabels(construction_labels, rotation=45)
    ax6.grid(True)

    # 8. Event Processing Analysis
    ax5 = fig.add_subplot(gs[3, 1])
    events_per_group = []
    for metrics in construction_metrics:
        events = [group["total_events"] for group in metrics["group_details"]]
        events_per_group.append(events)

    ax5.boxplot(events_per_group, tick_labels=construction_labels)
    ax5.set_xlabel("Workflow Construction")
    ax5.set_ylabel("Events per Group")
    ax5.set_title("Event Processing Distribution")
    ax5.set_xticklabels(construction_labels, rotation=45)
    ax5.grid(True)



    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "workflow_comparison.png"))
    plt.close()

    # Create a detailed comparison table
    with open(os.path.join(output_dir, "workflow_comparison.txt"), "w") as f:
        f.write("Workflow Construction Comparison\n")
        f.write("==============================\n\n")

        for i, metrics in enumerate(construction_metrics, 1):
            # Use custom label if provided, otherwise use default "Construction" label
            if custom_labels and i <= len(custom_labels):
                construction_label = custom_labels[i-1]
            else:
                construction_label = f"Construction {i}"
            f.write(f"{construction_label}:\n")
            f.write(f"  Groups: {metrics['groups']}\n")
            f.write(f"  Number of Groups: {metrics['num_groups']}\n")
            f.write(f"  Event Throughput: {metrics['event_throughput']:.4f} events/second\n")
            f.write(f"  Total CPU Time: {metrics['total_cpu_time']:.2f} seconds\n")
            f.write("  Total Data Volumes for one job of each group:\n")
            f.write(f"    Remote Read Data: {metrics['total_read_remote_mb']:.2f} MB\n")
            f.write(f"    Local Write Data: {metrics['total_write_local_mb']:.2f} MB\n")
            f.write(f"    Remote Write Data: {metrics['total_write_remote_mb']:.2f} MB\n")
            f.write("  Data Flow Metrics (per event):\n")
            f.write(f"    Remote Read Data: {metrics['read_remote_per_event_mb']:.3f} MB/event\n")
            f.write(f"    Local Write Data: {metrics['write_local_per_event_mb']:.3f} MB/event\n")
            f.write(f"    Remote Write Data: {metrics['write_remote_per_event_mb']:.3f} MB/event\n")
            f.write(f"  Memory Utilization: {memory_utilization[i-1]:.2f}\n")
            f.write(f"  Network Transfer: {network_transfer[i-1]:.2f} MB\n")
           # f.write(f"  Estimated Cost: ${costs[i-1]:.2f}\n")
            f.write("  Workflow Performance Metrics:\n")
            f.write(f"    Total CPU Time: {metrics['total_cpu_time']:.2f} seconds\n")
            f.write(f"    Total Wallclock Time: {metrics['total_wallclock_time']:.2f} seconds\n")
            f.write(f"    Total Memory: {metrics['total_memory_mb']:,.0f} MB\n")
            f.write(f"    Total Network Transfer: {metrics['total_network_transfer_mb']:,.0f} MB\n")
            f.write("  Group Details:\n")
            for group in metrics["group_details"]:
                f.write(f"    {group['group_id']}:\n")
                f.write(f"      Tasks: {group['tasks']}\n")
                f.write(f"      Events per Task: {group['events_per_task']}\n")
                f.write(f"      CPU Time: {group['cpu_seconds']:.2f} seconds\n")
                f.write("      Data Flow (per event):\n")
                f.write(f"        Remote Read: {group['read_remote_per_event_mb']:.3f} MB/event\n")
                f.write(f"        Local Write: {group['write_local_per_event_mb']:.3f} MB/event\n")
                f.write(f"        Remote Write: {group['write_remote_per_event_mb']:.3f} MB/event\n")
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

    # Define custom labels for toy model
    custom_labels = ["Grouped", "Separated"]

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
    plot_group_data_volume_analysis(toy_constructions, str(output_path), custom_labels)

    # 2. Workflow construction comparison
    plot_workflow_comparison(toy_constructions, str(output_path), custom_labels)

    # 3. Storage efficiency analysis
    plot_storage_efficiency(toy_constructions, str(output_path))

    # 4. Job scaling analysis
    plot_job_scaling_analysis(toy_constructions, str(output_path), custom_labels)

    # 4. Workflow constructions overview
    plot_workflow_constructions(toy_constructions, str(output_path), custom_labels)

    # 5. Time analysis (dedicated plots)
    plot_time_analysis(toy_constructions, str(output_path), custom_labels)

    # 6. Resource utilization analysis (using filtered groups)
    plot_resource_utilization(toy_groups, str(output_path))

    # 7. Throughput analysis (using filtered groups)
    plot_throughput_analysis(toy_groups, str(output_path))

    # 8. Dependency analysis (using filtered groups)
    plot_dependency_analysis(toy_groups, str(output_path))

    # 9. Workflow topology visualization
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
    plot_job_scaling_analysis(construction_metrics, output_path)  # New job scaling analysis
    plot_time_analysis(construction_metrics, output_path)  # New dedicated time analysis
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

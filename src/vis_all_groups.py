import argparse
import json
import os
from typing import List, Dict

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
    fig.suptitle("Storage Efficiency Analysis", fontsize=16)

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
    sns.scatterplot(x=total_events, y=total_stored_data,
                   size=num_groups, sizes=(50, 200), ax=axes[1, 0])
    axes[1, 0].set_title("Total Stored Data vs Total Events\n(size indicates number of groups)")
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
    sns.scatterplot(x=total_cpu_times, y=total_events,
                   size=stored_data_per_event, sizes=(50, 200), ax=axes[0, 1])
    axes[0, 1].set_title("Total Events vs Total CPU Time\n(size indicates stored data per event)")
    axes[0, 1].set_xlabel("Total CPU Time (seconds)")
    axes[0, 1].set_ylabel("Total Events")
    axes[0, 1].grid(True)

    # Plot 3: Event Throughput Distribution
    sns.histplot(event_throughputs, bins=20, ax=axes[1, 0])
    axes[1, 0].set_title("Event Throughput Distribution")
    axes[1, 0].set_xlabel("Events per Second")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].grid(True)

    # Plot 4: Stored Data per Event vs Event Throughput
    sns.scatterplot(x=event_throughputs, y=stored_data_per_event,
                   size=num_groups, sizes=(50, 200), ax=axes[1, 1])
    axes[1, 1].set_title("Storage Efficiency vs Event Throughput\n(size indicates number of groups)")
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

    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(5, 2)

    # 1. Group Size Distribution (moved to top)
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
    ax3.grid(True)
    ax3.set_ylim(bottom=0)  # Set y-axis to start at 0

    # 2. Group-level Data Volume Analysis
    ax11 = fig.add_subplot(gs[0, 1])

    # Calculate positions for each group in each construction
    group_positions = []
    group_labels = []
    current_pos = 0

    for i, metrics in enumerate(construction_metrics):
        groups_in_construction = len(metrics["group_details"])
        # Create positions for this construction's groups
        positions = np.arange(current_pos, current_pos + groups_in_construction)
        group_positions.extend(positions)
        # Create labels for each group
        group_labels.extend([f"Const {i+1}\nGroup {j+1}" for j in range(groups_in_construction)])
        current_pos += groups_in_construction + 1  # Add 1 for spacing between constructions

    # Plot stacked bars for each group
    width = 0.8
    bottom = np.zeros(len(group_positions))

    # Plot input data
    input_data = [group["input_data_mb"] for metrics in construction_metrics
                 for group in metrics["group_details"]]
    ax11.bar(group_positions, input_data, width, label='Input Data', bottom=bottom)
    bottom += input_data

    # Plot output data
    output_data = [group["output_data_mb"] for metrics in construction_metrics
                  for group in metrics["group_details"]]
    ax11.bar(group_positions, output_data, width, label='Output Data', bottom=bottom)
    bottom += output_data

    # Plot stored data
    stored_data = [group["stored_data_mb"] for metrics in construction_metrics
                  for group in metrics["group_details"]]
    ax11.bar(group_positions, stored_data, width, label='Stored Data', bottom=bottom)

    # Add vertical lines to separate constructions
    for i in range(len(construction_metrics) - 1):
        sep_pos = group_positions[sum(len(metrics["group_details"])
                                    for metrics in construction_metrics[:i+1])] + 0.5
        ax11.axvline(x=sep_pos, color='gray', linestyle='--', alpha=0.5)

    ax11.set_xlabel("Workflow Construction and Groups")
    ax11.set_ylabel("Total Data Volume (MB)")
    ax11.set_title("Group-level Data Volume Analysis\n(Data Volumes per Group)")
    ax11.set_xticks(group_positions)
    ax11.set_xticklabels(group_labels, rotation=45, ha='right')
    ax11.legend()
    ax11.grid(True)
    ax11.set_ylim(bottom=0)  # Start y-axis at 0


    # 3. Data Flow Analysis (Updated to use per-event metrics)
    ax2 = fig.add_subplot(gs[1, 0])
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

    # 4. Total Data Volume Analysis (Stacked Bar)
    ax10 = fig.add_subplot(gs[1, 1])
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

    # 5. Performance vs Storage Efficiency
    ax1 = fig.add_subplot(gs[2, 0])

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

    # 6. Network Transfer Analysis
    ax7 = fig.add_subplot(gs[2, 1])
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

    # 7. CPU Utilization Analysis
    ax4 = fig.add_subplot(gs[3, 0])
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
    ax4.grid(True)

    # 8. Memory Utilization Analysis
    ax6 = fig.add_subplot(gs[3, 1])
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

    # 9. Event Processing Analysis
    ax5 = fig.add_subplot(gs[4, 0])
    events_per_group = []
    for metrics in construction_metrics:
        events = [group["total_events"] for group in metrics["group_details"]]
        events_per_group.append(events)

    ax5.boxplot(events_per_group, tick_labels=[f"Const {i+1}" for i in range(len(construction_metrics))])
    ax5.set_xlabel("Workflow Construction")
    ax5.set_ylabel("Events per Group")
    ax5.set_title("Event Processing Distribution")
    ax5.grid(True)

    # 10. Parallelism Analysis
    ax9 = fig.add_subplot(gs[4, 1])
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
    parser.add_argument('--template-file', type=str, help='Path to the template JSON file')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Base output directory (default: output)')
    args = parser.parse_args()

    # Convert template file path to Path object for easier manipulation
    template_path = Path(args.template_file)

    print(f"Parsing workflow data from template: {template_path}")
    with open(template_path) as f:
        workflow_data = json.load(f)

    groups, tasks, construction_metrics, dag = create_workflow_from_json(workflow_data)
    visualize_groups(groups, construction_metrics, template_path, args.output_dir, dag)

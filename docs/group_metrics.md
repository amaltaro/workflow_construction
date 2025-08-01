# Group Metrics Documentation

This document describes the comprehensive metrics calculated for individual task groups in the workflow construction analysis system. These metrics provide insights into resource utilization, performance characteristics, and data flow patterns for each group of tasks.

## Overview

Group metrics analyze the performance and resource characteristics of individual task groups within workflow constructions. Each group represents a collection of tasks that can be executed together as a single unit, and the metrics help understand the efficiency and characteristics of these groupings.

## Quick Reference

### Core Group Metrics

| Metric Category | Key Metrics | Description |
|----------------|-------------|-------------|
| **CPU** | `max_cores`, `cpu_seconds`, `utilization_ratio` | CPU allocation and efficiency |
| **Memory** | `max_mb`, `min_mb`, `occupancy` | Memory requirements and usage |
| **Throughput** | `total_eps`, `max_eps`, `min_eps` | Event processing rates |
| **I/O** | `input_data_mb`, `output_data_mb`, `stored_data_mb` | Data volumes and storage |
| **Utilization** | `resource_utilization`, `event_throughput` | Overall efficiency metrics |

### Example Group Metrics

```json
{
  "group_id": "group_4",
  "task_ids": ["Taskset2", "Taskset3"],
  "entry_point_task": "Taskset2",
  "exit_point_task": "Taskset3",
  "events_per_job": 1440,
  "resource_metrics": {
    "cpu": {
      "max_cores": 2,
      "cpu_seconds": 86400,
      "utilization_ratio": 1.0
    },
    "memory": {
      "max_mb": 4000,
      "min_mb": 3000,
      "occupancy": 0.9166666666666666
    },
    "throughput": {
      "total_eps": 0.03333333333333333,
      "max_eps": 0.05,
      "min_eps": 0.025
    },
    "io": {
      "input_data_mb": 281.25,
      "output_data_mb": 492.1875,
      "stored_data_mb": 70.3125,
      "input_data_per_event_mb": 0.1953125,
      "output_data_per_event_mb": 0.1708984375,
      "stored_data_per_event_mb": 0.0244140625
    },
    "accelerator": {
      "types": []
    }
  },
  "utilization_metrics": {
    "resource_utilization": 0.9583333333333333,
    "event_throughput": 0.03333333333333333
  },
  "dependency_paths": [["Taskset2","Taskset3"]]
}
```

## Detailed Metric Calculations

### Events per Job Calculation (`events_per_job`)

**Description:** Number of events that enter - and exit - a taskset group.

**Formula:**
```
Target Wallclock Time = 12.0 hours = 43,200 seconds
Total Time per Event = Σ(task.time_per_event for all tasks in group)
Events per Job = max(1, Target Wallclock Time / Total Time per Event)
```

**Example:** 1440 events per job

### CPU Metrics

#### Maximum CPU Cores (`cpu.max_cores`)
**Description:** Maximum number of CPU cores across all tasks within a group.

**Formula:**
```
max_cpu_cores = max(task.cpu_cores for all tasks in group)
```

**Example:** 2 cores

#### CPU Seconds (`cpu.cpu_seconds`)
**Description:** How long CPUs are allocated for a grid job instance of a taskset group.

**Formula:**
```
total_wallclock_time = events_per_job × total_time_per_event
cpu_seconds = max_cpu_cores × total_wallclock_time
```

**Example:** 86400 seconds

#### CPU Utilization Ratio (`cpu.utilization_ratio`)
**Description:** How well CPUs are utilized for a grid job instance of a taskset group (ratio of allocated versus used over time).

**Formula:**
```
For each task t:
    task_duration = events_per_job × task.time_per_event
    weighted_cpu_utilization += task.cpu_cores × task_duration

total_duration = Σ(task_duration for all tasks)
max_possible_utilization = max_cpu_cores × total_duration
cpu_utilization_ratio = weighted_cpu_utilization / max_possible_utilization
```

**Example:** 1.0 (100% utilization)

### Memory Metrics

#### Memory Range (`memory.min_mb` and `memory.max_mb`)
**Description:** Minimum and maximum memory required in all tasks present in a taskset group.

**Formula:**
```
max_memory_mb = max(task.memory_mb for all tasks in group)
min_memory_mb = min(task.memory_mb for all tasks in group)
```

**Example:** min_mb: 3000, max_mb: 4000

#### Memory Occupancy (`memory.occupancy`)
**Description:** Time-weighted average memory usage relative to maximum allocated memory.

**Formula:**
```
For each task t:
    task_duration = events_per_job × task.time_per_event
    weighted_memory += task.memory_mb × task_duration

total_duration = Σ(task_duration for all tasks)
time_weighted_avg_memory = weighted_memory / total_duration
memory_occupancy = time_weighted_avg_memory / max_memory_mb
```

**Example:** 0.9166666666666666 (91.67% occupancy)

### Throughput Metrics

#### Total Throughput (`throughput.total_eps`)
**Description:** Events processed through the entire group (i.e., how long it takes to process 1 event from the first to the last task in a group).

**Formula:**
```
total_throughput = events_per_job / cpu_seconds
```

**Example:** 0.03333333333333333 events/second

#### Individual Task Throughput (`throughput.min_eps` and `throughput.max_eps`)
**Description:** Tasksets in the group that provide the minimum and the largest event throughput, as a function of task cpu_seconds.

**Formula:**
```
For each task t:
    task_cpu_seconds = task.cpu_cores × task.time_per_event × events_per_job
    task_throughput = task.input_events / task_cpu_seconds

max_throughput = max(task_throughput for all tasks)
min_throughput = min(task_throughput for all tasks)
```

**Example:** min_eps: 0.025, max_eps: 0.05

### I/O Metrics with Storage Rules

#### Input Data Volume (`io.input_data_mb`)
**Description:** Data volume of input data, in megabytes, that is read from the shared storage by the entry point taskset.

**Formula:**
```
If entry_point_task has a parent task:
    parent_task = tasks[entry_point_task.input_task]
    input_data_mb = (events_per_job × parent_task.size_per_event) / 1024.0
Else:
    input_data_mb = 0.0

input_data_per_event_mb = input_data_mb / events_per_job
```

**Example:** 281.25 MB input data

#### Output Data Volume (`io.output_data_mb`)
**Description:** Data volume of output data, in megabytes, that all tasks in the group write to the local storage. For the output data per event, it keeps consistency with throughput, hence it is given for events per job of the whole group.

**Formula:**
```
For each task t:
    task_output_size = (events_per_job × task.size_per_event) / 1024.0
    output_data_mb += task_output_size

output_data_per_event_mb = output_data_mb / events_per_job
```

**Example:** 492.1875 MB output data

#### Stored Data Volume (`io.stored_data_mb`)
**Description:** Data volume of stored data, in megabytes, that all tasks in the group write to the shared storage. For the stored data per event, to keep it consistent with the throughput calculation, we calculate it for events per job of the whole group.

**Formula:**
```
For each task t:
    task_output_size = (events_per_job × task.size_per_event) / 1024.0
    If task.keep_output == True OR task.id == exit_point_task:
        stored_data_mb += task_output_size

stored_data_per_event_mb = stored_data_mb / events_per_job
```

**Example:** 70.3125 MB stored data

### Resource Utilization Metrics

#### Overall Resource Utilization (`utilization_metrics.resource_utilization`)
**Description:** This is a weighted average of CPU utilization ratio and memory occupancy.

**Formula:**
```
resource_utilization = (cpu_utilization_ratio + memory_occupancy) / 2.0
```

**Example:** 0.9583333333333333 (95.83% overall utilization)

#### Event Throughput (`utilization_metrics.event_throughput`)
**Description:** This is the number of events processed per second (events_per_job / cpu_seconds).

**Formula:**
```
event_throughput = total_throughput
```

**Example:** 0.03333333333333333 events/second

### Dependency Analysis

#### Dependency Paths (`dependency_paths`)
**Description:** All dependency relationships between tasks in the group.

**Formula:**
```
For each pair of tasks (src, dst) in group:
    If src ≠ dst AND there exists a path from src to dst in DAG:
        Add all simple paths from src to dst to dependency_paths
```

**Example:** `[["Taskset2","Taskset3"]]`

### Accelerator Analysis

#### Accelerator Types (`accelerator.types`)
**Description:** Types of accelerators (GPUs, etc.) required by tasks in the group.

**Formula:**
```
accelerator_types = {task.accelerator for task in group if task.accelerator is not None}
```

**Example:** `[]` (no accelerators required)

## Group Configuration

### Core Group Properties

- **`group_id`**: Unique identifier for the group
- **`task_ids`**: List of task IDs in the group
- **`entry_point_task`**: First task to execute in the group
- **`exit_point_task`**: Last task to execute in the group
- **`events_per_job`**: Number of events each task processes (based on target wallclock time)

### Example Group Configuration

```json
{
  "group_id": "group_4",
  "task_ids": ["Taskset2", "Taskset3"],
  "entry_point_task": "Taskset2",
  "exit_point_task": "Taskset3",
  "events_per_job": 1440
}
```

## Scoring System (Future Implementation)

Groups are evaluated using a weighted scoring system (`GroupScore` class) that considers:

- **CPU Score**: Efficiency of CPU core utilization
- **Memory Score**: Compatibility of memory requirements
- **Throughput Score**: Alignment of processing speeds
- **Accelerator Score**: GPU requirement compatibility

Each aspect can be weighted differently using customizable weights, with scores normalized to [0,1].

## Usage Example

```python
from find_all_groups import create_workflow_from_json

# Load workflow specification
with open("workflow.json", "r") as file:
    workflow_data = json.load(file)

# Generate all possible groups and workflow constructions
groups, tasks, construction_metrics, dag = create_workflow_from_json(workflow_data)

# Each group contains detailed metrics:
{
    "group_id": "group_0",
    "task_ids": ["Task1", "Task2", ...],
    "resource_metrics": {
        "cpu": {
            "max_cores": 4,
            "cpu_seconds": 1200.0,
            "utilization_ratio": 0.75
        },
        "memory": {
            "max_mb": 8000,
            "min_mb": 4000,
            "occupancy": 0.6
        },
        "io": {
            "input_data_mb": 100.0,
            "output_data_mb": 200.0,
            "stored_data_mb": 150.0,
            "input_data_per_event_mb": 0.1,
            "output_data_per_event_mb": 0.2,
            "stored_data_per_event_mb": 0.15
        }
        # ... other metrics
    },
    "events_per_job": 1440
}
```

## Analysis Workflow

1. **Workflow Parsing**:
   - Loads workflow specification from JSON
   - Creates task and resource models
   - Builds dependency graph
   - Calculates events per job based on target wallclock time

2. **Group Generation**:
   - Generates all possible valid task groups
   - Validates dependency constraints
   - Calculates metrics for each group including storage rules

3. **Metrics Calculation**:
   - Resource utilization (CPU, memory)
   - Throughput analysis
   - I/O requirements with storage rules
   - Dependency patterns
   - Parallel execution efficiency

4. **Output Generation**:
   - Returns detailed metrics for all groups
   - Provides data for visualization
   - Enables comparison of different grouping strategies

## Key Insights

1. **Resource Efficiency**: Groups with higher resource utilization ratios are more efficient
2. **Throughput Balance**: Groups should balance individual task throughputs for optimal performance
3. **Memory Management**: Memory occupancy indicates how well memory is utilized over time
4. **Data Flow**: I/O metrics help understand data transfer requirements and storage needs
5. **Dependency Patterns**: Dependency paths reveal the complexity of task relationships within groups

## Integration with Visualization

The group metrics work seamlessly with the visualization tool (`vis_all_groups.py`):
```python
from find_all_groups import create_workflow_from_json
from vis_all_groups import visualize_groups

# Generate groups and visualize
groups, tasks, construction_metrics, dag = create_workflow_from_json(workflow_data)
visualize_groups(groups, construction_metrics, template_path, output_dir="output", dag=dag)
```

This integration allows for:
- Visual analysis of different grouping strategies
- Comparison of resource utilization patterns
- Identification of optimal group sizes
- Understanding of tradeoffs between different metrics
- Analysis of workflow construction alternatives
- Evaluation of parallel execution potential
# Workflow Grouping Composition

A Python module for intelligently grouping workflow tasks based on their resource requirements and dependencies. This tool analyzes task characteristics and dependencies to optimize workflow execution by grouping compatible tasks together.

Research for: Optimizing Heterogeneous Workflow Construction for Enhanced Event Throughput and Efficient Resource Utilization in CMS


## Overview

The Workflow Task Grouper analyzes workflow tasks and their relationships to create optimal task groups while respecting both hard constraints (OS compatibility, architecture requirements) and soft constraints (resource utilization, performance characteristics). The system also generates all possible workflow constructions and provides comprehensive analysis and visualization capabilities.

For a realistic 5 steps workflow configuration, under the campaign `Run3Summer23wmLHEGS`, please see [this](https://cmsweb.cern.ch/reqmgr2/fetch?rid=cmsunified_task_HIG-Run3Summer23wmLHEGS-00565__v1_T_250718_210653_3406) workflow.


## Features

- Task dependency analysis using directed acyclic graphs (DAG)
- Resource compatibility checking
- Configurable scoring system for group optimization
- Support for various resource types (CPU, Memory, GPU)
- Visualization of workflow DAG structure (ASCII and Mermaid formats)
- Customizable weights for different scoring aspects
- Generation of all possible workflow constructions
- Comprehensive workflow construction analysis and comparison
- Mermaid-based HTML workflow topology visualization
- Storage rules and data volume calculation
- Events per job calculation based on target wallclock time


## Installation

```bash
# Clone the repository
git clone <repository-url>
cd workflow-task-grouper

# Install dependencies
pip install -r requirements.txt
```


## Usage

```python
from src.group_tasks import create_workflow_from_json

# Load your workflow JSON
with open("workflow.json", "r") as file:
    workflow_data = json.load(file)

# Create groups with custom minimum score
groups, tasks = create_workflow_from_json(workflow_data, min_group_score=0.7)
```


## Testing

The project includes test suites for different workflow patterns:
- Sequential workflows (`tests/test_sequential.py`)
- Fork workflows (`tests/test_fork.py`)

Run tests using:
```bash
python -m pytest tests/
```


## Visualization

The module supports two visualization formats for workflow DAGs:

### ASCII Format

```
Task1
|-- Task2
    |-- Task3
    |-- Task4
```

### Mermaid Format
```mermaid
graph TD
    Task1 --> Task2
    Task2 --> Task3
    Task2 --> Task4
```


## Comprehensive Workflow Analysis Module

The `find_all_groups.py` module provides a comprehensive framework for analyzing all possible task groupings and workflow constructions in a workflow. This module generates and analyzes all valid task groupings based on dependency constraints and finds all possible ways to construct workflows from these groups.

### Key Features

- Generates all possible valid task groups based on dependency constraints
- Calculates detailed resource utilization metrics for each group
- Finds all possible workflow constructions (combinations of groups that cover all tasks)
- Calculates comprehensive workflow-level metrics
- Implements storage rules for data volume calculation
- Calculates optimal events per job based on target wallclock time
- Provides comprehensive analysis of CPU, memory, I/O, and dependency patterns
- Supports visualization of grouping strategies and their impacts

### Core Components

1. **Task and Resource Models**:
   ```python
   @dataclass
   class TaskResources:
       os_version: str
       cpu_arch: str
       memory_mb: int
       accelerator: Optional[str]
       cpu_cores: int
       events_per_second: float
       time_per_event: float
       size_per_event: float
       input_events: int
       keep_output: bool  # Whether the task's output needs to be kept in shared storage

   @dataclass
   class Task:
       id: str
       resources: TaskResources
       input_task: Optional[str] = None
       output_tasks: Set[str] = None
       order: int = 0  # Order of the task in the workflow
   ```

2. **Group Metrics Calculation**:
   - CPU utilization and efficiency
   - Memory allocation and efficiency
   - Throughput analysis
   - I/O requirements with storage rules
   - Dependency path analysis
   - Events per job calculation

3. **Workflow Construction Analysis**:
   - Finds all possible combinations of groups that cover all tasks
   - Respects task dependencies between groups
   - Calculates workflow-level metrics including:
     - Event throughput
     - Total data volumes
     - Resource utilization
     - Parallel execution efficiency

4. **Storage Rules**:
   - Input data volume based on parent task's size per event
   - Output data volume for all tasks in the group
   - Stored data volume for tasks with `keep_output=True` or exit point tasks
   - Per-event data volume calculations

5. **Events per Job Calculation**:
   - Based on target wallclock time (default: 12 hours)
   - Calculates optimal events per job for each group
   - Ensures consistent event processing across workflow


## 📊 Metrics Documentation

The workflow construction analysis system calculates comprehensive metrics for both individual task groups and complete workflow constructions. These metrics provide insights into performance, resource utilization, and data flow characteristics.

### Workflow Construction Metrics

**Core Metrics:**
- **Event Throughput**: Events processed per second through the entire workflow
- **Total Events**: Total events processed across all groups
- **CPU Time**: Total CPU time required across all groups
- **Data Volumes**: Input, output, and stored data totals

**Key Insights:**
- Output data per event is constant across all constructions
- Throughput generally decreases with more groups due to overhead
  - However, more groups are meant to improve CPU utilization and Memory occupancy
- Stored data per event varies based on grouping strategy and exit points

📖 **[Detailed Workflow Construction Metrics Documentation](docs/workflow_construction_metrics.md)**

### Group Metrics

**Core Metrics:**
- **Resource Utilization**: CPU and memory efficiency ratios
- **Event Throughput**: Events processed per second per group
- **I/O Analysis**: Data volumes and storage requirements
- **Dependency Analysis**: Task relationships and execution paths

📖 **[Detailed Group Metrics Documentation](docs/group_metrics.md)**

### Usage in Analysis

These metrics enable:
- **Performance Comparison**: Compare throughput across different grouping strategies
- **Resource Planning**: Estimate CPU time and data storage requirements
- **Cost Analysis**: Calculate resource costs based on CPU time and storage
- **Optimization**: Identify optimal grouping strategies for specific requirements

### Example Data

All examples in the documentation are based on real data from `output/fork/5tasks/construction_metrics.json`, providing concrete values for understanding the metrics in practice.


## Metrics Analysis and Visualization

The module includes comprehensive metrics analysis and visualization capabilities through `vis_all_groups.py`. This tool generates detailed visualizations to analyze different aspects of task grouping strategies and workflow constructions.

### Installation

```bash
pip install matplotlib seaborn pandas
```

### Available Visualizations

1. **Resource Utilization Analysis** (`resource_utilization.png`):
   - CPU Efficiency vs Group Size
   - Memory Efficiency vs Group Size
   - Overall Resource Utilization vs Group Size
   - CPU vs Memory Efficiency (with group size as bubble size)

2. **Throughput and I/O Analysis** (`throughput_analysis.png`):
   - Total Throughput vs Group Size
   - Throughput Range (Min/Max) vs Group Size
   - Total Output Size vs Group Size
   - Throughput vs Output Size (with group size as bubble size)

3. **Dependency Analysis** (`dependency_analysis.png`):
   - Number of Dependency Paths vs Group Size
   - Average Path Length vs Group Size

4. **Metric Correlation Heatmap** (`metric_correlation.png`):
   - Shows correlations between all metrics
   - Helps identify relationships between different aspects of the grouping

5. **Workflow Construction Analysis** (`workflow_comparison.png`):
   - Group Size Distribution: Shows the distribution of tasks per group in each construction
   - Performance vs Storage Efficiency: Scatter plot showing the trade-off between event throughput and stored data per event
   - Data Flow Analysis: Bar plot comparing input, output, and stored data volumes across constructions
   - Network Transfer Analysis: Shows the total data transferred between groups in each construction
   - CPU Utilization Analysis: Box plot showing the distribution of CPU utilization ratios across groups
   - Memory Utilization Analysis: Bar plot with error bars showing average memory occupancy and its variation
   - Event Processing Distribution: Box plot showing the distribution of events processed per group
   - Parallel Execution Analysis: Shows the parallel efficiency of each construction based on DAG dependencies

6. **Storage Efficiency Analysis** (`storage_efficiency.png`):
   - Storage Efficiency vs Number of Groups
   - Storage Efficiency vs Event Throughput
   - Total Stored Data vs Total Events
   - Stored Data per Event Distribution

7. **Workflow Constructions Analysis** (`workflow_constructions.png`):
   - Event Throughput vs Number of Groups
   - Total Events vs Total CPU Time
   - Event Throughput Distribution
   - Storage Efficiency vs Event Throughput

8. **Workflow Topology** (`workflow_topologies.html`):
   - Interactive HTML visualization with Mermaid diagrams
   - Visual representation of the DAG structure for each construction
   - Shows task dependencies and grouping relationships
   - Color-coded groups with legend
   - Side-by-side comparison of all constructions

Each visualization provides unique insights into different aspects of the workflow constructions:
- Resource utilization and efficiency
- Data flow and storage patterns
- Parallel execution potential
- Performance characteristics
- Group size distributions
- Network transfer requirements

The visualizations help in:
1. Comparing different workflow construction strategies
2. Identifying optimal group sizes and compositions
3. Understanding resource utilization patterns
4. Analyzing data flow and storage requirements
5. Evaluating parallel execution potential
6. Making informed decisions about workflow optimization

### Metrics Tracked

The visualization tool tracks and analyzes comprehensive metrics for both individual groups and workflow constructions:

#### **Group-Level Metrics**

**CPU Metrics**
- **`max_cores`**: Maximum CPU cores allocated to the group
- **`cpu_seconds`**: Total CPU time required (cores × wallclock time)
- **`utilization_ratio`**: Actual CPU usage / allocated CPU (efficiency metric)

**Memory Metrics**
- **`max_mb`/`min_mb`**: Maximum/minimum memory requirements across tasks (in megabytes)
- **`occupancy`**: Time-weighted average memory usage / max memory allocated

**Throughput Metrics**
- **`total_eps`**: Group-level events per second (total events / total CPU time)
- **`max_eps`**: Highest events/second achieved by any single task in the group
- **`min_eps`**: Lowest events/second achieved by any single task in the group
- **Note**: For single-task groups, all three throughput values are identical

**I/O Metrics**
- **`input_data_mb`**: Total input data volume for the group
- **`output_data_mb`**: Total output data volume from all tasks
- **`stored_data_mb`**: Data volume that needs persistent storage
- **`*_per_event_mb`**: Per-event data volumes for each category

**Resource Utilization Metrics**
- **`resource_utilization`**: Overall efficiency (average of CPU and memory utilization)
- **`event_throughput`**: Events processed per second at group level

**Group Configuration**
- **`events_per_job`**: Number of events each task processes (based on target wallclock time)
- **`entry_point_task`**: First task to execute in the group
- **`exit_point_task`**: Last task to execute in the group
- **`dependency_paths`**: All dependency relationships between tasks in the group

#### **Workflow Construction Metrics**

For each workflow construction, the following metrics are calculated and analyzed:

1. **Group Composition Metrics**
   - Number of groups in the construction
   - Tasks per group distribution
   - Group size statistics (min, max, average)

2. **Performance Metrics**
   - Event throughput (events processed per second)
   - Total CPU time across all groups
   - Parallel execution efficiency (ratio of sequential to parallel execution time)
   - Critical path length (longest path in terms of CPU time)

3. **Data Flow Metrics**
   - Total input data volume (MB)
   - Total output data volume (MB)
   - Total stored data volume (MB)
   - Stored data per event ratio
   - Network transfer volume (sum of input and output data)

4. **Resource Utilization Metrics**
   - CPU utilization ratio per group (actual CPU usage / allocated CPU)
   - Memory occupancy per group (actual memory usage / allocated memory)
   - Average resource utilization across groups
   - Standard deviation of resource utilization

5. **Event Processing Metrics**
   - Events per group distribution
   - Total events processed
   - Events per task statistics
   - Event processing efficiency

6. **Storage Efficiency Metrics**
   - Storage efficiency (stored data per event)
   - Storage utilization ratio
   - Data retention patterns

7. **Network Transfer Metrics**
   - Total data transferred between groups
   - Input/output data ratios
   - Network transfer efficiency

8. **Parallel Execution Metrics**
   - Maximum parallel groups possible
   - Sequential execution time
   - Parallel execution time
   - Parallel efficiency ratio

These metrics help in:
1. Evaluating the efficiency of different workflow constructions
2. Identifying bottlenecks in resource utilization
3. Understanding data flow patterns
4. Optimizing group compositions
5. Balancing performance and resource usage
6. Making informed decisions about workflow optimization

### Usage

```python
from vis_all_groups import visualize_groups

# After getting groups from create_workflow_from_json
visualize_groups(groups, construction_metrics, template_path, output_dir="output", dag=dag)
```

Or run directly from command line:
```bash
# Basic usage with default output directory
python src/vis_all_groups.py tests/sequential/3tasks.json

# With a custom output directory
python src/vis_all_groups.py tests/sequential/3tasks.json --output-dir custom_output

# Toy model mode - visualize only two extreme constructions
python src/vis_all_groups.py tests/sequential/5tasks.json --toy-model --output-dir output/toy_model
```

### Toy Model Mode

The `--toy-model` flag enables a simplified analysis that focuses on two extreme workflow construction strategies:

1. **Grouped Strategy**: All tasks in a single group (minimum number of groups)
2. **Separated Strategy**: One group per task (maximum number of groups)

This mode is useful for:
- Quick comparison of extreme grouping strategies
- Understanding the trade-offs between grouping and separation
- Simplified analysis for educational or demonstration purposes
- Focused visualization of key metrics without overwhelming detail

The toy model generates the same comprehensive visualizations but only for these two extreme cases, making it easier to understand the fundamental differences between grouped and separated approaches.

The script will:
1. Automatically detect the template directory name (e.g., "sequential" or "fork") from the input file path
2. Create the appropriate output directory structure (e.g., `output/sequential/3tasks/` or `custom_output/sequential/3tasks/`)
3. Save all visualizations and data files in that directory
4. Generate an interactive HTML file with Mermaid diagrams for workflow topology visualization

You can get help on the command line arguments by running:
```bash
python src/vis_all_groups.py --help
```

The tool saves both the visualizations and the raw metrics data in JSON format for further analysis.

### Analysis Benefits

The visualizations help:
1. Identify optimal group sizes for different metrics
2. Understand tradeoffs between resource utilization and throughput
3. Analyze dependency patterns
4. Find correlations between different metrics
5. Make informed decisions about task grouping strategies
6. Compare different workflow construction approaches
7. Evaluate storage and network transfer requirements

### Integration with Visualization

The module works seamlessly with the visualization tool (`vis_all_groups.py`):
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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

# Workflow Construction Metrics Documentation

This document describes the metrics calculated for workflow constructions, which represent different ways to organize tasks into groups for execution.

## Overview

Workflow construction metrics provide insights into the performance, resource utilization, and data flow characteristics of different task grouping strategies. Each construction represents a valid combination of task groups that covers all tasks in the workflow while respecting dependencies.

## Quick Reference

### Core Metrics

| Metric | Description | Formula | Example Value |
|--------|-------------|---------|---------------|
| `total_events` | Total events processed | `sum(group.events_per_job)` | 5348 |
| `event_throughput` | Events per second | `max_events_per_job / total_cpu_time_all_jobs` | 0.003125 |
| `total_cpu_time` | Total CPU seconds | `sum(group.cpu_seconds)` | 345560 |
| `num_groups` | Number of groups | `len(construction)` | 2 |

### Data Volume Metrics

| Metric | Description | Formula | Example Value |
|--------|-------------|---------|---------------|
| `total_input_data_mb` | Total input data | `sum(group.input_data_mb)` | 241.015625 |
| `total_output_data_mb` | Total output data | `sum(group.output_data_mb)` | 786.71875 |
| `total_stored_data_mb` | Total stored data | `sum(group.stored_data_mb)` | 190.8203125 |

### Per-Event Metrics

| Metric | Description | Formula | Example Value |
|--------|-------------|---------|---------------|
| `input_data_per_event_mb` | Input per event | `sum(group.input_data_per_event_mb)` | 0.1953125 |
| `output_data_per_event_mb` | Output per event | `sum(group.output_data_per_event_mb)` | 0.7080078125 |
| `stored_data_per_event_mb` | Stored per event | `sum(group.stored_data_per_event_mb)` | 0.1708984375 |

### Key Insights

- **Output data per event is constant** across all constructions (0.7080078125 MB/event)
- **Throughput decreases** with more groups due to overhead
- **Stored data per event varies** based on grouping strategy and exit points
- **Total events varies** based on events_per_job of each group

## Metric Categories

### 1. Event Processing Metrics

#### `total_events`
**Description:** Total number of events processed across all groups in the construction, considering one instance of each group (1 grid job for each group).

**Calculation:**
```python
total_events = sum(group.events_per_job for group in construction)
```

**Example:** 5348 events (from construction_metrics.json)

**Formula:** Sum of events per job for each group (each group processes events_per_job as a unit)

#### `event_throughput`
**Description:** Number of events processed per second through the entire workflow construction. This accounts for groups of different sizes (events_per_job) and potential bottlenecks imposed by them. So this calculation considers scaling factor for groups that might need multiple jobs to process the same number of events processed by different groups.

**Calculation:**
```python
# Find the maximum events_per_job across all groups (common baseline)
max_events_per_job = max(group.events_per_job for group in construction)

# Calculate total CPU time across all jobs needed
total_cpu_time_all_jobs = 0.0
for group in construction:
    jobs_needed = max_events_per_job / group.events_per_job
    total_cpu_time_all_jobs += group.cpu_seconds * jobs_needed

# Event throughput is the common number of events divided by total CPU time
event_throughput = max_events_per_job / total_cpu_time_all_jobs
```

**Example:** 0.003125 events/second

**Formula:** `max_events_per_job / total_cpu_time_all_jobs`

#### `total_cpu_time` (FIXME: how about calculating this based on the total number of events requested?)
**Description:** Total CPU time required across all groups in the construction, for a single instance of each group.

**Calculation:**
```python
total_cpu_time = sum(group.cpu_seconds for group in construction)
```

**Example:** 345560 seconds

**Formula:** Sum of CPU seconds for all groups

#### `initial_input_events`
**Description:** Number of events that enter the first group in the construction.

**Calculation:**
```python
initial_input_events = construction[0].events_per_job
```

**Example:** 960 events

**Formula:** Events per job of the first group in the construction

### 2. Data Volume Metrics

#### `total_input_data_mb` (FIXME: factor in overall requested events, as each group may process a different number of events)
**Description:** Total input data volume in MB across all groups, for a single instance of each group.

**Calculation:**
```python
total_input_data_mb = sum(group.input_data_mb for group in construction)
```

**Example:** 241.015625 MB

**Formula:** Sum of input data volumes from all groups

#### `total_output_data_mb` (FIXME)
**Description:** Total output data volume in MB across all groups, for a single instance of each group.

**Calculation:**
```python
total_output_data_mb = sum(group.output_data_mb for group in construction)
```

**Example:** 786.71875 MB

**Formula:** Sum of output data volumes from all groups

#### `total_stored_data_mb` (FIXME)
**Description:** Total data volume in MB that needs to be stored in shared storage, for a single instance of each group.

**Calculation:**
```python
total_stored_data_mb = sum(group.stored_data_mb for group in construction)
```

**Example:** 190.8203125 MB

**Formula:** Sum of stored data volumes from all groups

### 3. Per-Event Data Metrics

#### `input_data_per_event_mb`
**Description:** Input data volume per event for the entire workflow construction.

**Calculation:**
```python
input_data_per_event_mb = sum(group.input_data_per_event_mb for group in construction)
```

**Example:** 0.1953125 MB/event

**Formula:** Sum of per-event input data from all groups

#### `output_data_per_event_mb`
**Description:** Output data volume per event for the entire workflow construction. Note that this value is constant across all constructions because every event is processed by all tasks, generating the same total output volume regardless of grouping.

**Calculation:**
```python
output_data_per_event_mb = sum(group.output_data_per_event_mb for group in construction)
```

**Example:** 0.7080078125 MB/event

**Formula:** Sum of per-event output data from all groups

#### `stored_data_per_event_mb`
**Description:** Stored data volume per event for the entire workflow construction.

**Calculation:**
```python
stored_data_per_event_mb = sum(group.stored_data_per_event_mb for group in construction)
```

**Example:** 0.1708984375 MB/event

**Formula:** Sum of per-event stored data from all groups

### 4. Construction Metadata

#### `num_groups`
**Description:** Number of groups in the construction.

**Calculation:**
```python
num_groups = len(construction)
```

**Example:** 2 groups

**Formula:** Count of groups in the construction

#### `groups`
**Description:** List of group IDs in the construction.

**Calculation:**
```python
groups = [group.group_id for group in construction]
```

**Example:** `["group_10", "group_7"]`

**Formula:** Array of group identifiers

## Group Details

Each construction includes detailed information about each group:

### Group-Level Metrics

#### `group_id`
**Description:** Unique identifier for the group.

**Example:** `"group_10"`

#### `tasks`
**Description:** List of task IDs in the group.

**Example:** `["Taskset1", "Taskset3", "Taskset5"]`

#### `events_per_task`
**Description:** Number of events processed by each task in the group.

**Example:** 960 events

**Formula:** `group.events_per_job`

#### `total_events`
**Description:** Total events processed by the group (sum of all tasks).

**Example:** 2880 events

**Formula:** `group.events_per_job` (each group processes events_per_job as a unit)

#### `cpu_seconds`
**Description:** Total CPU time required for the group (sum of cpu_seconds for all tasks).

**Example:** 172800 seconds

**Formula:** `group.cpu_seconds`

#### `input_data_mb`
**Description:** Input data volume for the group.

**Example:** 0.0 MB

**Formula:** `group.input_data_mb`

#### `output_data_mb`
**Description:** Output data volume for the group.

**Example:** 304.6875 MB

**Formula:** `group.output_data_mb`

#### `stored_data_mb`
**Description:** Stored data volume for the group.

**Example:** 70.3125 MB

**Formula:** `group.stored_data_mb`

#### `input_data_per_event_mb`
**Description:** Input data volume per event for the group.

**Example:** 0.0 MB/event

**Formula:** `group.input_data_per_event_mb`

#### `output_data_per_event_mb`
**Description:** Output data volume per event for the group.

**Example:** 0.3173828125 MB/event

**Formula:** `group.output_data_per_event_mb`

#### `stored_data_per_event_mb`
**Description:** Stored data volume per event for the group.

**Example:** 0.0732421875 MB/event

**Formula:** `group.stored_data_per_event_mb`

## Example Construction Analysis

From the provided JSON file, here's an analysis of the first construction:

```json
{
  "total_events": 5348,
  "total_cpu_time": 345560,
  "event_throughput": 0.003125,
  "total_input_data_mb": 241.015625,
  "total_output_data_mb": 786.71875,
  "total_stored_data_mb": 190.8203125,
  "input_data_per_event_mb": 0.1953125,
  "output_data_per_event_mb": 0.7080078125,
  "stored_data_per_event_mb": 0.1708984375,
  "initial_input_events": 960,
  "num_groups": 2,
  "groups": ["group_10", "group_7"]
}
```

**Analysis:**
- **Construction Type:** 2-group construction
- **Performance:** 0.003125 events/second throughput
- **Data Flow:** Each event requires 0.195 MB input, generates 0.708 MB output, stores 0.171 MB
- **Resource Usage:** 345,560 CPU seconds total
- **Event Processing:** 5,348 total events processed

## Key Insights

1. **Throughput vs Groups:** Generally, fewer groups result in higher throughput due to reduced overhead
2. **Data Consistency:** Output data per event is constant across all constructions (0.7080078125 MB/event)
3. **Storage Variation:** Stored data per event varies based on grouping strategy and exit points
4. **Resource Trade-offs:** More groups may improve parallelism but increase overhead

## Usage in Analysis

These metrics enable:
- **Performance Comparison:** Compare throughput across different grouping strategies
- **Resource Planning:** Estimate CPU time and data storage requirements
- **Cost Analysis:** Calculate resource costs based on CPU time and storage
- **Optimization:** Identify optimal grouping strategies for specific requirements 
# Workflow Construction Metrics Documentation

This document describes the metrics calculated for workflow constructions, which represent different ways to organize tasks into groups for execution.

## Overview

Workflow construction metrics provide insights into the performance, resource utilization, and data flow characteristics of different task grouping strategies. Each construction represents a valid combination of task groups that covers all tasks in the workflow while respecting dependencies.

## Quick Reference

### Core Metrics

| Metric | Description | Formula | Example Value |
|--------|-------------|---------|---------------|
| `total_events` | Total events processed | `request_num_events` | 1000000 |
| `event_throughput` | Events per second | `request_num_events / total_cpu_time` | 0.0125 |
| `total_cpu_time` | Total CPU seconds (accounting for job scaling) | `sum(group.cpu_seconds * jobs_needed)` | 320000000 |
| `num_groups` | Number of groups | `len(construction)` | 2 |
| `cpu_time_per_event` | CPU time per event | `total_cpu_time / request_num_events` | 320.0 |

### Data Volume Metrics

| Metric | Description | Formula | Example Value |
|--------|-------------|---------|---------------|
| `total_read_remote_mb` | Total remote read data (accounting for job scaling) | `sum(group.read_remote_mb * jobs_needed)` | 195312.50 |
| `total_write_local_mb` | Total local write data (accounting for job scaling) | `sum(group.write_local_mb * jobs_needed)` | 537109.38 |
| `total_write_remote_mb` | Total remote write data (accounting for job scaling) | `sum(group.write_remote_mb * jobs_needed)` | 48828.13 |

### Per-Event Metrics

| Metric | Description | Formula | Example Value |
|--------|-------------|---------|---------------|
| `read_remote_per_event_mb` | Remote read per event | `total_read_remote_mb / request_num_events` | 0.1953125 |
| `write_local_per_event_mb` | Local write per event | `total_write_local_mb / request_num_events` | 0.537109375 |
| `write_remote_per_event_mb` | Remote write per event | `total_write_remote_mb / request_num_events` | 0.048828125 |

### Job Scaling Metrics

| Metric | Description | Formula | Example Value |
|--------|-------------|---------|---------------|
| `request_num_events` | Total events requested | From JSON template | 1000000 |
| `group_jobs_needed` | Jobs needed per group | `request_num_events / group.events_per_job` | {"group_0": 231.5, "group_1": 463.0} |

### Memory Scaling Metrics

| Metric | Description | Formula | Example Value |
|--------|-------------|---------|---------------|
| `total_memory_mb` | Total memory required (accounting for job scaling) | `sum(group.max_memory_mb * jobs_needed)` | 3703704 |
| `memory_per_event_mb` | Memory per event | `total_memory_mb / request_num_events` | 3.704 |

### Network Transfer Metrics

| Metric | Description | Formula | Example Value |
|--------|-------------|---------|---------------|
| `total_network_transfer_mb` | Total network transfer (remote read + remote write) | `total_read_remote_mb + total_write_remote_mb` | 244140 |
| `network_transfer_per_event_mb` | Network transfer per event | `total_network_transfer_mb / request_num_events` | 0.244 |

### Efficiency Metrics

| Metric | Description | Formula | Example Value |
|--------|-------------|---------|---------------|
| `total_jobs` | Total jobs needed | `sum(group_jobs_needed.values())` | 926 |
| `jobs_per_event` | Jobs per event (efficiency metric) | `total_jobs / request_num_events` | 0.000926 |

### Time Metrics

| Metric | Description | Formula | Example Value |
|--------|-------------|---------|---------------|
| `total_wallclock_time` | Total wallclock time (accounting for job scaling) | `sum(wallclock_time_per_job * jobs_needed)` | 40000000 |
| `wallclock_time_per_event` | Wallclock time per event | `total_wallclock_time / request_num_events` | 40.0 |

### Key Insights

- **Local write data per event is constant** across all constructions (example 0.537109375 MB/event)
- **Remote read/write data per event varies** based on grouping strategy and data flow patterns
- **Total data volumes account for job scaling** - groups that process fewer events per job need more jobs, increasing total data volumes
- **Throughput decreases** with more groups due to overhead
- **Total CPU time accounts for job scaling** - groups that process fewer events per job need more jobs to complete the workflow
- **CPU time per event provides fair comparison** across different workflow constructions
- **Job scaling reveals efficiency differences** - some constructions require more total jobs than others
- **Memory requirements scale with job scaling** - groups with higher memory requirements and more jobs needed will have higher total memory usage
- **Network transfer varies significantly** - constructions with more data flow between groups have higher network transfer requirements
- **Jobs per event is a key efficiency metric** - lower values indicate more efficient processing (fewer jobs needed per event)
- **Wallclock time is consistent** across constructions since all jobs target the same wallclock time (12 hours)

## Metric Categories

### 1. Event Processing Metrics

#### `total_events`
**Description:** Total number of events processed by the workflow construction (the requested number of events).

**Calculation:**
```python
total_events = request_num_events
```

**Example:** 1000000 events (from RequestNumEvents parameter)

**Formula:** Sum of events per job for each group (each group processes events_per_job as a unit)

#### `event_throughput`
**Description:** Number of events processed per second through the entire workflow construction. This provides a consistent measure of overall workflow performance, accounting for job scaling across all groups.

**Calculation:**
```python
event_throughput = request_num_events / total_cpu_time
```

**Example:** 0.0125 events/second

**Formula:** `request_num_events / total_cpu_time`

#### `total_cpu_time`
**Description:** Total workflow CPU time required across all groups in the construction, accounting for the number of jobs each group needs to run to process the requested number of events.

**Calculation:**
```python
# Calculate how many jobs each group needs to run
group_jobs_needed = {}
for group in construction:
    jobs_needed = request_num_events / group.events_per_job
    group_jobs_needed[group.group_id] = jobs_needed

# Calculate total CPU time for the entire workflow
total_cpu_time = sum(
    group.cpu_seconds * group_jobs_needed[group.group_id]
    for group in construction
)
```

**Example:** 320,000,000 seconds (for 1,000,000 requested events, accounting for job scaling)

**Formula:** `sum(group.cpu_seconds * jobs_needed)` where `jobs_needed = request_num_events / group.events_per_job`

#### `cpu_time_per_event`
**Description:** Normalized CPU time per event, calculated by dividing the total CPU time by the requested number of events.

**Calculation:**
```python
cpu_time_per_event = total_cpu_time / request_num_events
```

**Example:** 320.0 seconds/event (for 1,000,000 requested events)

**Formula:** `total_cpu_time / request_num_events`

#### `request_num_events`
**Description:** Total number of events requested to be processed by the workflow, specified in the JSON template.

**Example:** 1,000,000 events

**Source:** `RequestNumEvents` field in the JSON template

#### `group_jobs_needed`
**Description:** Dictionary mapping each group to the number of jobs it needs to run to process the requested number of events.

**Calculation:**
```python
group_jobs_needed = {}
for group in construction:
    jobs_needed = request_num_events / group.events_per_job
    group_jobs_needed[group.group_id] = jobs_needed
```

**Example:** `{"group_0": 231.5, "group_1": 463.0, "group_2": 231.5}`

**Formula:** `request_num_events / group.events_per_job` for each group

### 2. Data Volume Metrics

#### `total_read_remote_mb`
**Description:** Total remote read data volume across all groups in the construction, accounting for the number of jobs each group needs to run to process the requested number of events.

**Calculation:**
```python
total_read_remote = sum(
    group.read_remote_mb * group_jobs_needed[group.group_id]
    for group in construction
)
```

**Example:** 195,312.50 MB (for 1,000,000 requested events)

**Formula:** `sum(group.read_remote_mb * jobs_needed)` where `jobs_needed = request_num_events / group.events_per_job`

#### `total_write_local_mb`
**Description:** Total local write data volume across all groups in the construction, accounting for the number of jobs each group needs to run to process the requested number of events.

**Calculation:**
```python
total_write_local = sum(
    group.write_local_mb * group_jobs_needed[group.group_id]
    for group in construction
)
```

**Example:** 537,109.38 MB (for 1,000,000 requested events)

**Formula:** `sum(group.write_local_mb * jobs_needed)` where `jobs_needed = request_num_events / group.events_per_job`

#### `total_write_remote_mb`
**Description:** Total remote write data volume across all groups in the construction, accounting for the number of jobs each group needs to run to process the requested number of events.

**Calculation:**
```python
total_write_remote = sum(
    group.write_remote_mb * group_jobs_needed[group.group_id]
    for group in construction
)
```

**Example:** 48,828.13 MB (for 1,000,000 requested events)

**Formula:** `sum(group.write_remote_mb * jobs_needed)` where `jobs_needed = request_num_events / group.events_per_job`

### 3. Per-Event Data Metrics

#### `read_remote_per_event_mb`
**Description:** Normalized remote read data per event, calculated by dividing the total remote read data by the requested number of events.

**Calculation:**
```python
read_remote_per_event = total_read_remote / request_num_events
```

**Example:** 0.1953125 MB/event (for 1,000,000 requested events)

**Formula:** `total_read_remote_mb / request_num_events`

#### `write_local_per_event_mb`
**Description:** Normalized local write data per event, calculated by dividing the total local write data by the requested number of events.

**Calculation:**
```python
write_local_per_event = total_write_local / request_num_events
```

**Example:** 0.537109375 MB/event (for 1,000,000 requested events)

**Formula:** `total_write_local_mb / request_num_events`

#### `write_remote_per_event_mb`
**Description:** Normalized remote write data per event, calculated by dividing the total remote write data by the requested number of events.

**Calculation:**
```python
write_remote_per_event = total_write_remote / request_num_events
```

**Example:** 0.048828125 MB/event (for 1,000,000 requested events)

**Formula:** `total_write_remote_mb / request_num_events`

#### `initial_input_events`
**Description:** Number of events that enter the first group in the construction.

**Calculation:**
```python
initial_input_events = construction[0].events_per_job
```

**Example:** 960 events

**Formula:** Events per job of the first group in the construction

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

### 4. Memory Scaling Metrics

#### `total_memory_mb`
**Description:** Total memory required across all groups in the construction, accounting for the number of jobs each group needs to run to process the requested number of events.

**Calculation:**
```python
total_memory_mb = sum(
    group.max_memory_mb * group_jobs_needed[group.group_id]
    for group in construction
)
```

**Example:** 3,703,704 MB (for 1,000,000 requested events)

**Formula:** `sum(group.max_memory_mb * jobs_needed)` where `jobs_needed = request_num_events / group.events_per_job`

#### `memory_per_event_mb`
**Description:** Normalized memory per event, calculated by dividing the total memory by the requested number of events.

**Calculation:**
```python
memory_per_event_mb = total_memory_mb / request_num_events
```

**Example:** 3.704 MB/event (for 1,000,000 requested events)

**Formula:** `total_memory_mb / request_num_events`

### 5. Network Transfer Metrics

#### `total_network_transfer_mb`
**Description:** Total network transfer volume (remote read + remote write) across all groups in the construction, accounting for job scaling.

**Calculation:**
```python
total_network_transfer_mb = total_read_remote_mb + total_write_remote_mb
```

**Example:** 244,140 MB (for 1,000,000 requested events)

**Formula:** `total_read_remote_mb + total_write_remote_mb`

#### `network_transfer_per_event_mb`
**Description:** Normalized network transfer per event, calculated by dividing the total network transfer by the requested number of events.

**Calculation:**
```python
network_transfer_per_event_mb = total_network_transfer_mb / request_num_events
```

**Example:** 0.244 MB/event (for 1,000,000 requested events)

**Formula:** `total_network_transfer_mb / request_num_events`

### 6. Efficiency Metrics

#### `total_jobs`
**Description:** Total number of jobs needed across all groups to process the requested number of events.

**Calculation:**
```python
total_jobs = sum(group_jobs_needed.values())
```

**Example:** 926 jobs (for 1,000,000 requested events)

**Formula:** `sum(jobs_needed)` for all groups

#### `jobs_per_event`
**Description:** Normalized jobs per event, calculated by dividing the total number of jobs by the requested number of events. This is an efficiency metric - lower values indicate more efficient processing.

**Calculation:**
```python
jobs_per_event = total_jobs / request_num_events
```

**Example:** 0.000926 jobs/event (for 1,000,000 requested events)

**Formula:** `total_jobs / request_num_events`

### 7. Time Metrics

#### `total_wallclock_time`
**Description:** Total wallclock time required across all groups in the construction, accounting for job scaling. This assumes each job runs for the target wallclock time (12 hours by default).

**Calculation:**
```python
total_wallclock_time = sum(
    (TARGET_WALLCLOCK_TIME_HOURS * 3600) * group_jobs_needed[group.group_id]
    for group in construction
)
```

**Example:** 40,000,000 seconds (for 1,000,000 requested events)

**Formula:** `sum(wallclock_time_per_job * jobs_needed)` where `wallclock_time_per_job = TARGET_WALLCLOCK_TIME_HOURS * 3600`

#### `wallclock_time_per_event`
**Description:** Normalized wallclock time per event, calculated by dividing the total wallclock time by the requested number of events.

**Calculation:**
```python
wallclock_time_per_event = total_wallclock_time / request_num_events
```

**Example:** 40.0 seconds/event (for 1,000,000 requested events)

**Formula:** `total_wallclock_time / request_num_events`

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

#### `read_remote_mb`
**Description:** Remote read data volume for the group.

**Example:** 0.0 MB

**Formula:** `group.read_remote_mb`

#### `write_local_mb`
**Description:** Local write data volume for the group.

**Example:** 304.6875 MB

**Formula:** `group.write_local_mb`

#### `write_remote_mb`
**Description:** Remote write data volume for the group.

**Example:** 70.3125 MB

**Formula:** `group.write_remote_mb`

#### `read_remote_per_event_mb`
**Description:** Remote read data volume per event for the group.

**Example:** 0.0 MB/event

**Formula:** `group.read_remote_per_event_mb`

#### `write_local_per_event_mb`
**Description:** Local write data volume per event for the group.

**Example:** 0.3173828125 MB/event

**Formula:** `group.write_local_per_event_mb`

#### `write_remote_per_event_mb`
**Description:** Remote write data volume per event for the group.

**Example:** 0.0732421875 MB/event

**Formula:** `group.write_remote_per_event_mb`

## Example Construction Analysis

From the provided JSON file, here's an analysis of the first construction:

```json
{
  "total_events": 1000000,
  "total_cpu_time": 80000000,
  "event_throughput": 0.0125,
  "total_read_remote_mb": 0.0,
  "total_write_local_mb": 537109.38,
  "total_write_remote_mb": 48828.12,
  "read_remote_per_event_mb": 0.0,
  "write_local_per_event_mb": 0.537,
  "write_remote_per_event_mb": 0.049,
  "total_memory_mb": 3703704,
  "memory_per_event_mb": 3.704,
  "total_network_transfer_mb": 48828,
  "network_transfer_per_event_mb": 0.049,
  "total_jobs": 926,
  "jobs_per_event": 0.000926,
  "total_wallclock_time": 40000000,
  "wallclock_time_per_event": 40.0,
  "initial_input_events": 1080,
  "num_groups": 1,
  "groups": ["group_5"]
}
```

**Analysis:**
- **Construction Type:** Single-group construction (all tasks in one group)
- **Performance:** 0.0125 events/second throughput
- **Data Flow:** Each event generates 0.537 MB local write, stores 0.049 MB remote write
- **Resource Usage:** 80,000,000 CPU seconds total, 3,703,704 MB memory
- **Efficiency:** 926 total jobs needed, 0.000926 jobs per event
- **Network:** 48,828 MB total network transfer (0.049 MB/event)
- **Time:** 40,000,000 seconds wallclock time (40.0 seconds/event)

## Key Insights

1. **Throughput vs Groups:** Generally, fewer groups result in higher throughput due to reduced overhead
2. **Data Consistency:** Local write data per event is constant across all constructions (0.537 MB/event)
3. **Storage Variation:** Remote write data per event varies based on grouping strategy and exit points
4. **Resource Trade-offs:** More groups may improve parallelism but increase overhead
5. **Memory Scaling:** Total memory requirements scale with job scaling and group memory needs
6. **Network Efficiency:** Network transfer varies significantly based on data flow patterns between groups
7. **Job Efficiency:** Jobs per event is a key metric for comparing processing efficiency across constructions
8. **Time Consistency:** Wallclock time per event is consistent since all jobs target the same wallclock time

## Usage in Analysis

These metrics enable:
- **Performance Comparison:** Compare throughput across different grouping strategies
- **Resource Planning:** Estimate CPU time and data storage requirements
- **Cost Analysis:** Calculate resource costs based on CPU time and storage
- **Optimization:** Identify optimal grouping strategies for specific requirements
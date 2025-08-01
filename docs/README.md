# Documentation Directory

This directory contains detailed documentation for the workflow construction analysis system.

## Documentation Files

### [Workflow Construction Metrics](workflow_construction_metrics.md)
Complete documentation of all metrics calculated for workflow constructions, including:
- Quick reference tables with formulas and examples
- Detailed explanations of event processing metrics
- Data volume metrics and calculations
- Per-event metrics for data flow analysis
- Group-level details and calculations
- Example analysis with real data

### Group Metrics Documentation (Coming Soon)
Future documentation for group-level metrics analysis, including:
- Resource utilization calculations
- Event throughput analysis
- I/O requirements and storage rules
- Dependency path analysis
- Group performance characteristics

## Key Concepts

### Workflow Constructions
A workflow construction represents a valid way to organize tasks into groups for execution. Each construction:
- Covers all tasks in the workflow
- Respects task dependencies
- Provides performance and resource utilization metrics

### Metric Categories
1. **Event Processing:** Throughput, CPU time, total events
2. **Data Volume:** Input, output, and stored data totals
3. **Per-Event Analysis:** Data flow per individual event
4. **Construction Metadata:** Number of groups, group IDs

### Analysis Insights
- Output data per event is constant across all constructions
- Throughput generally decreases with more groups
- Stored data per event varies based on grouping strategy
- Resource trade-offs between parallelism and overhead

## Usage

These metrics enable:
- **Performance comparison** across different grouping strategies
- **Resource planning** based on CPU time and storage requirements
- **Cost analysis** for different workflow configurations
- **Optimization** to find optimal grouping strategies

## Example Data

All examples in the documentation are based on real data from `output/fork/5tasks/construction_metrics.json`, providing concrete values for understanding the metrics in practice. 
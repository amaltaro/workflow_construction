# Documentation Directory

This directory contains detailed documentation for the workflow construction analysis system.

## Documentation Files

### [Workflow Construction Metrics](workflow_construction_metrics.md)
Complete documentation of all metrics calculated for workflow constructions, including:
- Quick reference tables with formulas and examples
- Detailed explanations of event processing metrics
- Data volume metrics and calculations (remote read, local write, remote write)
- Per-event metrics for data flow analysis
- Job scaling and efficiency metrics
- Memory scaling and network transfer metrics
- Time analysis with resource constraints
- Group-level details and calculations
- Example analysis with real data

### [Group Metrics Documentation](group_metrics.md)
Complete documentation of all metrics calculated for individual task groups, including:
- Resource utilization calculations and formulas
- Event throughput analysis and calculations
- I/O requirements and storage rules (remote read, local write, remote write)
- Dependency path analysis
- Group performance characteristics
- Events per job calculation based on target wallclock time
- Detailed mathematical formulas for all metrics
- Usage examples and integration with visualization tools

### [Visualization and Analysis Documentation](visualization_analysis.md)
Comprehensive guide to the visualization and analysis capabilities, including:
- Available visualization types and their insights (10 different plots)
- Metrics tracked for groups and workflow constructions
- Job scaling analysis and time analysis visualizations
- Usage examples for Python API and command line interface
- Toy model mode for simplified analysis
- Integration with core analysis tools
- Analysis benefits and key insights

## Key Concepts

### Workflow Constructions
A workflow construction represents a valid way to organize tasks into groups for execution. Each construction:
- Covers all tasks in the workflow
- Respects task dependencies
- Provides performance and resource utilization metrics

### Metric Categories
1. **Event Processing:** Throughput, CPU time, total events
2. **Data Volume:** Remote read, local write, and remote write data totals
3. **Per-Event Analysis:** Data flow per individual event
4. **Construction Metadata:** Number of groups, group IDs
5. **Group-Level Metrics:** Resource utilization, I/O analysis, dependency patterns
6. **Job Scaling:** Total jobs, jobs per event, group job requirements
7. **Memory Scaling:** Total memory, memory per event
8. **Network Transfer:** Total network transfer, network transfer per event
9. **Time Analysis:** Baseline, resource-constrained execution times

### Analysis Insights
- Local write data per event is constant across all constructions
- Remote read/write data per event varies based on grouping strategy
- Throughput depends on resource utilization and group composition
- Job scaling reveals efficiency differences between constructions
- Memory and network transfer metrics scale with job requirements
- Time analysis shows impact of resource constraints vs. dependency constraints
- Group-level resource utilization affects overall workflow efficiency
- Memory occupancy and CPU utilization ratios guide optimal grouping

## Usage

These metrics enable:
- **Performance comparison** across different grouping strategies
- **Resource planning** based on CPU time and storage requirements
- **Cost analysis** for different workflow configurations
- **Optimization** to find optimal grouping strategies

## Example Data

All examples in the documentation are based on real data from `output/fork/5tasks/construction_metrics.json`, providing concrete values for understanding the metrics in practice.
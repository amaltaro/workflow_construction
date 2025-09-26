# Workflow Template Export for Standalone Simulation

This document describes the workflow template export functionality that allows you to generate JSON template files for each workflow composition, enabling standalone simulation and verification of results.

## Overview

The workflow construction system can now export each workflow composition as a standalone JSON template file that includes all the necessary information for simulation. Each template contains:

- **CompositionNumber**: An integer index that matches the workflow constructions used in all diagrams
- **GroupName**: The name of the group each taskset belongs to
- **GroupInputEvents**: The actual number of events that enter each group (same for all tasksets in the same group)

## Usage

Workflow templates are generated as part of the full visualization suite to ensure consistency between visualizations and template files.

Add the `--export-templates` flag to the main visualization script:

```bash
python src/vis_all_groups.py tests/fork/3tasks.json --export-templates --output-dir output/my_analysis
```

This will:
1. Generate all visualizations and metrics as usual
2. Export workflow composition templates to `output/sequential/3tasks/templates/` (or equivalent path based on input file)

## Output Structure

### Template Files

Each composition is exported as a separate JSON file with the naming pattern:
- `{template_name}_composition_{number:03d}.json`

For example:
- `3tasks_composition_001.json`
- `3tasks_composition_002.json`
- `3tasks_composition_003.json`

### Template File Format

Each template file contains the original workflow structure plus additional fields:

```json
{
  "Comments": "Workflow Composition 1 - 2 groups",
  "NumTasks": 3,
  "RequestNumEvents": 1000000,
  "CompositionNumber": 1,
  "Taskset1": {
    "KeepOutput": false,
    "Memory": 2000,
    "Multicore": 1,
    "RequiresGPU": "forbidden",
    "ScramArch": ["el9_amd64_gcc11"],
    "SizePerEvent": 200,
    "TimePerEvent": 10,
    "GroupName": "group_3",
    "GroupInputEvents": 1440
  },
  "Taskset2": {
    "KeepOutput": true,
    "Memory": 4000,
    "Multicore": 2,
    "RequiresGPU": "forbidden",
    "ScramArch": ["el9_amd64_gcc11"],
    "SizePerEvent": 300,
    "TimePerEvent": 20,
    "InputTaskset": "Taskset1",
    "GroupName": "group_3",
    "GroupInputEvents": 1440
  },
  "Taskset3": {
    "KeepOutput": true,
    "Memory": 3000,
    "Multicore": 2,
    "RequiresGPU": "forbidden",
    "ScramArch": ["el9_amd64_gcc11"],
    "SizePerEvent": 50,
    "TimePerEvent": 10,
    "InputTaskset": "Taskset1",
    "GroupName": "group_2",
    "GroupInputEvents": 4320
  }
}
```

### Summary File

A summary file is also created: `{template_name}_compositions_summary.json`

This contains an overview of all compositions with key metrics:

```json
{
  "template_name": "3tasks",
  "total_compositions": 3,
  "compositions": [
    {
      "composition_number": 1,
      "num_groups": 2,
      "groups": ["group_3", "group_2"],
      "total_cpu_time": 80000000.0,
      "event_throughput": 0.0125,
      "total_memory_mb": 3476000,
      "total_jobs": 925.925925925926,
      "group_details": [
        {
          "group_id": "group_3",
          "tasks": ["Taskset1", "Taskset2"],
          "events_per_task": 1440
        }
      ]
    }
  ]
}
```

## Key Features

### Composition Numbering
- Compositions are numbered starting from 1
- The numbering matches the order used in all visualizations and diagrams
- This ensures consistency between analysis and simulation

### Group Information
- Each taskset includes its `GroupName` (e.g., "group_3")
- All tasksets in the same group have the same `GroupInputEvents` value
- This allows simulators to understand group boundaries and event flow

### Event Flow
- `GroupInputEvents` represents the number of events that enter each group
- This is calculated based on the group's `events_per_job` metric
- Multiple tasksets in the same group will have identical `GroupInputEvents` values

## Use Cases

### Standalone Simulation
These templates can be used to:
1. **Verify Results**: Compare simulation results with the original workflow construction analysis
2. **Independent Testing**: Run simulations without the full workflow construction system
3. **Integration**: Use templates in other simulation frameworks or tools
4. **Reproducibility**: Ensure consistent results across different environments

### Example Integration
```python
import json

# Load a composition template
with open('3tasks_composition_001.json', 'r') as f:
    template = json.load(f)

# Extract group information
for i in range(1, template['NumTasks'] + 1):
    taskset = template[f'Taskset{i}']
    group_name = taskset['GroupName']
    input_events = taskset['GroupInputEvents']
    print(f"Taskset{i} belongs to {group_name} with {input_events} input events")
```

## File Organization

The templates are organized in the same directory structure as the analysis data:

```
output/
└── sequential/3tasks/            # Analysis output directory
    ├── templates/                # Template files directory
    │   ├── 3tasks_composition_001.json
    │   ├── 3tasks_composition_002.json
    │   ├── 3tasks_composition_003.json
    │   └── 3tasks_compositions_summary.json
    ├── construction_metrics.json
    ├── workflow_topologies.html
    └── ...
```

## Command Line Options

```bash
python src/vis_all_groups.py <template_file> [options]

Options:
  --export-templates    Export workflow compositions as JSON templates
  --output-dir DIR      Base output directory (default: output)
  --toy-model          Create toy model with extreme constructions only
```

## Examples

### Export all compositions from a fork workflow
```bash
python src/vis_all_groups.py tests/fork/3tasks.json --export-templates
```

### Export with custom output directory
```bash
python src/vis_all_groups.py tests/sequential/5tasks.json --export-templates --output-dir my_simulation_templates
```

### Export during full analysis
```bash
python src/vis_all_groups.py tests/fork/3tasks.json --export-templates --output-dir analysis_results
```

## Technical Details

### Implementation
The export functionality is implemented in:
- `src/export_templates.py`: Core export functions
- `src/vis_all_groups.py`: Integration with main visualization

### Dependencies
- Uses the existing workflow construction system
- No additional dependencies beyond the standard workflow construction requirements
- Compatible with all existing workflow types (fork, sequential, etc.)

### Performance
- Export adds minimal overhead to the existing workflow construction process
- Templates are generated in parallel with other analysis outputs
- File I/O is optimized for large numbers of compositions

### Consistency Guarantee
- Templates are always generated alongside visualizations to ensure consistency
- Composition numbering matches exactly between visualizations and template files
- This approach prevents discrepancies that could arise from separate generation processes


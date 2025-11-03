"""Module for exporting workflow compositions as JSON templates for standalone simulation."""

import json
import os
from typing import Dict, List, Any
from pathlib import Path


def create_composition_template(
    original_workflow: Dict[str, Any],
    construction_metrics: Dict[str, Any],
    composition_number: int,
    output_dir: str = "output/templates"
) -> Dict[str, Any]:
    """Create a JSON template for a specific workflow composition.
    
    Args:
        original_workflow: The original workflow JSON data
        construction_metrics: Metrics for the specific construction
        composition_number: Index of this composition (1-based)
        output_dir: Directory to save the template
        
    Returns:
        Dictionary containing the template JSON structure
    """
    # Start with the original workflow structure
    template = original_workflow.copy()
    
    # Add composition metadata
    template["CompositionNumber"] = composition_number
    template["Comments"] = f"Workflow Composition {composition_number} - {len(construction_metrics['groups'])} groups"
    
    # Create a mapping from task to group for easy lookup
    task_to_group = {}
    group_input_events = {}
    
    for group_detail in construction_metrics["group_details"]:
        group_id = group_detail["group_id"]
        tasks = group_detail["tasks"]
        input_events = group_detail["events_per_task"]
        
        # Map each task to its group
        for task in tasks:
            task_to_group[task] = group_id
            
        # Store input events for this group
        group_input_events[group_id] = input_events
    
    # Add group information to each taskset
    for i in range(1, template["NumTasks"] + 1):
        task_name = f"Taskset{i}"
        if task_name in template:
            # Add group name
            template[task_name]["GroupName"] = task_to_group.get(task_name, "unknown")
            
            # Add group input events
            group_name = task_to_group.get(task_name, "unknown")
            template[task_name]["GroupInputEvents"] = group_input_events.get(group_name, 0)
    
    return template


def export_all_compositions(
    original_workflow: Dict[str, Any],
    construction_metrics_list: List[Dict[str, Any]],
    template_name: str,
    output_dir: str = "output/templates"
) -> None:
    """Export all workflow compositions as JSON templates.
    
    Args:
        original_workflow: The original workflow JSON data
        construction_metrics_list: List of construction metrics for all compositions
        template_name: Base name for the template files
        output_dir: Directory to save the templates
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Exporting {len(construction_metrics_list)} workflow compositions as JSON templates...")
    
    for i, construction_metrics in enumerate(construction_metrics_list, 1):
        # Create the template for this composition
        template = create_composition_template(
            original_workflow, 
            construction_metrics, 
            i, 
            output_dir
        )
        
        # Generate filename
        filename = f"{template_name}_const_{i:03d}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Write the template to file
        with open(filepath, 'w') as f:
            json.dump(template, f, indent=2)
        
        print(f"  Exported composition {i}: {filename}")
    
    print(f"All templates exported to: {output_dir}")


def export_composition_summary(
    construction_metrics_list: List[Dict[str, Any]],
    template_name: str,
    output_dir: str = "output/templates"
) -> None:
    """Export a summary of all compositions for easy reference.
    
    Args:
        construction_metrics_list: List of construction metrics for all compositions
        template_name: Base name for the template files
        output_dir: Directory to save the summary
    """
    summary = {
        "template_name": template_name,
        "total_compositions": len(construction_metrics_list),
        "compositions": []
    }
    
    for i, construction_metrics in enumerate(construction_metrics_list, 1):
        composition_info = {
            "composition_number": i,
            "num_groups": construction_metrics["num_groups"],
            "groups": construction_metrics["groups"],
            "total_cpu_time": construction_metrics["total_cpu_time"],
            "event_throughput": construction_metrics["event_throughput"],
            "total_memory_mb": construction_metrics["total_memory_mb"],
            "total_jobs": construction_metrics["total_jobs"],
            "group_details": []
        }
        
        # Add group details
        for group_detail in construction_metrics["group_details"]:
            composition_info["group_details"].append({
                "group_id": group_detail["group_id"],
                "tasks": group_detail["tasks"],
                "events_per_task": group_detail["events_per_task"]
            })
        
        summary["compositions"].append(composition_info)
    
    # Write summary to file
    summary_file = os.path.join(output_dir, f"{template_name}_compositions_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Composition summary exported to: {summary_file}")


def main():
    """Example usage of the template export functionality."""
    import argparse
    from find_all_groups import create_workflow_from_json
    
    parser = argparse.ArgumentParser(description='Export workflow compositions as JSON templates.')
    parser.add_argument('template_file', type=str, help='Path to the template JSON file')
    parser.add_argument('--output-dir', type=str, default='output/templates',
                       help='Output directory for templates (default: output/templates)')
    args = parser.parse_args()
    
    # Load the original workflow
    with open(args.template_file) as f:
        original_workflow = json.load(f)
    
    # Generate all compositions
    groups, tasks, construction_metrics_list, dag = create_workflow_from_json(original_workflow)
    
    # Extract template name from file path
    template_name = Path(args.template_file).stem
    
    # Export all compositions
    export_all_compositions(
        original_workflow, 
        construction_metrics_list, 
        template_name, 
        args.output_dir
    )
    
    # Export summary
    export_composition_summary(
        construction_metrics_list, 
        template_name, 
        args.output_dir
    )


if __name__ == "__main__":
    main()


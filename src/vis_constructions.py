"""Module for visualizing workflow constructions using Mermaid diagrams."""

from typing import List, Dict
import os
import networkx as nx


def plot_workflow_topology(construction_metrics: List[Dict], output_dir: str = "output", template_name: str = None, dag: nx.DiGraph = None) -> None:
    """Create topology visualization for each workflow construction using Mermaid diagrams.
    
    This function creates a single HTML file containing Mermaid diagrams for all workflow constructions
    displayed side by side, with a consistent color scheme and legend for groups.
    
    Args:
        construction_metrics: List of dictionaries containing workflow construction metrics
        output_dir: Directory where the visualization will be saved
        template_name: Name of the template file being visualized
        dag: The directed acyclic graph representing task dependencies
    """
    print(f"Creating Mermaid visualizations for {len(construction_metrics)} workflow constructions")
    
    # Create a consistent color palette for groups using distinct colors
    # Using a carefully selected set of distinct colors
    distinct_colors = [
        "#FF6B6B",  # Coral Red
        "#4ECDC4",  # Turquoise
        "#45B7D1",  # Sky Blue
        "#96CEB4",  # Sage Green
        "#FFEEAD",  # Cream
        "#D4A5A5",  # Dusty Rose
        "#9B59B6",  # Purple
        "#3498DB",  # Blue
        "#E67E22",  # Orange
        "#2ECC71",  # Green
        "#F1C40F",  # Yellow
        "#E74C3C",  # Red
        "#1ABC9C",  # Teal
        "#34495E",  # Dark Blue
        "#F39C12",  # Dark Orange
    ]
    
    # Create a color map for groups
    all_groups = set()
    for metrics in construction_metrics:
        all_groups.update(metrics["groups"])
    group_colors = {group: distinct_colors[i % len(distinct_colors)] for i, group in enumerate(sorted(all_groups))}
    
    # Start building the HTML content
    title = "Workflow Construction Topologies"
    if template_name:
        title += f" - {template_name}"

    # Read the CSS file content to inline it in the HTML
    css_path = os.path.join(os.path.dirname(__file__), '../css/workflow_styles.css')
    try:
        with open(css_path, 'r') as css_file:
            css_content = css_file.read()
    except Exception as e:
        print(f"Warning: Could not read CSS file at {css_path}: {e}")
        css_content = ''

    html_content = f"""<!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>\n{css_content}\n</style>
        <script src=\"https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js\"></script>
    </head>
    <body>
        <h1>{title}</h1>
        
        <!-- Group Composition -->
        <div class="group-composition">
            <h3>Group/Task Composition:</h3>
            <ul>
    """
    
    # Add task descriptions for each group
    for group in sorted(all_groups):
        # Find the first construction that contains this group
        group_info = None
        for metrics in construction_metrics:
            for g in metrics["group_details"]:
                if g["group_id"] == group:
                    group_info = g
                    break
            if group_info:
                break
        
        if group_info:
            tasks_str = ", ".join(group_info["tasks"])
            html_content += f'<li>Group {group}: {tasks_str}</li>\n'
    
    html_content += """
            </ul>
        </div>
        
        <!-- Legend -->
        <div class="legend">
            <h3>Groups</h3>
    """
    
    # Add legend items
    for group in sorted(all_groups):
        html_content += f"""
            <div class="legend-item">
                <div class="legend-color" style="background-color: {group_colors[group]}"></div>
                <span>Group {group}</span>
            </div>
        """
    
    html_content += """
        </div>
        
        <!-- Diagrams Container -->
        <div class="container">
    """
    
    # Add each construction's diagram
    for i, metrics in enumerate(construction_metrics, 1):
        # Start the Mermaid diagram
        mermaid_content = f"""
        <div class="construction">
            <div class="construction-title">Workflow Construction {i}</div>
            <div class="mermaid">
            graph TD
        """
        
        # Create a mapping of tasks to their groups
        task_to_group = {}
        for group in metrics["group_details"]:
            for task in group["tasks"]:
                task_to_group[task] = group["group_id"]

        # Add subgraphs for each group
        for group in metrics["group_details"]:
            group_id = group["group_id"]
            mermaid_content += f'    subgraph {group_id} [Group {group_id}]\n'
            for task in group["tasks"]:
                # Optionally, keep color classes for nodes
                mermaid_content += f'        {task}["{task}"]:::group{group_id}\n'
            mermaid_content += '    end\n'

        # Add edges based on the DAG
        if dag is not None:
            for edge in dag.edges():
                source, target = edge
                mermaid_content += f'    {source} --> {target}\n'

        # Add style definitions for each group
        mermaid_content += "\n    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px;\n"
        for group_id in metrics["groups"]:
            mermaid_content += f'    classDef group{group_id} fill:{group_colors[group_id]},stroke:#333,stroke-width:2px,color:white;\n'
        
        mermaid_content += "</div></div>\n"
        html_content += mermaid_content
    
    # Close the HTML content
    html_content += """
        </div>
        <script>
            mermaid.initialize({
                startOnLoad: true,
                theme: 'default',
                flowchart: {
                    useMaxWidth: false,
                    htmlLabels: true,
                    curve: 'basis'
                }
            });
        </script>
    </body>
    </html>
    """
    
    # Save the HTML file
    with open(os.path.join(output_dir, "workflow_topologies.html"), "w") as f:
        f.write(html_content)
    
    print(f"Created workflow topology visualization at {os.path.join(output_dir, 'workflow_topologies.html')}") 
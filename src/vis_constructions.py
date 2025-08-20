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
    
    # Use only one color for all groups
    node_color = "#34495E"  # Dark Blue

    # Collect all unique groups and their tasks for the summary
    group_to_tasks = {}
    for metrics in construction_metrics:
        for group in metrics["group_details"]:
            group_id = group["group_id"]
            if group_id not in group_to_tasks:
                group_to_tasks[group_id] = list(group["tasks"])

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
    
    def extract_group_number(group):
        """
        Accepts either a group_id string (e.g., 'group_7') or a dict with a 'group_id' key.
        Returns the integer group number for sorting.
        """
        if isinstance(group, dict):
            group_id = group.get("group_id", "")
        else:
            group_id = group
        try:
            return int(group_id.split('_')[1])
        except (IndexError, ValueError):
            return float('inf')

    def task_sort_key(task_id):
        try:
            return int(task_id.replace("Taskset", ""))
        except ValueError:
            return float('inf')

    # Add task descriptions for each group, sorted by group number
    for group in sorted(group_to_tasks, key=extract_group_number):
        tasks_str = ", ".join(group_to_tasks[group])
        html_content += f'<li>Group {group}: {tasks_str}</li>\n'
    
    html_content += """
            </ul>
        </div>
        
        <!-- Diagrams Container -->
        <div class="container">
    """
    
    def is_sequential_workflow(metrics):
        # Heuristic: if every group has at most one input and one output, treat as sequential
        all_edges = set()
        for group in metrics["group_details"]:
            for task in group["tasks"]:
                # Find all outgoing and incoming edges for this task
                outgoing = [e for e in dag.edges() if e[0] == task]
                incoming = [e for e in dag.edges() if e[1] == task]
                if len(outgoing) > 1 or len(incoming) > 1:
                    return False
        return True

    # Determine if this is a sequential workflow for CSS styling (DAG structure doesn't change)
    is_sequential = dag is not None and is_sequential_workflow(construction_metrics[0]) if construction_metrics else False
    # Determine Mermaid direction
    mermaid_direction = 'LR' if is_sequential else 'TD'
    # Determine if this is a sequential workflow for CSS styling
    sequential_class = " sequential" if is_sequential else ""

    # Add each construction's diagram
    for i, metrics in enumerate(construction_metrics, 1):
        # Start the Mermaid diagram
        mermaid_content = f"""
        <div class="construction{sequential_class}">
            <div class="construction-title">Workflow Construction {i}</div>
            <div class="mermaid{sequential_class}">
            graph {mermaid_direction}
        """
        
        # Add subgraphs for each group, sorted by group number
        for group in sorted(metrics["group_details"], key=extract_group_number):
            group_id = group["group_id"]
            mermaid_content += f'    subgraph {group_id} [Group {group_id}]\n'
            # Add tasks in sorted order by task number
            for task in sorted(group["tasks"], key=task_sort_key):
                mermaid_content += f'        {task}["{task}"]:::groupnode\n'
            mermaid_content += '    end\n'

        # Add edges based on the DAG, sorted by source and target task number
        if dag is not None:
            for source, target in sorted(dag.edges(), key=lambda e: (task_sort_key(e[0]), task_sort_key(e[1]))):
                mermaid_content += f'    {source} --> {target}\n'

        # Add style definition for all nodes (single color)
        mermaid_content += "\n    classDef groupnode fill:{} ,stroke:#333,stroke-width:2px,color:white;\n".format(node_color)
        mermaid_content += "    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px;\n"
        
        mermaid_content += "</div></div>\n"
        html_content += mermaid_content

    
    # Close the HTML content
    html_content += """
        </div>
        <script>
            mermaid.initialize({{
                startOnLoad: true,
                theme: 'default',
                flowchart: {{
                    useMaxWidth: true,
                    htmlLabels: true,
                    curve: 'basis',
                    nodeSpacing: 50,
                    rankSpacing: 50,
                    diagramPadding: 20
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    # Save the HTML file
    with open(os.path.join(output_dir, "workflow_topologies.html"), "w") as f:
        f.write(html_content)
    
    print(f"Created workflow topology visualization at {os.path.join(output_dir, 'workflow_topologies.html')}") 
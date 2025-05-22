"""Module for visualizing workflow constructions using Mermaid diagrams."""

from typing import List, Dict
import os


def plot_workflow_topology(construction_metrics: List[Dict], output_dir: str = "plots") -> None:
    """Create topology visualization for each workflow construction using Mermaid diagrams.
    
    This function creates a single HTML file containing Mermaid diagrams for all workflow constructions
    displayed side by side, with a consistent color scheme and legend for groups.
    
    Args:
        construction_metrics: List of dictionaries containing workflow construction metrics
        output_dir: Directory where the visualization will be saved
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
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Workflow Construction Topologies</title>
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <style>
            .container {
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                padding: 20px;
            }
            .construction {
                flex: 1 1 23%;
                min-width: 300px;
                margin-bottom: 15px;
            }
            .mermaid {
                margin: 10px 0;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .construction-title {
                font-size: 1.1em;
                font-weight: bold;
                margin: 10px 0;
            }
            .legend {
                position: fixed;
                top: 20px;
                right: 20px;
                background: white;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                max-height: 80vh;
                overflow-y: auto;
            }
            .legend-item {
                display: flex;
                align-items: center;
                margin: 5px 0;
            }
            .legend-color {
                width: 20px;
                height: 20px;
                margin-right: 10px;
                border: 1px solid #333;
            }
            .group-composition {
                margin: 20px;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 5px;
                font-size: 0.9em;
                line-height: 1.4;
            }
            .group-composition h3 {
                margin: 0 0 10px 0;
            }
            .group-composition ul {
                margin: 0;
                padding-left: 20px;
            }
            .group-composition li {
                margin: 2px 0;
            }
        </style>
    </head>
    <body>
        <h1>Workflow Construction Topologies</h1>
        
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
        
        # Add nodes and edges for each group
        for group in metrics["group_details"]:
            tasks = group["tasks"]
            group_id = group["group_id"]
            
            # Add nodes for this group with color
            for task in tasks:
                mermaid_content += f'    {task}["{task}"]:::group{group_id}\n'
            
            # Add edges within the group
            for j in range(len(tasks) - 1):
                mermaid_content += f'    {tasks[j]} --> {tasks[j + 1]}\n'
        
        # Add edges between groups
        for j in range(len(metrics["group_details"]) - 1):
            current_group = metrics["group_details"][j]
            next_group = metrics["group_details"][j + 1]
            mermaid_content += f'    {current_group["tasks"][-1]} --> {next_group["tasks"][0]}\n'
        
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
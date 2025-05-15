import json
import pandas as pd

def parse_metrics(json_file):
    # Read the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Create a list to store the extracted data
    table_data = []
    
    # Extract the required fields from each group
    for group in data:
        row = {
            'group_id': group['group_id'],
            'task_ids': ', '.join(group['task_ids']),
            'cpu_total_cores': group['resource_metrics']['cpu']['total_cores'],
            'cpu_max_cores': group['resource_metrics']['cpu']['max_cores'],
            'cpu_utilization_ratio': group['resource_metrics']['cpu']['utilization_ratio'],
            'cpu_seconds': group['resource_metrics']['cpu']['cpu_seconds'],
            'memory_total_mb': group['resource_metrics']['memory']['total_mb'],
            'memory_max_mb': group['resource_metrics']['memory']['max_mb'],
            'memory_min_mb': group['resource_metrics']['memory']['min_mb'],
            'memory_occupancy': group['resource_metrics']['memory']['occupancy'],
            'io_total_output_mb': group['resource_metrics']['io']['total_output_mb'],
            'io_max_output_mb': group['resource_metrics']['io']['max_output_mb'],
            'throughput_total_eps': group['resource_metrics']['throughput']['total_eps'],
            'events_per_job': group['events_per_job']
        }
        table_data.append(row)

    # Create a DataFrame
    df = pd.DataFrame(table_data)
    
    # Display the table
    print("\nMetrics Table:")
    print(df.to_string(index=False))
    
    # Optionally save to CSV
    df.to_csv('metrics_table.csv', index=False)
    print("\nTable has been saved to 'metrics_table.csv'")

if __name__ == "__main__":
    parse_metrics('plots/group_metrics.json') 
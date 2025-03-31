import json
from src.group_tasks import TaskResources, Task, TaskGrouper, extract_os_and_arch


def test_workflow_from_json():
    # Load the JSON file
    with open("tests/ex0_perfect.json", "r") as file:
        workflow = json.load(file)

    # Create tasks dictionary
    tasks = {}
    for i in range(1, workflow["NumTasks"] + 1):
        task_name = f"Task{i}"
        task_data = workflow[task_name]
        
        # Extract OS version and architecture from ScramArch
        os_version, cpu_arch = extract_os_and_arch(task_data["ScramArch"])
        
        # Calculate events per second from TimePerEvent
        events_per_second = 1000 / task_data.get("TimePerEvent", 1)  # Default to 1 if not present
        
        # Create TaskResources
        resources = TaskResources(
            os_version=os_version,
            cpu_arch=cpu_arch,
            memory_mb=task_data.get("Memory", 1000),  # Default to 1000 if not present
            accelerator="GPU" if task_data.get("RequiresGPU") == "required" else None,
            cpu_cores=task_data.get("Multicore", 1),  # Default to 1 if not present
            events_per_second=events_per_second
        )

        # Create Task
        tasks[task_name] = Task(
            id=task_name,
            resources=resources,
            input_task=task_data.get("InputTask", None),
            output_tasks=set()  # Will be populated later
        )

    # Add output tasks
    for task_name, task in tasks.items():
        if task.input_task:
            if tasks[task.input_task].output_tasks is None:
                tasks[task.input_task].output_tasks = set()
            tasks[task.input_task].output_tasks.add(task_name)

    # Create task grouper
    grouper = TaskGrouper(tasks)
    
    # Get task groups
    groups = grouper.group_tasks()
    
    # Print results
    print("\nTask Groups:")
    # Sort the groups by their first task's ID number
    sorted_groups = sorted(groups, key=lambda g: min(int(task_name[4:]) for task_name in g))
    
    for i, group in enumerate(sorted_groups, 1):
        # Sort tasks within the group by their ID number
        sorted_tasks = sorted(group, key=lambda t: int(t[4:]))  # Extract number from "TaskX"
        print(f"Group {i}: {sorted_tasks}")
        
        # Print group details
        print("Group details:")
        for task_name in sorted_tasks:  # Use sorted tasks list
            task = tasks[task_name]
            print(f"  {task_name}:")
            print(f"    OS: {task.resources.os_version}")
            print(f"    CPU: {task.resources.cpu_arch}")
            print(f"    Memory: {task.resources.memory_mb}MB")
            print(f"    Cores: {task.resources.cpu_cores}")
            print(f"    Accelerator: {task.resources.accelerator}")
            print(f"    Events/sec: {task.resources.events_per_second:.2f}")
        print()

    # Add some basic assertions
    assert len(groups) > 0, "Should create at least one group"
    for group in groups:
        assert len(group) > 0, "Groups should not be empty"
        
    # Check that each task appears exactly once
    all_tasks = set()
    for group in groups:
        all_tasks.update(group)
    assert all_tasks == set(tasks.keys()), "All tasks should be assigned to exactly one group"


if __name__ == "__main__":
    test_workflow_from_json() 
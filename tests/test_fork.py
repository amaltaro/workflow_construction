# Standard library imports
import json
import logging
import os
import sys

# Local application imports
from src.group_tasks import create_workflow_from_json


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def test_workflow_from_json(json_template):
    """Provided a JSON template, executes the workflow composition algorithm"""
    # Load the JSON file
    json_template = f"tests/fork/{json_template}"
    logger.info(f"\n\n\nProcessing template: {json_template}")
    with open(json_template, "r") as file:
        workflow = json.load(file)

    # Create workflow and get task groups
    groups, tasks = create_workflow_from_json(workflow, min_group_score=0.7)
    
    # Print results
    logger.info("\nTask Groups:")
    # Sort the groups by their first task's ID number
    sorted_groups = sorted(groups, key=lambda g: min(int(task_name.replace("Taskset", "")) for task_name in g))
    
    for i, group in enumerate(sorted_groups, 1):
        # Sort tasks within the group by their ID number
        sorted_tasks = sorted(group, key=lambda t: int(t.replace("Taskset", "")))  # Extract number from "TasksetX"
        logger.info(f"Group {i}: {sorted_tasks}")
        
        # Print group details
        logger.info("Group details:")
        for task_name in sorted_tasks:  # Use sorted tasks list
            task = tasks[task_name]
            logger.info(f"  {task_name}:")
            logger.info(f"    OS: {task.resources.os_version}")
            logger.info(f"    CPU: {task.resources.cpu_arch}")
            logger.info(f"    Memory: {task.resources.memory_mb}MB")
            logger.info(f"    Cores: {task.resources.cpu_cores}")
            logger.info(f"    Accelerator: {task.resources.accelerator}")
            logger.info(f"    Events/sec: {task.resources.events_per_second:.2f}")
        logger.info("")

    # Add some basic assertions
    assert len(groups) > 0, "Should create at least one group"
    for group in groups:
        assert len(group) > 0, "Groups should not be empty"
        
    # Check that each task appears exactly once
    all_tasks = set()
    for group in groups:
        all_tasks.update(group)
    assert all_tasks == set(tasks.keys()), "All tasks should be assigned to exactly one group"

    logger.info(f"Expected grouping: {workflow['Comments']}")


if __name__ == "__main__":
    for json_template in os.listdir("/Users/amaltar2/Master/workflow_construction/tests/fork/"):
        test_workflow_from_json(json_template) 
    #test_workflow_from_json("3group_perfect.json")
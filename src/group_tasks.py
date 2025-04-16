# Standard library imports
import logging
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

# Third-party library imports
import networkx as nx


# Add this after the imports
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Keep it simple since we're just replacing print statements
    stream=sys.stdout  # Ensure output goes to stdout
)
logger = logging.getLogger(__name__)


def extract_os_and_arch(scram_arch: List[str]) -> Tuple[str, str]:
    """Extract OS version and CPU architecture from ScramArch string.
    
    Args:
        scram_arch: List of ScramArch strings (e.g., ["el8_amd64_gcc11"])
        
    Returns:
        Tuple of (os_version, cpu_arch)
        e.g., ("8", "amd64")
    """
    # Split the ScramArch string into its components
    os_part, arch, _ = scram_arch[0].split('_')  # Using [0] as ScramArch is a list
    # Extract just the numeric version from the OS part
    os_version = ''.join(char for char in os_part if char.isdigit())
    return os_version, arch


@dataclass
class TaskResources:
    """
    Class to store the resources of a task
    """
    os_version: str
    cpu_arch: str
    memory_mb: int
    accelerator: Optional[str]
    cpu_cores: int
    events_per_second: float


@dataclass
class Task:
    """
    Class to store the task resources and dependencies
    """
    id: str
    resources: TaskResources
    input_task: Optional[str] = None
    output_tasks: Set[str] = None


class GroupScore:
    """
    Class to calculate the score of a group of tasks
    """
    ALLOWED_KEYS = {'cpu', 'memory', 'throughput', 'accelerator'}

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        # Ensure only supported weight parameters are provided
        weights = weights or {}
        if weights:
            invalid_keys = set(weights.keys()) - self.ALLOWED_KEYS
            if invalid_keys:
                raise ValueError(f"Invalid weight keys: {invalid_keys}. Allowed keys are: {self.ALLOWED_KEYS}")

        # Set user weights, default to 1.0 if not provided
        self.weights = {}
        for param in self.ALLOWED_KEYS:
            self.weights.setdefault(param, weights.get(param, 1.0))
            if self.weights[param] <= 0.0:
                raise ValueError(f"Weight for '{param}' must be > 0.0, got {self.weights[param]}")

        self.cpu_score: float = 0.0
        self.memory_score: float = 0.0
        self.throughput_score: float = 0.0
        self.accelerator_score: float = 0.0
    
    def total_score(self) -> float:
        weighted_sum = (
            self.weights['cpu'] * self.cpu_score +
            self.weights['memory'] * self.memory_score +
            self.weights['throughput'] * self.throughput_score +
            self.weights['accelerator'] * self.accelerator_score
        )
        total_weight = sum(self.weights.values())
        return weighted_sum / total_weight if total_weight > 0 else 0.0


class TaskGrouper:
    """
    Class to group tasks into groups based on their resources and dependencies
    """
    def __init__(self, tasks: Dict[str, Task], min_group_score: float = 0.7):
        self.tasks = tasks
        self.min_group_score = min_group_score
        self.groups: List[Set[str]] = []
        self.dag = self._build_dag()

    def _build_dag(self) -> nx.DiGraph:
        """Build a directed acyclic graph from tasks"""
        dag = nx.DiGraph()
        for task_id, task in self.tasks.items():
            dag.add_node(task_id)
            if task.input_task:
                dag.add_edge(task.input_task, task_id)
        return dag

    def _can_be_grouped(self, task1: Task, task2: Task) -> bool:
        """Check hard constraints for grouping tasks"""
        # OS and CPU architecture must match
        if (task1.resources.os_version != task2.resources.os_version or
            task1.resources.cpu_arch != task2.resources.cpu_arch):
            return False

        # Check dependency relationship
        if not self._check_dependency_chain(task1.id, task2.id):
            return False

        return True

    def _check_dependency_chain(self, task1_id: str, task2_id: str) -> bool:
        """Check if tasks have a valid dependency chain for grouping"""
        # If tasks are independent, they must have a complete chain between them
        if not nx.has_path(self.dag, task1_id, task2_id) and \
           not nx.has_path(self.dag, task2_id, task1_id):
            # Check if there's a common ancestor with complete chains
            common_ancestors = nx.ancestors(self.dag, task1_id) & nx.ancestors(self.dag, task2_id)
            if not common_ancestors:
                return False
        return True

    def _calculate_group_score(self, tasks: List[Task]) -> GroupScore:
        """Calculate normalized scores for a group of tasks"""
        score = GroupScore()
        
        # Calculate CPU utilization score
        total_cores = sum(t.resources.cpu_cores for t in tasks)
        max_cores = max(t.resources.cpu_cores for t in tasks)
        score.cpu_score = min(1.0, max_cores / total_cores)

        # Calculate memory compatibility score
        max_memory = max(t.resources.memory_mb for t in tasks)
        min_memory = min(t.resources.memory_mb for t in tasks)
        score.memory_score = min_memory / max_memory if max_memory > 0 else 1.0

        # Calculate throughput compatibility score
        max_throughput = max(t.resources.events_per_second for t in tasks)
        min_throughput = min(t.resources.events_per_second for t in tasks)
        score.throughput_score = min_throughput / max_throughput if max_throughput > 0 else 1.0

        # Calculate accelerator compatibility score
        accelerators = set(t.resources.accelerator for t in tasks if t.resources.accelerator)
        score.accelerator_score = 1.0 if len(accelerators) <= 1 else 0.5

        return score

    def _all_dependency_paths_within_group(self, group: Set[str]) -> bool:
        """
        For every pair of tasks in the group, if there is a path between them,
        all tasks on that path must also be in the group.
        """
        for src in group:
            for dst in group:
                if src == dst:
                    continue
                if nx.has_path(self.dag, src, dst):
                    for path in nx.all_simple_paths(self.dag, src, dst):
                        if not all(node in group for node in path):
                            return False
        return True

    def group_tasks(self):
        """Main method to group tasks"""
        ungrouped_tasks = set(self.tasks.keys())
        
        while ungrouped_tasks:
            # Start with the task that has the most dependencies
            current_task = max(
                ungrouped_tasks,
                key=lambda t: len(list(self.dag.predecessors(t))) + len(list(self.dag.successors(t)))
            )
            
            # Create new group starting with this task
            current_group = {current_task}
            logger.info(f"\nStarting new group with: {current_task}")
            ungrouped_tasks.remove(current_task)
            
            # Try to add compatible tasks to the group
            for task_id in sorted(
                ungrouped_tasks,
                key=lambda t: len(list(self.dag.predecessors(t))) + len(list(self.dag.successors(t))),
                reverse=True
            ):
                # Check if task can be added to group - hard constraints
                if all(self._can_be_grouped(self.tasks[current], self.tasks[task_id]) 
                      for current in current_group):
                    # Calculate score with new task - aggregated soft constraints
                    potential_group = current_group | {task_id}
                    score = self._calculate_group_score(
                        [self.tasks[t] for t in potential_group]
                    )
                    
                    logger.info(f"Considering adding '{task_id}' to group {sorted(current_group)}")
                    logger.info(f"Scores for potential group {sorted(potential_group)}:")
                    logger.debug(f"  CPU Score: {score.cpu_score:.2f}")
                    logger.debug(f"  Memory Score: {score.memory_score:.2f}")
                    logger.debug(f"  Throughput Score: {score.throughput_score:.2f}")
                    logger.debug(f"  Accelerator Score: {score.accelerator_score:.2f}")
                    logger.info(f"  Total Score: {score.total_score():.2f}")

                    if score.total_score() >= self.min_group_score:
                        if self._all_dependency_paths_within_group(potential_group):
                            logger.info(f"[+] Adding {task_id} to group (score: {score.total_score():.2f})")
                            current_group.add(task_id)
                            ungrouped_tasks.remove(task_id)
                        else:
                            logger.info(f"[-] Cannot add {task_id} - dependency path constraint not met")
                    else:
                        logger.info(f"[-] Cannot add {task_id} - score too low ({score.total_score():.2f} < {self.min_group_score})")
            
            logger.info(f"Finalized group: {sorted(current_group)}")
            self.groups.append(current_group)

        return self.groups


def create_workflow_from_json(workflow_data: dict, min_group_score: float = 0.7) -> List[Set[str]]:
    """Create a workflow of tasks from JSON data and perform task grouping.

    Args:
        workflow_data: Dictionary containing the workflow specification
        min_group_score: Minimum score required for grouping tasks (default: 0.7)

    Returns:
        List of sets, where each set contains the task IDs for that group
    """
    # Create tasks dictionary
    tasks = {}
    for i in range(1, workflow_data["NumTasks"] + 1):
        task_name = f"Task{i}"
        task_data = workflow_data[task_name]

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

    # Create task grouper and perform grouping
    grouper = TaskGrouper(tasks, min_group_score=min_group_score)
    return grouper.group_tasks(), tasks


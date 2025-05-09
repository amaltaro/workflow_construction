# Standard library imports
import logging
import sys
from dataclasses import dataclass
from enum import Enum
from pprint import pformat
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
    time_per_event: float
    size_per_event: float
    input_events: int


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


@dataclass
class GroupMetrics:
    """
    Class to store detailed metrics about a group of tasks
    """
    group_id: str
    task_ids: Set[str]
    # CPU metrics
    total_cpu_cores: int
    max_cpu_cores: int
    cpu_seconds: float  # Total CPU-seconds required
    cpu_efficiency: float  # Ratio of actual CPU usage to allocated CPU-seconds
    # Memory metrics
    total_memory_mb: int
    max_memory_mb: int
    min_memory_mb: int
    memory_occupancy: float  # Ratio of total memory used to max memory allocated
    # Throughput metrics
    total_throughput: float
    max_throughput: float
    min_throughput: float
    # I/O metrics
    total_output_size_mb: float
    max_output_size_mb: float
    # Accelerator metrics
    accelerator_types: Set[str]
    # Dependency metrics
    dependency_paths: List[List[str]]
    # Resource utilization metrics
    resource_utilization: float  # Overall resource utilization efficiency
    event_throughput: float  # Events processed per second

    def to_dict(self) -> dict:
        """Convert metrics to a dictionary for easy serialization"""
        return {
            "group_id": self.group_id,
            "task_ids": sorted(list(self.task_ids)),
            "resource_metrics": {
                "cpu": {
                    "total_cores": self.total_cpu_cores,
                    "max_cores": self.max_cpu_cores,
                    "cpu_seconds": self.cpu_seconds,
                    "efficiency": self.cpu_efficiency
                },
                "memory": {
                    "total_mb": self.total_memory_mb,
                    "max_mb": self.max_memory_mb,
                    "min_mb": self.min_memory_mb,
                    "occupancy": self.memory_occupancy
                },
                "throughput": {
                    "total_eps": self.total_throughput,
                    "max_eps": self.max_throughput,
                    "min_eps": self.min_throughput
                },
                "io": {
                    "total_output_mb": self.total_output_size_mb,
                    "max_output_mb": self.max_output_size_mb
                },
                "accelerator": {
                    "types": list(self.accelerator_types)
                }
            },
            "utilization_metrics": {
                "resource_utilization": self.resource_utilization,
                "event_throughput": self.event_throughput
            },
            "dependency_paths": self.dependency_paths
        }


class TaskGrouper:
    """
    Class to group tasks into groups based on their dependencies
    """
    def __init__(self, tasks: Dict[str, Task]):
        self.tasks = tasks
        self.groups: List[Set[str]] = []
        self.dag = self._build_dag()
        self.all_possible_groups: List[GroupMetrics] = []

    def _build_dag(self) -> nx.DiGraph:
        """Build a directed acyclic graph from tasks"""
        dag = nx.DiGraph()
        for task_id, task in self.tasks.items():
            dag.add_node(task_id)
            if task.input_task:
                dag.add_edge(task.input_task, task_id)
        return dag

    def _can_be_grouped(self, task1: Task, task2: Task) -> bool:
        """Check if tasks can be grouped based on dependencies"""
        return self._check_dependency_chain(task1.id, task2.id)

    def _check_dependency_chain(self, task1_id: str, task2_id: str) -> bool:
        """Check if tasks have a valid dependency chain for grouping.

        Tasks can only be grouped if there is a direct dependency path
        between them in either direction.
        """
        return nx.has_path(self.dag, task1_id, task2_id) or \
               nx.has_path(self.dag, task2_id, task1_id)

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
                    # fetch all possible paths between src and dst
                    for path in nx.all_simple_paths(self.dag, src, dst):
                        if not all(node in group for node in path):
                            return False
        return True

    def _calculate_group_metrics(self, group: Set[str]) -> GroupMetrics:
        """Calculate detailed metrics for a group of tasks"""
        print(f"Calculating group metrics for {group}")
        tasks = [self.tasks[task_id] for task_id in group]
        
        # Calculate CPU metrics
        total_cores = sum(t.resources.cpu_cores for t in tasks)
        max_cores = max(t.resources.cpu_cores for t in tasks)
        
        # Calculate CPU-seconds and efficiency
        cpu_seconds = 0.0
        max_cpu_seconds = 0.0
        for task in tasks:
            # Calculate task duration in seconds based on total events
            task_duration = task.resources.input_events * task.resources.time_per_event
            # Add CPU-seconds for this task
            task_cpu_seconds = task.resources.cpu_cores * task_duration
            cpu_seconds += task_cpu_seconds
            max_cpu_seconds = max(max_cpu_seconds, task_cpu_seconds)
        
        # CPU efficiency is the ratio of actual CPU usage to allocated CPU-seconds
        cpu_efficiency = max_cpu_seconds / cpu_seconds if cpu_seconds > 0 else 0.0

        # Calculate memory metrics
        total_memory = sum(t.resources.memory_mb for t in tasks)
        max_memory = max(t.resources.memory_mb for t in tasks)
        min_memory = min(t.resources.memory_mb for t in tasks)
        
        # Calculate time-weighted memory occupancy
        total_duration = 0.0
        weighted_memory = 0.0
        for task in tasks:
            task_duration = task.resources.input_events * task.resources.time_per_event
            total_duration += task_duration
            weighted_memory += task.resources.memory_mb * task_duration
        
        # Calculate time-weighted average memory
        time_weighted_avg_memory = weighted_memory / total_duration if total_duration > 0 else 0.0
        
        # Memory occupancy is the ratio of time-weighted average memory to max memory
        memory_occupancy = time_weighted_avg_memory / max_memory if max_memory > 0 else 0.0

        # Calculate throughput metrics
        total_duration = 0.0
        weighted_throughput = 0.0
        for task in tasks:
            task_duration = task.resources.input_events * task.resources.time_per_event
            total_duration += task_duration
            weighted_throughput += task.resources.events_per_second * task_duration
        
        # Calculate time-weighted average throughput
        total_throughput = weighted_throughput / total_duration if total_duration > 0 else 0.0
        max_throughput = max(t.resources.events_per_second for t in tasks)
        min_throughput = min(t.resources.events_per_second for t in tasks)

        # Calculate I/O metrics
        total_output_size = 0.0
        max_output_size = 0.0
        for task in tasks:
            # Convert KB to MB for consistency with other memory metrics
            task_output_size = (task.resources.input_events * task.resources.size_per_event) / 1024.0
            total_output_size += task_output_size
            max_output_size = max(max_output_size, task_output_size)

        # Calculate accelerator metrics
        accelerators = set(t.resources.accelerator for t in tasks if t.resources.accelerator)

        # Calculate dependency paths
        dependency_paths = []
        for src in group:
            for dst in group:
                if src != dst and nx.has_path(self.dag, src, dst):
                    paths = list(nx.all_simple_paths(self.dag, src, dst))
                    dependency_paths.extend(paths)

        # Calculate overall resource utilization
        # This is a weighted average of CPU and memory efficiency
        resource_utilization = (cpu_efficiency + memory_occupancy) / 2.0

        # Calculate event throughput
        # This is the total number of events processed per second
        event_throughput = total_throughput

        return GroupMetrics(
            group_id=f"group_{len(self.all_possible_groups)}",
            task_ids=group,
            total_cpu_cores=total_cores,
            max_cpu_cores=max_cores,
            cpu_seconds=cpu_seconds,
            cpu_efficiency=cpu_efficiency,
            total_memory_mb=total_memory,
            max_memory_mb=max_memory,
            min_memory_mb=min_memory,
            memory_occupancy=memory_occupancy,
            total_throughput=total_throughput,
            max_throughput=max_throughput,
            min_throughput=min_throughput,
            total_output_size_mb=total_output_size,
            max_output_size_mb=max_output_size,
            accelerator_types=accelerators,
            dependency_paths=dependency_paths,
            resource_utilization=resource_utilization,
            event_throughput=event_throughput
        )

    def _generate_groups_recursive(self, current_group: Set[str],
                                   remaining_tasks: Set[str],
                                   seen_groups: Set[frozenset]) -> None:
        """Recursively generate all possible valid groups of tasks.
        
        Args:
            current_group: Set of tasks in the current group being built
            remaining_tasks: Set of tasks that haven't been considered yet
            seen_groups: Set of frozen sets tracking unique groups we've already processed
        """
        if current_group:
            # Convert current group to frozen set for hashing
            frozen_group = frozenset(current_group)
            if frozen_group not in seen_groups:
                seen_groups.add(frozen_group)
                # Calculate metrics for current group
                metrics = self._calculate_group_metrics(current_group)
                self.all_possible_groups.append(metrics)

        if not remaining_tasks:
            return

        for task in remaining_tasks:
            # Check if task can be added to current group
            if not current_group or all(self._can_be_grouped(self.tasks[t], self.tasks[task]) 
                                     for t in current_group):
                new_group = current_group | {task}
                if self._all_dependency_paths_within_group(new_group):
                    new_remaining = remaining_tasks - {task}
                    self._generate_groups_recursive(new_group, new_remaining, seen_groups)

    def generate_all_possible_groups(self) -> List[GroupMetrics]:
        """Generate all possible valid groups of tasks with their metrics"""
        all_tasks = set(self.tasks.keys())
        self.all_possible_groups = []
        seen_groups = set()  # Track unique groups using frozen sets

        self._generate_groups_recursive(set(), all_tasks, seen_groups)
        return self.all_possible_groups

    def get_group_metrics(self) -> List[dict]:
        """Get all possible groups with their metrics in a format suitable for visualization"""
        return [metrics.to_dict() for metrics in self.all_possible_groups]


def validate_task_parameters(task_data: dict, task_name: str) -> None:
    """Validate that all required parameters are present in the task data.
    
    Args:
        task_data: Dictionary containing the task specification
        task_name: Name of the task being validated
        
    Raises:
        ValueError: If any required parameter is missing
    """
    required_parameters = ["ScramArch", "TimePerEvent", "Memory", "Multicore", "SizePerEvent"]
    missing_parameters = [param for param in required_parameters if param not in task_data]
    
    if missing_parameters:
        raise ValueError(f"Missing required parameters for {task_name}: {', '.join(missing_parameters)}")


def get_events_per_job(task_data: dict, tasks: Dict[str, Task], task_name: str) -> int:
    """Recursively get EventsPerJob from task or its input task.
    
    Args:
        task_data: Dictionary containing the task specification
        tasks: Dictionary of already created tasks
        task_name: Name of the current task
        
    Returns:
        Number of events per job
    """
    if "EventsPerJob" in task_data:
        return task_data["EventsPerJob"]
    
    input_task = task_data.get("InputTask")
    if input_task and input_task in tasks:
        return get_events_per_job(tasks[input_task].resources.__dict__, tasks, input_task)
    
    raise ValueError(f"No EventsPerJob found for {task_name} or any of its input tasks")


def create_workflow_from_json(workflow_data: dict) -> Tuple[List[dict], Dict[str, Task]]:
    """Create a workflow of tasks from JSON data and generate all possible task groups with metrics.

    Args:
        workflow_data: Dictionary containing the workflow specification

    Returns:
        Tuple of (List of group metrics dictionaries, Dictionary of tasks)
    """
    # Create tasks dictionary
    tasks = {}
    for i in range(1, workflow_data["NumTasks"] + 1):
        task_name = f"Task{i}"
        task_data = workflow_data[task_name]

        # Validate required parameters
        validate_task_parameters(task_data, task_name)

        # Extract OS version and architecture from ScramArch
        os_version, cpu_arch = extract_os_and_arch(task_data["ScramArch"])

        # Calculate events per second from TimePerEvent
        time_per_event = task_data["TimePerEvent"]
        events_per_second = 1 / time_per_event

        # Create TaskResources
        resources = TaskResources(
            os_version=os_version,
            cpu_arch=cpu_arch,
            memory_mb=task_data["Memory"],
            accelerator="GPU" if task_data.get("RequiresGPU") == "required" else None,
            cpu_cores=task_data["Multicore"],
            events_per_second=events_per_second,
            time_per_event=time_per_event,
            size_per_event=task_data["SizePerEvent"],
            input_events=task_data.get("EventsPerJob", None) 
        )

        # Create Task
        tasks[task_name] = Task(
            id=task_name,
            resources=resources,
            input_task=task_data.get("InputTask", None),
            output_tasks=set()  # Will be populated later
        )

    # Add output tasks and resolve input events
    for task_name, task in tasks.items():
        # Handle output tasks
        if task.input_task:
            if tasks[task.input_task].output_tasks is None:
                tasks[task.input_task].output_tasks = set()
            tasks[task.input_task].output_tasks.add(task_name)
        
        # Resolve input events
        if task.resources.input_events is None and task.input_task:
            input_task = tasks[task.input_task]
            task.resources.input_events = input_task.resources.input_events
            if task.resources.input_events is None:
                raise ValueError(f"No EventsPerJob found for {task_name} or any of its input tasks")
    
    # Print content of each task
    for task_name, task in tasks.items():
        print(f"Final task creation for {task_name}:\n{pformat(task)}\n")

    # Create task grouper and generate all possible groups
    grouper = TaskGrouper(tasks)
    grouper.generate_all_possible_groups()
    
    # Return group metrics and tasks
    return grouper.get_group_metrics(), tasks


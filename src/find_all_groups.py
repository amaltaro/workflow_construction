# Standard library imports
import logging
import sys
from dataclasses import dataclass
from pprint import pformat
from typing import Dict, List, Optional, Set, Tuple

# Third-party library imports
import networkx as nx

# Global configuration
TARGET_WALLCLOCK_TIME_HOURS = 12.0  # Target wallclock time in hours

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
    keep_output: bool  # Whether the task's output needs to be kept in shared storage


@dataclass
class Task:
    """
    Class to store the task resources and dependencies
    """
    id: str
    resources: TaskResources
    input_task: Optional[str] = None
    output_tasks: Set[str] = None
    order: int = 0  # Order of the task in the workflow


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
    entry_point_task: str  # The first task to be executed in the group
    exit_point_task: str   # The last task to be executed in the group
    # CPU metrics
    max_cpu_cores: int
    cpu_seconds: float  # Total CPU-seconds required
    cpu_utilization_ratio: float  # Ratio of CPU cores utilized to CPU cores allocated over time
    # Memory metrics
    max_memory_mb: int
    min_memory_mb: int
    memory_occupancy: float  # Ratio of total memory used to max memory allocated
    # Throughput metrics
    total_throughput: float
    max_throughput: float
    min_throughput: float
    # I/O metrics
    input_data_mb: float  # Total input data volume in MB
    output_data_mb: float  # Total output data volume in MB
    stored_data_mb: float  # Total data volume that needs to be stored
    input_data_per_event_mb: float  # Input data per event in MB
    output_data_per_event_mb: float  # Output data per event in MB
    stored_data_per_event_mb: float  # Stored data per event in MB
    # Accelerator metrics
    accelerator_types: Set[str]
    # Dependency metrics
    dependency_paths: List[List[str]]
    # Resource utilization metrics
    resource_utilization: float  # Overall resource utilization efficiency
    event_throughput: float  # Events processed per second
    # Events per job
    events_per_job: int  # Number of events processed by each task in the group

    def to_dict(self) -> dict:
        """Convert metrics to a dictionary for easy serialization"""
        return {
            "group_id": self.group_id,
            "task_ids": sorted(list(self.task_ids)),
            "entry_point_task": self.entry_point_task,
            "exit_point_task": self.exit_point_task,
            "resource_metrics": {
                "cpu": {
                    "max_cores": self.max_cpu_cores,
                    "cpu_seconds": self.cpu_seconds,
                    "utilization_ratio": self.cpu_utilization_ratio
                },
                "memory": {
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
                    "input_data_mb": self.input_data_mb,
                    "output_data_mb": self.output_data_mb,
                    "stored_data_mb": self.stored_data_mb,
                    "input_data_per_event_mb": self.input_data_per_event_mb,
                    "output_data_per_event_mb": self.output_data_per_event_mb,
                    "stored_data_per_event_mb": self.stored_data_per_event_mb
                },
                "accelerator": {
                    "types": list(self.accelerator_types)
                }
            },
            "utilization_metrics": {
                "resource_utilization": self.resource_utilization,
                "event_throughput": self.event_throughput
            },
            "dependency_paths": self.dependency_paths,
            "events_per_job": self.events_per_job
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

    def _find_entry_point_task(self, tasks: List[Task]) -> str:
        """Find the entry point task in a group of tasks.

        The entry point task is the first task to be executed in the group.
        It is identified as a task that either:
        - Has no input task, or
        - Has an input task that is not in the current group

        Args:
            tasks: List of tasks in the group

        Returns:
            ID of the entry point task

        Raises:
            ValueError: If no entry point task is found (should never happen in a valid DAG)
        """
        for task in tasks:
            if not task.input_task or task.input_task not in {t.id for t in tasks}:
                return task.id
        raise ValueError("No entry point task found in the group")

    def _find_exit_point_task(self, tasks: List[Task]) -> str:
        """Find the exit point task in a group of tasks.

        The exit point task is the last task to be executed in the group.
        It is identified as a task that either:
        - Has no output tasks, or
        - Has output tasks that are all outside the current group

        Args:
            tasks: List of tasks in the group

        Returns:
            ID of the exit point task

        Raises:
            ValueError: If no exit point task is found (should never happen in a valid DAG)
        """
        task_ids = {t.id for t in tasks}
        for task in tasks:
            if not task.output_tasks or all(output_task not in task_ids for output_task in task.output_tasks):
                return task.id
        raise ValueError("No exit point task found in the group")

    def _calculate_group_metrics(self, group: Set[str]) -> GroupMetrics:
        """Calculate detailed metrics for a group of tasks"""
        group_id = f"group_{len(self.all_possible_groups)}"
        print(f"Calculating group metrics for {group_id} with tasks {group}")
        tasks = [self.tasks[task_id] for task_id in group]
        
        # Find entry and exit point tasks
        entry_point_task = self._find_entry_point_task(tasks)
        exit_point_task = self._find_exit_point_task(tasks)

        # Calculate total time per event for the entire group
        # TimePerEvent should already account for the actual performance with the given number of cores
        total_time_per_event = sum(task.resources.time_per_event for task in tasks)

        # Calculate events per job based on total time per event for the group
        target_wallclock_seconds = TARGET_WALLCLOCK_TIME_HOURS * 3600
        events_per_job = int(target_wallclock_seconds / total_time_per_event)
        events_per_job = max(1, events_per_job)

        # Update events_per_job for all tasks in the group
        for task in tasks:
            task.resources.input_events = events_per_job

        # Calculate CPU metrics
        max_cores = max(t.resources.cpu_cores for t in tasks)

        # Calculate CPU utilization ratio
        # For each task: (cores_used * duration) / (max_cores * total_duration)
        total_duration = 0.0
        weighted_cpu_utilization = 0.0

        for task in tasks:
            task_duration = task.resources.input_events * task.resources.time_per_event
            total_duration += task_duration
            weighted_cpu_utilization += task.resources.cpu_cores * task_duration

        # CPU utilization ratio is the ratio of weighted CPU utilization to maximum possible utilization
        max_possible_utilization = max_cores * total_duration
        cpu_utilization_ratio = weighted_cpu_utilization / max_possible_utilization if max_possible_utilization > 0 else 0.0

        # Calculate CPU seconds (total CPU time used)
        cpu_seconds = weighted_cpu_utilization

        # Calculate memory metrics
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
        total_events = sum(task.resources.input_events for task in tasks)
        total_throughput = total_events / cpu_seconds if cpu_seconds > 0 else 0.0

        # For individual task throughput
        task_throughputs = []
        for task in tasks:
            task_cpu_seconds = task.resources.cpu_cores * task.resources.time_per_event * task.resources.input_events
            task_throughput = task.resources.input_events / task_cpu_seconds if task_cpu_seconds > 0 else 0.0
            task_throughputs.append(task_throughput)

        max_throughput = max(task_throughputs)
        min_throughput = min(task_throughputs)

        ### Calculate I/O metrics with storage rules
        # Calculate input data volume based on:
        # - input events for the entry point task (using events_per_job)
        # - size per event of the parent task that feeds into the entry point task
        input_data_mb = 0.0
        entry_task = self.tasks[entry_point_task]
        if entry_task.input_task and entry_task.input_task in self.tasks:
            # Get the parent task that feeds into this group's entry point
            parent_task = self.tasks[entry_task.input_task]
            # Input data is based on the entry point task's events in this group
            # and the parent task's size per event
            input_data_mb = (events_per_job * parent_task.resources.size_per_event) / 1024.0
        # normalize the data volume per event
        input_data_per_event_mb = input_data_mb / events_per_job if events_per_job > 0 else 0.0

        # Calculate output sizes of all tasks in the group based on storage rules
        output_data_mb = 0.0
        stored_data_mb = 0.0
        for task in tasks:
            # Convert KB to MB for consistency with other memory metrics
            task_output_size = (task.resources.input_events * task.resources.size_per_event) / 1024.0
            output_data_mb += task_output_size

            # Add to stored data if:
            # * it has to save the output to the storage
            # * it is the exit point task of the group
            if task.resources.keep_output or task.id == exit_point_task:
                stored_data_mb += task_output_size
        # normalize the output and stored data volume per event
        output_data_per_event_mb = output_data_mb / total_events if total_events > 0 else 0.0
        stored_data_per_event_mb = stored_data_mb / total_events if total_events > 0 else 0.0

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
        resource_utilization = (cpu_utilization_ratio + memory_occupancy) / 2.0

        # Calculate event throughput
        # This is the total number of events processed per second
        event_throughput = total_throughput

        return GroupMetrics(
            group_id=group_id,
            task_ids=group,
            entry_point_task=entry_point_task,
            exit_point_task=exit_point_task,
            max_cpu_cores=max_cores,
            cpu_seconds=cpu_seconds,
            cpu_utilization_ratio=cpu_utilization_ratio,
            max_memory_mb=max_memory,
            min_memory_mb=min_memory,
            memory_occupancy=memory_occupancy,
            total_throughput=total_throughput,
            max_throughput=max_throughput,
            min_throughput=min_throughput,
            input_data_mb=input_data_mb,
            output_data_mb=output_data_mb,
            stored_data_mb=stored_data_mb,
            input_data_per_event_mb=input_data_per_event_mb,
            output_data_per_event_mb=output_data_per_event_mb,
            stored_data_per_event_mb=stored_data_per_event_mb,
            accelerator_types=accelerators,
            dependency_paths=dependency_paths,
            resource_utilization=resource_utilization,
            event_throughput=event_throughput,
            events_per_job=events_per_job
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

        # First, generate all single-task groups in order
        if not current_group:  # Only do this at the root level
            # Sort tasks by their order
            sorted_tasks = sorted(remaining_tasks, key=lambda t: self.tasks[t].order)
            for task in sorted_tasks:
                single_group = {task}
                frozen_group = frozenset(single_group)
                if frozen_group not in seen_groups:
                    seen_groups.add(frozen_group)
                    metrics = self._calculate_group_metrics(single_group)
                    self.all_possible_groups.append(metrics)

        # Then proceed with multi-task groups
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


def calculate_events_per_job(task_data: dict, tasks: Dict[str, Task], task_name: str) -> int:
    """Calculate optimal events per job based on target wallclock time.
    
    Args:
        task_data: Dictionary containing the task specification
        tasks: Dictionary of already created tasks
        task_name: Name of the current task
        
    Returns:
        Number of events per job that would result in target wallclock time
    """
    # Convert target wallclock time to seconds
    target_wallclock_seconds = TARGET_WALLCLOCK_TIME_HOURS * 3600
    
    # Get time per event for this task
    time_per_event = task_data["TimePerEvent"]
    
    # Calculate events per job that would result in target wallclock time
    events_per_job = int(target_wallclock_seconds / time_per_event)

    # Ensure we have at least 1 event per job
    return max(1, events_per_job)


def find_all_workflow_constructions(grouper: TaskGrouper) -> List[List[GroupMetrics]]:
    """Find all possible ways to run the workflow using the given groups.

    Args:
        grouper: TaskGrouper instance containing the DAG and groups

    Returns:
        List of lists, where each inner list represents a valid workflow construction
        (a set of groups that together contain all tasks, respecting dependencies)
    """
    # Get all unique task IDs from all groups
    all_tasks = set()
    for group in grouper.all_possible_groups:
        all_tasks.update(group.task_ids)

    # print(f"\nAll tasks that need to be covered: {all_tasks}")

    # Get topologically sorted tasks
    sorted_tasks = list(nx.topological_sort(grouper.dag))
    print(f"\nTopologically sorted tasks: {sorted_tasks}")

    # Find all possible combinations of groups that cover all tasks
    valid_constructions = []
    seen_constructions = set()  # Track unique constructions using frozen sets of group IDs

    def get_available_tasks(construction: List[GroupMetrics]) -> Set[str]:
        """Get tasks that are available to be added to the construction.
        A task is available if all its dependencies are already in the construction.
        """
        # Get all tasks already in the construction
        tasks_in_construction = set()
        for group in construction:
            tasks_in_construction.update(group.task_ids)

        # Find tasks whose dependencies are all satisfied
        available_tasks = set()
        for task in all_tasks - tasks_in_construction:
            predecessors = set(nx.ancestors(grouper.dag, task))
            if predecessors.issubset(tasks_in_construction):
                available_tasks.add(task)

        return available_tasks

    def get_valid_groups(construction: List[GroupMetrics], available_tasks: Set[str]) -> List[GroupMetrics]:
        """Get all groups that could be added to the construction.
        A group is valid if:
        1. It contains at least one available task
        2. It doesn't contain any task already in the construction
        """
        tasks_in_construction = set()
        for group in construction:
            tasks_in_construction.update(group.task_ids)

        valid_groups = []
        for group in grouper.all_possible_groups:
            # Check if group contains any available task and doesn't overlap with current construction
            if group.task_ids & available_tasks and not (group.task_ids & tasks_in_construction):
                valid_groups.append(group)

        return valid_groups

    def find_constructions(current_construction: List[GroupMetrics],
                          start_idx: int) -> None:
        """Recursively find all valid workflow constructions.

        Args:
            current_construction: Current partial construction being built
            start_idx: Index to start looking for next group from
        """
        # Get tasks already in the construction
        tasks_in_construction = set()
        for group in current_construction:
            tasks_in_construction.update(group.task_ids)

        # If all tasks are covered, we have a valid construction
        if tasks_in_construction == all_tasks:
            # Create a frozen set of group IDs for uniqueness check
            construction_key = frozenset(g.group_id for g in current_construction)
            if construction_key not in seen_constructions:
                seen_constructions.add(construction_key)
                valid_constructions.append(current_construction.copy())
            return

        # Get tasks that are available to be added
        available_tasks = get_available_tasks(current_construction)
        if not available_tasks:
            return

        # Get all valid groups that could be added
        valid_groups = get_valid_groups(current_construction, available_tasks)

        # Try adding each valid group
        for group in valid_groups:
            current_construction.append(group)
            find_constructions(current_construction, start_idx + 1)
            current_construction.pop()

    # Start the recursive search
    find_constructions([], 0)

    # Sort constructions by number of groups (ascending) and then by event throughput (descending)
    valid_constructions.sort(key=lambda x: (len(x), -sum(g.event_throughput for g in x)))

    print(f"Found {len(valid_constructions)} valid constructions:")
    for i, construction in enumerate(valid_constructions, 1):
        print(f"Construction {i}: {[g.group_id for g in construction]}")
        # Print tasks in each group for verification
        for group in construction:
            print(f"  {group.group_id}: {group.task_ids}")

    return valid_constructions


def calculate_workflow_metrics(construction: List[GroupMetrics]) -> dict:
    """Calculate overall workflow metrics for a given construction.

    Args:
        construction: List of groups representing a workflow construction

    Returns:
        Dictionary containing workflow metrics
    """
    # Calculate total events and total CPU time
    # Each task in a group processes the same number of events
    total_events = sum(len(group.task_ids) * group.events_per_job for group in construction)
    total_cpu_time = sum(group.cpu_seconds for group in construction)

    # Calculate overall event throughput
    event_throughput = total_events / total_cpu_time if total_cpu_time > 0 else 0.0

    # Calculate total data volumes
    total_input_data = sum(group.input_data_mb for group in construction)
    total_output_data = sum(group.output_data_mb for group in construction)
    total_stored_data = sum(group.stored_data_mb for group in construction)

    # Calculate per-event metrics by summing up each group's per-event metrics
    input_data_per_event = sum(group.input_data_per_event_mb for group in construction)
    output_data_per_event = sum(group.output_data_per_event_mb for group in construction)
    stored_data_per_event = sum(group.stored_data_per_event_mb for group in construction)

    # Create detailed group information
    group_details = []
    for group in construction:
        group_details.append({
            "group_id": group.group_id,
            "tasks": sorted(list(group.task_ids)),
            "events_per_task": group.events_per_job,
            "total_events": len(group.task_ids) * group.events_per_job,
            "cpu_seconds": group.cpu_seconds,
            "input_data_mb": group.input_data_mb,
            "output_data_mb": group.output_data_mb,
            "stored_data_mb": group.stored_data_mb,
            "input_data_per_event_mb": group.input_data_per_event_mb,
            "output_data_per_event_mb": group.output_data_per_event_mb,
            "stored_data_per_event_mb": group.stored_data_per_event_mb
        })

    return {
        "total_events": total_events,
        "total_cpu_time": total_cpu_time,
        "event_throughput": event_throughput,
        "total_input_data_mb": total_input_data,
        "total_output_data_mb": total_output_data,
        "total_stored_data_mb": total_stored_data,
        "input_data_per_event_mb": input_data_per_event,
        "output_data_per_event_mb": output_data_per_event,
        "stored_data_per_event_mb": stored_data_per_event,
        "initial_input_events": construction[0].events_per_job if construction else 0,
        "num_groups": len(construction),
        "groups": [group.group_id for group in construction],
        "group_details": group_details
    }


def create_workflow_from_json(workflow_data: dict) -> Tuple[List[dict], Dict[str, Task], List[dict], nx.DiGraph]:
    """Create a workflow of tasks from JSON data and generate all possible task groups with metrics.

    Args:
        workflow_data: Dictionary containing the workflow specification

    Returns:
        Tuple of (List of group metrics dictionaries, Dictionary of tasks, List of construction metrics, DAG)
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

        # Calculate events per job based on target wallclock time
        events_per_job = calculate_events_per_job(task_data, tasks, task_name)

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
            input_events=events_per_job,
            keep_output=task_data.get("KeepOutput", True)  # Default to True if not specified
        )

        # Create Task with order information
        tasks[task_name] = Task(
            id=task_name,
            resources=resources,
            input_task=task_data.get("InputTask", None),
            output_tasks=set(),  # Will be populated later
            order=i  # Store the order based on task number
        )

    # Add output tasks
    for task_name, task in tasks.items():
        if task.input_task:
            if tasks[task.input_task].output_tasks is None:
                tasks[task.input_task].output_tasks = set()
            tasks[task.input_task].output_tasks.add(task_name)
    
    # Print content of each task
    for task_name, task in tasks.items():
        print(f"Final task creation for {task_name}:\n{pformat(task)}\n")

    # Create task grouper and generate all possible groups
    grouper = TaskGrouper(tasks)
    grouper.generate_all_possible_groups()

    # Find all possible workflow constructions
    all_constructions = find_all_workflow_constructions(grouper)

    # Calculate metrics for each construction
    construction_metrics = [calculate_workflow_metrics(construction) for construction in all_constructions]

    # Print the results
    print("\nPossible workflow constructions and their metrics:")
    for i, metrics in enumerate(construction_metrics, 1):
        print(f"\nConstruction {i}:")
        print(f"  Groups: {metrics['groups']}")
        print(f"  Total Events: {metrics['total_events']}")
        print(f"  Total CPU Time: {metrics['total_cpu_time']:.2f} seconds")
        print(f"  Event Throughput: {metrics['event_throughput']:.3f} events/second")
        print(f"  Total Input Data: {metrics['total_input_data_mb']:.2f} MB")
        print(f"  Total Output Data: {metrics['total_output_data_mb']:.2f} MB")
        print(f"  Total Stored Data: {metrics['total_stored_data_mb']:.2f} MB")
        print(f"  Input Data per Event: {metrics['input_data_per_event_mb']:.3f} MB/event")
        print(f"  Output Data per Event: {metrics['output_data_per_event_mb']:.3f} MB/event")
        print(f"  Stored Data per Event: {metrics['stored_data_per_event_mb']:.3f} MB/event")
        print("  Group Details:")
        for group in metrics['group_details']:
            print(f"    {group['group_id']}:")
            print(f"      Tasks: {group['tasks']}")
            print(f"      Events per Task: {group['events_per_task']}")
            print(f"      Total Events: {group['total_events']}")
            print(f"      CPU Time: {group['cpu_seconds']:.2f} seconds")
            print(f"      Input Data: {group['input_data_mb']:.2f} MB")
            print(f"      Output Data: {group['output_data_mb']:.2f} MB")
            print(f"      Stored Data: {group['stored_data_mb']:.2f} MB")

    # Return group metrics, tasks, construction metrics, and the DAG
    return grouper.get_group_metrics(), tasks, construction_metrics, grouper.dag


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
    read_remote_mb: float  # Total remote read data volume in MB
    write_local_mb: float  # Total local write data volume in MB
    write_remote_mb: float  # Total remote write data volume in MB
    read_remote_per_event_mb: float  # Remote read data per event in MB
    write_local_per_event_mb: float  # Local write data per event in MB
    write_remote_per_event_mb: float  # Remote write data per event in MB
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
                    "read_remote_mb": self.read_remote_mb,
                    "write_local_mb": self.write_local_mb,
                    "write_remote_mb": self.write_remote_mb,
                    "read_remote_per_event_mb": self.read_remote_per_event_mb,
                    "write_local_per_event_mb": self.write_local_per_event_mb,
                    "write_remote_per_event_mb": self.write_remote_per_event_mb
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
        """
        Check if tasks can be grouped based on hard requirements, such as:
        - direct dependency path between them in both directions
        - same OS version
        - same CPU architecture
        """
        # do these tasks have a direct dependency path between them in both directions?
        if not (nx.has_path(self.dag, task1.id, task2.id) or nx.has_path(self.dag, task2.id, task1.id)):
            return False
        # do these tasks request the same OS version?
        if task1.resources.os_version != task2.resources.os_version:
            return False
        # do these tasks request the same CPU architecture?
        if task1.resources.cpu_arch != task2.resources.cpu_arch:
            return False
        # FIXME: do these tasks request the same accelerator?

        return True

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
        sorted_group = sorted(list(group))
        print(f"Calculating group metrics for {group_id} with tasks {sorted_group}")
        tasks = [self.tasks[task_id] for task_id in sorted_group]

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

        # NOTE: Update events_per_job for all tasks in the group
        for task in tasks:
            task.resources.input_events = events_per_job

        # Calculate CPU metrics
        max_cores = max(t.resources.cpu_cores for t in tasks)

        # Calculate CPU seconds (total CPU time used)
        # The group is allocated max_cores for the entire duration
        total_wallclock_time = events_per_job * total_time_per_event
        cpu_seconds = max_cores * total_wallclock_time

        # Calculate CPU utilization ratio and time-weighted memory occupancy
        # For each task: (cores_used * duration) / (max_cores * total_duration)
        total_duration = 0.0
        weighted_cpu_utilization = 0.0
        weighted_memory = 0.0
        for task in tasks:
            task_duration = events_per_job * task.resources.time_per_event
            total_duration += task_duration
            weighted_cpu_utilization += task.resources.cpu_cores * task_duration
            weighted_memory += task.resources.memory_mb * task_duration

        # CPU utilization ratio is the ratio of weighted CPU utilization to maximum possible utilization
        max_possible_utilization = max_cores * total_duration
        cpu_utilization_ratio = weighted_cpu_utilization / max_possible_utilization

        # Calculate memory metrics
        max_memory = max(t.resources.memory_mb for t in tasks)
        min_memory = min(t.resources.memory_mb for t in tasks)

        # Calculate time-weighted average memory
        time_weighted_avg_memory = weighted_memory / total_duration
        # Memory occupancy is the ratio of time-weighted average memory to max memory
        memory_occupancy = time_weighted_avg_memory / max_memory

        # Calculate throughput metrics
        # Use events_per_job for consistency with workflow construction metrics
        total_throughput = events_per_job / cpu_seconds

        # For individual task throughput (note that all tasks process the same number of events)
        task_throughputs = []
        for task in tasks:
            task_cpu_seconds = task.resources.cpu_cores * task.resources.time_per_event * events_per_job
            task_throughput = events_per_job / task_cpu_seconds
            task_throughputs.append(task_throughput)
        max_throughput = max(task_throughputs)
        min_throughput = min(task_throughputs)

        ### Calculate I/O metrics with storage rules
        # Calculate remote read data volume based on:
        # - input events for the entry point task (using events_per_job)
        # - size per event of the parent task that feeds into the entry point task
        read_remote_mb = 0.0
        entry_task = self.tasks[entry_point_task]
        if entry_task.input_task and entry_task.input_task in self.tasks:
            # Get the parent task that feeds into this group's entry point
            parent_task = self.tasks[entry_task.input_task]
            read_remote_mb = (events_per_job * parent_task.resources.size_per_event) / 1024.0
        # normalize the data volume per event
        read_remote_per_event_mb = read_remote_mb / events_per_job

        # Calculate local write and remote write sizes of all tasks in the group based on storage rules
        write_local_mb = 0.0
        write_remote_mb = 0.0
        for task in tasks:
            # Convert KB to MB for consistency with other memory metrics
            task_output_size = (events_per_job * task.resources.size_per_event) / 1024.0
            write_local_mb += task_output_size

            # Add to remote write data if:
            # * it has to save the output to the storage
            # * it is the exit point task of the group
            if task.resources.keep_output or task.id == exit_point_task:
                write_remote_mb += task_output_size
        # normalize the local write and remote write data volume per event
        # Use events_per_job for consistency with throughput and workflow construction metrics
        write_local_per_event_mb = write_local_mb / events_per_job
        write_remote_per_event_mb = write_remote_mb / events_per_job

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
        # This is the number of events processed per second (consistent with workflow construction)
        event_throughput = total_throughput

        return GroupMetrics(
            group_id=group_id,
            task_ids=sorted_group,
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
            read_remote_mb=read_remote_mb,
            write_local_mb=write_local_mb,
            write_remote_mb=write_remote_mb,
            read_remote_per_event_mb=read_remote_per_event_mb,
            write_local_per_event_mb=write_local_per_event_mb,
            write_remote_per_event_mb=write_remote_per_event_mb,
            accelerator_types=accelerators,
            dependency_paths=dependency_paths,
            resource_utilization=resource_utilization,
            event_throughput=event_throughput,
            events_per_job=events_per_job
        )

    def generate_all_possible_groups(self) -> List[List[str]]:
        """
        Generate all possible valid groups of tasks with their metrics using a deterministic algorithm.

        This algorithm ensures:
        1. Deterministic group generation order
        2. Deterministic group composition
        3. Reproducible results across different runs
        """
        all_tasks = set(self.tasks.keys())
        valid_groups = []

        # Sort tasks by order for deterministic behavior
        sorted_task_ids = sorted(all_tasks, key=lambda t: self.tasks[t].order)
        print(f"Generating groups for {len(sorted_task_ids)} tasks in deterministic order: {sorted_task_ids}")

        # Generate all possible subsets systematically
        from itertools import combinations

        for size in range(1, len(sorted_task_ids) + 1):
            print(f"Generating groups of size {size}...")
            for task_combo in combinations(sorted_task_ids, size):
                group = set(task_combo)

                # Validate the group
                if self._is_valid_group(group):
                    # Convert to sorted list for deterministic ordering
                    sorted_group = sorted(list(group))
                    valid_groups.append(sorted_group)
                    print(f"  Added valid group: {sorted_group}")

        print(f"Generated {len(valid_groups)} valid groups\n")
        return valid_groups

    def _is_valid_group(self, group: Set[str]) -> bool:
        """
        Check if a group of tasks is valid for grouping.

        Args:
            group: Set of task IDs to validate

        Returns:
            True if the group is valid, False otherwise
        """
        task_list = list(group)

        # Check if all tasks can be grouped together
        for i, task1 in enumerate(task_list):
            for task2 in task_list[i+1:]:
                if not self._can_be_grouped(self.tasks[task1], self.tasks[task2]):
                    return False

        # Check if all dependency paths are contained within the group
        return self._all_dependency_paths_within_group(group)

    def calculate_metrics_for_groups(self, groups: List[List[str]]):
        """
        Calculate metrics for a list of task groups.

        Args:
            groups: List of task groups (lists of task IDs)

        """
        for group in groups:
            # Convert list to set for the existing _calculate_group_metrics method
            group_set = set(group)
            metrics = self._calculate_group_metrics(group_set)
            self.all_possible_groups.append(metrics)

        # Sort by group_id for consistent ordering
        self.all_possible_groups.sort(key=lambda g: g.group_id)

        print(f"Calculated metrics for {len(self.all_possible_groups)} groups")


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
            if set(group.task_ids) & available_tasks and not (set(group.task_ids) & tasks_in_construction):
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


def calculate_workflow_metrics(construction: List[GroupMetrics], request_num_events: int) -> dict:
    """Calculate overall workflow metrics for a given construction.

    Args:
        construction: List of groups representing a workflow construction
        request_num_events: Total number of events to process in the workflow

    Returns:
        Dictionary containing workflow metrics
    """
    if not construction:
        raise RuntimeError("No construction provided")

    # Calculate how many jobs each group needs to run to process the requested events
    group_jobs_needed = {}
    for group in construction:
        jobs_needed = request_num_events / group.events_per_job
        group_jobs_needed[group.group_id] = jobs_needed

    # Calculate total CPU time for the entire workflow (accounting for job scaling)
    total_cpu_time = sum(
        group.cpu_seconds * group_jobs_needed[group.group_id]
        for group in construction
    )

    # Calculate normalized CPU time per event
    cpu_time_per_event = total_cpu_time / request_num_events

    # Calculate total events (should be the requested number of events)
    total_events = request_num_events

    # Calculate overall event throughput using the total CPU time
    # Event throughput = request_num_events / total_cpu_time
    # This provides a consistent measure of events processed per second
    event_throughput = request_num_events / total_cpu_time

    # Calculate total data volumes (accounting for job scaling)
    total_read_remote = sum(
        group.read_remote_mb * group_jobs_needed[group.group_id]
        for group in construction
    )
    total_write_local = sum(
        group.write_local_mb * group_jobs_needed[group.group_id]
        for group in construction
    )
    total_write_remote = sum(
        group.write_remote_mb * group_jobs_needed[group.group_id]
        for group in construction
    )

    # Calculate per-event metrics by normalizing the total data volumes
    read_remote_per_event = total_read_remote / request_num_events
    write_local_per_event = total_write_local / request_num_events
    write_remote_per_event = total_write_remote / request_num_events

    # Calculate memory scaling metrics
    total_memory_mb = sum(
        group.max_memory_mb * group_jobs_needed[group.group_id]
        for group in construction
    )
    memory_per_event_mb = total_memory_mb / request_num_events

    # Calculate network transfer metrics
    total_network_transfer_mb = total_read_remote + total_write_remote
    network_transfer_per_event_mb = total_network_transfer_mb / request_num_events

    # Calculate efficiency metrics
    total_jobs = sum(group_jobs_needed.values())
    jobs_per_event = total_jobs / request_num_events

    # Calculate wallclock time (assuming 12-hour target)
    total_wallclock_time = sum(
        (TARGET_WALLCLOCK_TIME_HOURS * 3600) * group_jobs_needed[group.group_id]
        for group in construction
    )
    wallclock_time_per_event = total_wallclock_time / request_num_events

    # Create detailed group information
    group_details = []
    for group in construction:
        group_details.append({
            "group_id": group.group_id,
            "tasks": sorted(list(group.task_ids)),
            "events_per_task": group.events_per_job,
            "total_events": group.events_per_job,  # Events processed by this group per job
            "cpu_seconds": group.cpu_seconds,
            "read_remote_mb": group.read_remote_mb,
            "write_local_mb": group.write_local_mb,
            "write_remote_mb": group.write_remote_mb,
            "read_remote_per_event_mb": group.read_remote_per_event_mb,
            "write_local_per_event_mb": group.write_local_per_event_mb,
            "write_remote_per_event_mb": group.write_remote_per_event_mb
        })

    return {
        "request_num_events": request_num_events,
        "total_events": total_events,
        "total_cpu_time": total_cpu_time,
        "cpu_time_per_event": cpu_time_per_event,
        "event_throughput": event_throughput,
        "total_read_remote_mb": total_read_remote,
        "total_write_local_mb": total_write_local,
        "total_write_remote_mb": total_write_remote,
        "read_remote_per_event_mb": read_remote_per_event,
        "write_local_per_event_mb": write_local_per_event,
        "write_remote_per_event_mb": write_remote_per_event,
        "total_memory_mb": total_memory_mb,
        "memory_per_event_mb": memory_per_event_mb,
        "total_network_transfer_mb": total_network_transfer_mb,
        "network_transfer_per_event_mb": network_transfer_per_event_mb,
        "total_jobs": total_jobs,
        "jobs_per_event": jobs_per_event,
        "total_wallclock_time": total_wallclock_time,
        "wallclock_time_per_event": wallclock_time_per_event,
        "group_jobs_needed": group_jobs_needed,
        "initial_input_events": construction[0].events_per_job,
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
    # Get RequestNumEvents with fallback for backward compatibility
    request_num_events = workflow_data.get("RequestNumEvents", 1000000)
    # Create tasks dictionary
    tasks = {}
    for i in range(1, workflow_data["NumTasks"] + 1):
        task_name = f"Taskset{i}"
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
            input_task=task_data.get("InputTaskset", None),
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
    all_groups = grouper.generate_all_possible_groups()
    grouper.calculate_metrics_for_groups(all_groups)

    # Find all possible workflow constructions
    all_constructions = find_all_workflow_constructions(grouper)

    # Calculate metrics for each construction
    construction_metrics = [calculate_workflow_metrics(construction, request_num_events) for construction in all_constructions]

    # Print the results
    print(f"\nRequested Events: {request_num_events:,}")
    print("\nPossible workflow constructions and their metrics:")
    for i, metrics in enumerate(construction_metrics, 1):
        print(f"\nConstruction {i}:")
        print(f"  Groups: {metrics['groups']}")
        print(f"  Requested Events: {metrics['request_num_events']:,}")
        print(f"  Total CPU Time: {metrics['total_cpu_time']:,.2f} seconds")
        print(f"  CPU Time per Event: {metrics['cpu_time_per_event']:.4f} seconds/event")
        print(f"  Event Throughput: {metrics['event_throughput']:.4f} events/second")
        print(f"  Total Remote Read Data: {metrics['total_read_remote_mb']:.2f} MB")
        print(f"  Total Local Write Data: {metrics['total_write_local_mb']:.2f} MB")
        print(f"  Total Remote Write Data: {metrics['total_write_remote_mb']:.2f} MB")
        print(f"  Remote Read Data per Event: {metrics['read_remote_per_event_mb']:.3f} MB/event")
        print(f"  Local Write Data per Event: {metrics['write_local_per_event_mb']:.3f} MB/event")
        print(f"  Remote Write Data per Event: {metrics['write_remote_per_event_mb']:.3f} MB/event")
        print(f"  Total Memory: {metrics['total_memory_mb']:,.0f} MB")
        print(f"  Memory per Event: {metrics['memory_per_event_mb']:.3f} MB/event")
        print(f"  Total Network Transfer: {metrics['total_network_transfer_mb']:,.0f} MB")
        print(f"  Network Transfer per Event: {metrics['network_transfer_per_event_mb']:.3f} MB/event")
        print(f"  Total Jobs: {metrics['total_jobs']:.0f}")
        print(f"  Jobs per Event: {metrics['jobs_per_event']:.6f}")
        print(f"  Total Wallclock Time: {metrics['total_wallclock_time']:,.0f} seconds")
        print(f"  Wallclock Time per Event: {metrics['wallclock_time_per_event']:.6f} seconds/event")
        print("  Group Jobs Needed:")
        for group_id, jobs_needed in metrics['group_jobs_needed'].items():
            print(f"    {group_id}: {jobs_needed:.1f} jobs")
        print("  Group Details:")
        for group in metrics['group_details']:
            print(f"    {group['group_id']}:")
            print(f"      Tasks: {group['tasks']}")
            print(f"      Events per Task: {group['events_per_task']}")
            print(f"      Total Events: {group['total_events']}")
            print(f"      CPU Time: {group['cpu_seconds']:.2f} seconds")
            print(f"      Remote Read Data: {group['read_remote_mb']:.2f} MB")
            print(f"      Local Write Data: {group['write_local_mb']:.2f} MB")
            print(f"      Remote Write Data: {group['write_remote_mb']:.2f} MB")

    # Return group metrics, tasks, construction metrics, and the DAG
    return grouper.get_group_metrics(), tasks, construction_metrics, grouper.dag


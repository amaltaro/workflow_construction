import json
import pulp
from pprint import pformat

# Load workflow from JSON
IN_FILE = "./tests/ex1_cpu.json"
with open(IN_FILE, "r") as file:
    workflow = json.load(file)

# List of tasks in the workflow
tasks = ["T1", "T2", "T3", "T4", "T5"]
# List of (T_i, T_j) task dependency pairs in the workflow
# [("T1", "T2"), ("T2", "T3"), ("T3", "T4"), ("T4", "T5")]
dependencies = [("T1", "T2"), ("T2", "T3"), ("T1", "T4"), ("T4", "T5")]

# map tasks to indices, then map the dependencies as well
task_to_idx = {task: idx for idx, task in enumerate(tasks)}
dependency_to_idx = {(task_to_idx[i], task_to_idx[j]) for i, j in dependencies}

# initialize the hard constraint matrix based on the amount of tasks
hard_constraints = [[None for _ in range(len(tasks))] for _ in range(len(tasks))]
# also initialize the soft constraint matrix with a default value of 0.9
soft_constraints = [[0.9 for _ in range(len(tasks))] for _ in range(len(tasks))]

# calculate the hard constraints from the dependencies
# also hard-code soft constraints to 0.9
for i in range(len(tasks)):
    for j in range(len(tasks)):
        if i == j:
            hard_constraints[i][j] = 1
            soft_constraints[i][j] = 1
        else:
            hard_constraints[i][j] = 1 if (i, j) in dependency_to_idx else 0
            soft_constraints[i][j] = 0.9

print(f"hard_constraints: \n{pformat(hard_constraints)}")
print(f"soft_constraints: \n{pformat(soft_constraints)}")

num_tasks = len(tasks)
task_ids = range(num_tasks)

# Define the PuLP problem
prob = pulp.LpProblem("Task_Grouping_Optimization", pulp.LpMaximize)

# Decision variable: x[i][g] = 1 if task i is in group g
x = pulp.LpVariable.dicts("x", ((i, g) for i in task_ids for g in task_ids), cat="Binary")
print(f"decision variables: \n{pformat(x)}")

# Constraint 1: Each task is assigned to exactly one group
for i in task_ids:
    prob += pulp.lpSum(x[i, g] for g in task_ids) == 1

# Constraint 2: Tasks in the same group must respect hard constraints
for i in task_ids:
    for j in task_ids:
        if i != j:
            for g in task_ids:
                prob += x[i, g] + x[j, g] <= 1 + hard_constraints[i][j]

# Constraint 3: Dependent tasks must be in the same or sequential groups
print(f"dependencies: {dependencies}")
print(f"dependency_to_idx: {dependency_to_idx}")
for (i, j) in dependency_to_idx:
    for g in task_ids:
        prob += x[i, g] <= x[j, g] + x[j, g + 1] if g + 1 < num_tasks else x[i, g]

# Create new binary variables for pairs of tasks in the same group
y = pulp.LpVariable.dicts("y", 
    ((i, j, g) for i in task_ids for j in task_ids for g in task_ids), 
    cat="Binary"
)

# Add linearization constraints
for i in task_ids:
    for j in task_ids:
        for g in task_ids:
            # y[i,j,g] = 1 only if both x[i,g] and x[j,g] are 1
            prob += y[i,j,g] <= x[i,g]
            prob += y[i,j,g] <= x[j,g]
            prob += y[i,j,g] >= x[i,g] + x[j,g] - 1

# Modified objective function using y instead of x[i,g] * x[j,g]
prob += (
    pulp.lpSum(soft_constraints[i][j] * y[i,j,g] 
               for i in task_ids 
               for j in task_ids 
               for g in task_ids)
    - 0.1 * pulp.lpSum(x[i,g] for i in task_ids for g in task_ids)
)

# Solve the problem
prob.solve()

# Print results
print("\nOptimal Task Grouping:")
for g in task_ids:
    group = [tasks[i] for i in task_ids if pulp.value(x[i, g]) == 1]
    if group:
        print(f"Group {g + 1}: {group}")

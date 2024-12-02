import numpy as np
from gurobipy import Model, GRB

# Set a seed for reproducibility
np.random.seed(42)

# supply chain logistics optimization problem parameters
num_locations = 5
num_products = 5  #  5 products

# Provided dataset
product_demand = np.array([500, 300, 700, 400, 600])
initial_inventory = np.array([1000, 800, 1200, 600, 1000])
transportation_costs = np.array([[0, 500, 0, 0, 0],
                                  [0, 0, 450, 0, 0],
                                  [0, 0, 0, 600, 0],
                                  [0, 0, 0, 0, 550],
                                  [0, 0, 0, 0, 0]])
delivery_time_constraints = np.array([3, 2, 4, 3, 2])

# Genetic Algorithm parameters
population_size = 50
num_generations = 100
crossover_probability = 0.8
mutation_probability = 0.1

# Genetic Algorithm functions
def initialize_population():
    return [np.random.permutation(range(num_locations)) for _ in range(population_size)]

def crossover(parent1, parent2):
    crossover_point = np.random.randint(0, num_locations - 1)
    child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    return child

def mutate(individual):
    mutation_point1 = np.random.randint(0, num_locations - 1)
    mutation_point2 = np.random.randint(0, num_locations - 1)
    individual[mutation_point1], individual[mutation_point2] = individual[mutation_point2], individual[mutation_point1]
    return individual

# Genetic Algorithm
population = initialize_population()

# Define the Gurobi optimization model
model = Model("SupplyChainOptimization")

# Decision variables
x = {}
for i in range(num_locations):
    for j in range(num_locations):
        for k in range(num_products):
            x[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")

# Objective function (minimize transportation costs)
model.setObjective(
    sum(transportation_costs[i, j] * x[i, j, k] for i in range(num_locations) for j in range(num_locations) for k in range(num_products)),
    GRB.MINIMIZE
)

# Constraints
# Demand constraints
for k in range(num_products):
    model.addConstr(
        sum(x[i, j, k] for i in range(num_locations) for j in range(num_locations)) == 1,
        f"ProductSource_{k}"
    )

# Inventory constraints
for i in range(len(initial_inventory)):  # Iterate up to the size of initial_inventory
    for k in range(num_products):
        model.addConstr(
            sum(x[i, j, k] for j in range(num_locations)) <= initial_inventory[i],
            f"Inventory_{i}_{k}"
        )

# Delivery time constraints
for i in range(num_locations):
    model.addConstr(
        sum(x[i, j, k] * delivery_time_constraints[i] for j in range(num_locations) for k in range(num_products)) <= 7,
        f"DeliveryTime_{i}"
    )

# Main Genetic Algorithm loop
for generation in range(num_generations):
    # Evaluate the fitness of each individual in the population
    fitness_scores = []

    for individual in population:
        # Re-initialize the Gurobi model for each individual
        model.reset()

        # Assign values to decision variables based on the current individual
        for i in range(num_locations):
            for j in range(num_locations):
                for k in range(num_products):
                    x[i, j, k].VarName = f"x_{i}_{j}_{k}"
                    x[i, j, k].Start = individual[i]

        # Optimize the model
        model.optimize()

        # Check if the optimization was successful
        if model.status == GRB.OPTIMAL:
            fitness_scores.append(model.ObjVal)
        else:
            # Handle infeasible or other cases
            fitness_scores.append(float('inf'))

    # Select the top-performing individuals
    selected_indices = np.argsort(fitness_scores)[:int(population_size * 0.2)]

    # Create the next generation through crossover and mutation
    next_generation = [population[i] for i in selected_indices]

    while len(next_generation) < population_size:
        parent1 = population[np.random.choice(selected_indices)]
        parent2 = population[np.random.choice(selected_indices)]
        child = crossover(parent1, parent2)
        if np.random.random() < mutation_probability:
            child = mutate(child)
        next_generation.append(child)

    population = next_generation

    # Print intermediate results
    print(f"Generation {generation + 1}, Best Solution: {min(fitness_scores)}")

    # Add a convergence criterion
    if generation > 10 and min(fitness_scores[-10:]) == min(fitness_scores[-5:]):
        print("Convergence reached. Exiting the loop.")
        break

# Get the best solution from the final population
best_solution = population[np.argmin(fitness_scores)]

# Print or use the best_solution for further analysis
print("Best Solution:", best_solution)


#output#
"text": [Set parameter Username
Academic license - for non-commercial use only - expires 2024-10-26
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.03s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.12 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xeeee569e
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xd8562053
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xf8330f5a
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xaec45516
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x47289cf4
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x1a521736
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xbd5c1249
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5ba151cd
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb0262faf
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xf1ce694c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xea686356
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xeeee569e
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xac392858
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x1fc3040e
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xd33b2da7
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x3f136579
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x17442045
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x3663889a
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x7b14c7dc
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xec0e767d
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xac392858
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xba6967d4
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x49608635
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xf3d8b46a
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x6cb4b9d0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xebfa8b50
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x45ec1682
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xe645db53
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xebfa8b50
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x303874c5
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5ff9b8e7
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xbb8cb7ba
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xe645db53
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xcaa4c03f
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xe645db53
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x224bed61
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xf5c4942a
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x4319ba2e
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x51ffa413
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x18fb70f5
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x05e49d6a
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x357ced4a
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xefcd3ea1
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x6cb2b696
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x050d7496
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xe5c6ebb3
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5ba151cd
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Generation 1, Best Solution: 0.0
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xebfa8b50
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x45ec1682
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xe645db53
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xebfa8b50
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x303874c5
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5ff9b8e7
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xbb8cb7ba
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xe645db53
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xcaa4c03f
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xebfa8b50
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5f3a16e6
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 24.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x8d9e3574
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x1398b0be
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xbb8cb7ba
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.04 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xf5c4942a
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5cd61e06
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.05 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xf08b225a
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.02s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xffb03619
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x262d3ea5
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x02f32444
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x02f32444
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x0c1bdc92
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x990427e9
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x4ecce073
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 54.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xe645db53
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x6042120b
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xcaa4c03f
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xd81c8a2f
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 54.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xebfa8b50
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xe645db53
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.04 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x02f32444
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xcb9c200c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 64.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb982c251
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

Warning: Completing partial solution with 125 unfixed non-continuous variables out of 125
User MIP start produced solution with objective 0 (0.02s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xf5aca081
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xebfa8b50
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5cd61e06
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5d631a45
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 69.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xffb03619
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x303874c5
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xebfa8b50
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5f60157b
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x13db5002
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 54.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xca9be56d
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xea686356
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x150eea51
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 54.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5ff9b8e7
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x8ab802ca
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.02s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Generation 2, Best Solution: 0.0
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x6042120b
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xcaa4c03f
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xd81c8a2f
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 54.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xebfa8b50
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xe645db53
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x02f32444
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xcb9c200c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 64.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb982c251
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

Warning: Completing partial solution with 125 unfixed non-continuous variables out of 125
User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xf5aca081
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb4275fbc
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x02f32444
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb9b67a85
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xcb9c200c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 64.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xfddfd269
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 54.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xfaa727ae
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 64.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x02f32444
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xd81c8a2f
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 54.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x4c8da590
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x02f32444
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x45ec1682
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xe645db53
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xcb9c200c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 64.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5cd61e06
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x3e3f7faa
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xc90f0220
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x7b950e8d
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 54.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xca9be56d
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xebfa8b50
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x45ec1682
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xcb9c200c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 64.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xca9be56d
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x184c0861
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb982c251
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

Warning: Completing partial solution with 125 unfixed non-continuous variables out of 125
User MIP start produced solution with objective 0 (0.02s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xe8371826
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.02s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x82152d16
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5cd61e06
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x8ca9ca41
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xd81c8a2f
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 54.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xf5aca081
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xcb9c200c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 64.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x02f32444
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x9d3165bc
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 69.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xcaa4c03f
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x6042120b
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xfddfd269
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 54.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xf5aca081
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x252343a0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xfaa727ae
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 64.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Generation 3, Best Solution: 0.0
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xca9be56d
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xebfa8b50
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x45ec1682
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xcb9c200c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 64.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xca9be56d
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x184c0861
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb982c251
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

Warning: Completing partial solution with 125 unfixed non-continuous variables out of 125
User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xe8371826
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x82152d16
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xa75d8ae2
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xebfa8b50
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xca9be56d
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x184c0861
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x184c0861
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xe8371826
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x6e2cfbb6
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 54.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xa69de9f9
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xca9be56d
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xcb9c200c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 64.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xcb9c200c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 64.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x184c0861
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x7ae4d351
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 54.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x3b74284d
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 54.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x3b74284d
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 54.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x82152d16
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xca9be56d
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xca9be56d
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5f60157b
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xe8371826
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xebfa8b50
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x0c05ac48
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 54.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xebfa8b50
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb982c251
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

Warning: Completing partial solution with 125 unfixed non-continuous variables out of 125
User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x6cb2b696
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xcb9c200c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 64.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xca9be56d
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x45ec1682
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xf94b783c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xcb9c200c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 64.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x6cb2b696
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xe3159222
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x63a07fb2
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xe200be30
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

Warning: Completing partial solution with 125 unfixed non-continuous variables out of 125
User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xebfa8b50
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Generation 4, Best Solution: 0.0
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x82152d16
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xca9be56d
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xca9be56d
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5f60157b
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xe8371826
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.02s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xebfa8b50
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x0c05ac48
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 54.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xebfa8b50
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb982c251
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

Warning: Completing partial solution with 125 unfixed non-continuous variables out of 125
User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xc0b349a1
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 29.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x0c05ac48
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 54.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5f60157b
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x4a9dc51f
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xcb9c200c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 64.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x2424ee41
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xca9be56d
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5f60157b
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5f60157b
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xca9be56d
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x9c1783f5
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x82152d16
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xe8371826
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xe8371826
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xd81c8a2f
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 54.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5f60157b
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xca9be56d
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xe8371826
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x3b55e47c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x0e4fb2c0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xc7d86d8a
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x91e1d0c3
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xacd23e0f
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb59bf23e
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xd81c8a2f
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 54.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x133fa274
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x82152d16
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5f60157b
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xe8371826
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5f60157b
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xe8371826
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.02s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb982c251
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

Warning: Completing partial solution with 125 unfixed non-continuous variables out of 125
User MIP start produced solution with objective 0 (0.02s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x82152d16
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xcb9c200c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 64.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xca9be56d
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x64479b62
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x82152d16
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x133fa274
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x82152d16
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Generation 5, Best Solution: 0.0
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xca9be56d
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xe8371826
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x3b55e47c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x0e4fb2c0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xc7d86d8a
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x91e1d0c3
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xacd23e0f
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb59bf23e
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xd81c8a2f
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 54.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xf94b783c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x91e1d0c3
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xd81c8a2f
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 54.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x3b55e47c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb59bf23e
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x6cb2b696
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb51a8a13
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xc0b349a1
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 29.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x0ba32090
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xc7d86d8a
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x91e1d0c3
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xc6428e11
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xecb37fa0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xfaa727ae
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 64.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xc0b349a1
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 29.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x3b55e47c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x6b56c8cd
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x3b55e47c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x91e1d0c3
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xacd23e0f
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb59bf23e
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xfaa727ae
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 64.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x0e4fb2c0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x3b55e47c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xabed9ac4
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xecb37fa0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x0e4fb2c0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x3b55e47c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb59bf23e
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.02s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xe8371826
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x924643ce
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x6b56c8cd
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x6b56c8cd
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xc7d86d8a
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Generation 6, Best Solution: 0.0
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x3b55e47c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x6b56c8cd
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x3b55e47c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x91e1d0c3
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xacd23e0f
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb59bf23e
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.02s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xfaa727ae
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 64.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x91e1d0c3
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x72917bb4
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xca9be56d
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x3b55e47c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x3b55e47c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xc6428e11
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xacd23e0f
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xacd23e0f
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5f60157b
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x6b56c8cd
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xfaa727ae
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 64.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x133fa274
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xc0751c87
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

Warning: Completing partial solution with 125 unfixed non-continuous variables out of 125
User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x3b55e47c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x91e1d0c3
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x91e1d0c3
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb59bf23e
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.02s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb59bf23e
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x6b56c8cd
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xabed9ac4
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x91e1d0c3
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.04 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x6b56c8cd
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xc31820a5
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x91e1d0c3
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x3b55e47c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x3d43e93c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xacd23e0f
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xfaa727ae
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 64.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x73a1a5a3
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 74.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x91e1d0c3
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x24b3266e
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

Warning: Completing partial solution with 125 unfixed non-continuous variables out of 125
User MIP start produced solution with objective 0 (0.02s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x3b55e47c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Generation 7, Best Solution: 0.0
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x91e1d0c3
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb59bf23e
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb59bf23e
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.02s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x6b56c8cd
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xabed9ac4
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x91e1d0c3
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x91e1d0c3
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xca9be56d
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x278d2c13
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x4586ca52
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x9d3165bc
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 69.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xca9be56d
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xd7e396c2
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x91e1d0c3
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x6b56c8cd
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x91e1d0c3
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x73a1a5a3
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 74.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x91e1d0c3
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xc7d86d8a
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x4a967f1d
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x91e1d0c3
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x4586ca52
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x6b56c8cd
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb59bf23e
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x6b56c8cd
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x51044c98
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.02s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xc7d86d8a
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xcb9c200c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 64.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xe4bc2b3f
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x91e1d0c3
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb59bf23e
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.02s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xc7d86d8a
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xabed9ac4
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xca9be56d
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xabed9ac4
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x91e1d0c3
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Generation 8, Best Solution: 0.0
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x4586ca52
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x6b56c8cd
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb59bf23e
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x6b56c8cd
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x51044c98
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x6b56c8cd
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x51044c98
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xcb9c200c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 64.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x1d711a66
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x4586ca52
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xc0751c87
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

Warning: Completing partial solution with 125 unfixed non-continuous variables out of 125
User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xcb9c200c
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 64.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xabed9ac4
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x6b56c8cd
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5520b001
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x4586ca52
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x6b56c8cd
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xc0b349a1
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 29.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x6b56c8cd
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x6b56c8cd
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xec9573ff
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x3b74284d
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 54.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb59bf23e
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xca9be56d
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x4586ca52
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x1d711a66
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start produced solution with objective 0 (0.01s)
Loaded user MIP start with objective 0


Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 1 (of 8 available processors)

Solution count 1: 0 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x6b56c8cd
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xca9be56d
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Generation 9, Best Solution: 0.0
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x4586ca52
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x6b56c8cd
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xc0b349a1
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 29.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x0d0f9c8a
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x6b56c8cd
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x0d0f9c8a
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x4586ca52
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x6cb2b696
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xabed9ac4
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5f60157b
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb9f642de
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xabed9ac4
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x0d0f9c8a
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xec9573ff
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xabed9ac4
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x0d0f9c8a
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x971a594e
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 29.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb9f642de
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x278d2c13
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb9f642de
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x4586ca52
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xec9573ff
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xabed9ac4
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xabed9ac4
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x62793844
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 54.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x0d0f9c8a
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xec9573ff
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xabed9ac4
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.04 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xc0b349a1
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 29.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x0d0f9c8a
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xabed9ac4
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x77b3820a
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 29.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x4586ca52
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Generation 11, Best Solution: 0.0
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.01 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb9f642de
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x278d2c13
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb9f642de
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x4586ca52
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xec9573ff
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xabed9ac4
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x278d2c13
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x62793844
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 54.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xabed9ac4
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb9f642de
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb9f642de
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xec9573ff
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb9f642de
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x4586ca52
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x62793844
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 54.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xc31820a5
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x4586ca52
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xabed9ac4
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xec0e767d
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x278d2c13
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xec9573ff
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xec9573ff
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xb9f642de
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 39.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x876fb1b8
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xec9573ff
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.03 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x77b3820a
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 29.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.04 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.04 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xabed9ac4
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xabed9ac4
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x278d2c13
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xabed9ac4
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x6f2f12fb
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 59.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x4586ca52
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x278d2c13
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x5e5426b0
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 49.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0x278d2c13
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 44.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Discarded solution information
Gurobi Optimizer version 10.0.3 build v10.0.3rc0 (win64)

CPU model: Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz, instruction set [SSE2|AVX|AVX2]
Thread count: 4 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 35 rows, 125 columns and 375 nonzeros
Model fingerprint: 0xabed9ac4
Variable types: 0 continuous, 125 integer (125 binary)
Coefficient statistics:
  Matrix range     [1e+00, 4e+00]
  Objective range  [5e+02, 6e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 1e+03]

User MIP start did not produce a new incumbent solution
User MIP start violates constraint ProductSource_0 by 34.000000000

Found heuristic solution: objective 600.0000000
Presolve removed 25 rows and 100 columns
Presolve time: 0.00s
Presolved: 10 rows, 25 columns, 50 nonzeros
Found heuristic solution: objective 0.0000000
Variable types: 0 continuous, 25 integer (25 binary)

Explored 0 nodes (0 simplex iterations) in 0.02 seconds (0.00 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 0 600 

Optimal solution found (tolerance 1.00e-04)
Best objective 0.000000000000e+00, best bound 0.000000000000e+00, gap 0.0000%
Generation 12, Best Solution: 0.0
Convergence reached. Exiting the loop.
Best Solution: [1 4 2 0 3]
]
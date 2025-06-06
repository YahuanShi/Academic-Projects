import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# Define problem data
q_start = np.array([0.5, 2.5])  # Start position
q_goal = np.array([9.5, 0.5])  # Goal position

obstacles = [
    np.array([[1, 3], [4, 3], [1, 5], [4, 5]]),
    np.array([[0, 0], [4, 0], [0, 2], [4, 2]]),
    np.array([[6, 1], [7, 1], [6, 5], [7, 5]])
]  # Obstacle positions

# Define decision variables
control_points = cp.Variable((12, 2))  # Control points of the Bézier curves

# Define constraints
constraints = [
    control_points[0] == q_start,  # Start position constraint
    control_points[-1] == q_goal,  # Goal position constraint
    control_points[3] == control_points[4],  # Continuity constraints
    control_points[7] == control_points[8],
    control_points[3] - control_points[2] == control_points[5] - control_points[4],
    control_points[7] - control_points[6] == control_points[9] - control_points[8],
    control_points[1] - 2 * control_points[2] + control_points[3] == control_points[4] - 2 * control_points[5] + control_points[6],
    control_points[5] - 2 * control_points[6] + control_points[7] == control_points[8] - 2 * control_points[9] + control_points[10]
]

# Range constraints based on segments
segments = [
    (0, 4, (0, 6, 2, 3)),
    (4, 8, (4, 6, 0, 5)),
    (8, 12, (4, 10, 0, 1))
]

for start, end, (x_min, x_max, y_min, y_max) in segments:
    for i in range(start, end):
        constraints.append(control_points[i, 0] >= x_min)
        constraints.append(control_points[i, 0] <= x_max)
        constraints.append(control_points[i, 1] >= y_min)
        constraints.append(control_points[i, 1] <= y_max)

# Define objective function (minimize curve length)
objective = cp.Minimize(cp.sum([cp.norm(control_points[i + 1] - control_points[i]) for i in range(11)]))

# Formulate and solve the optimization problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Extract optimal control points
control_points_opt = control_points.value

print("Problem status:", problem.status)
print("Optimal value:", problem.value)

# Plot the Bézier curve and control points
t = np.linspace(0, 1, 100)
curve = []

for i in range(3):
    p0, p1, p2, p3 = control_points_opt[i * 4:(i + 1) * 4, :]
    segment = ((1 - t) ** 3 * p0[:, np.newaxis] +
               3 * t * (1 - t) ** 2 * p1[:, np.newaxis] +
               3 * t ** 2 * (1 - t) * p2[:, np.newaxis] +
               t ** 3 * p3[:, np.newaxis])
    curve.append(segment)

plt.figure(figsize=(10, 5))
plt.plot(control_points_opt[:, 0], control_points_opt[:, 1], color='turquoise', marker='o', alpha=0.9, label='Control Points', linewidth=1.5)
plt.plot(q_start[0], q_start[1], color='red', marker='s', label='Start', markersize=10)
plt.plot(q_goal[0], q_goal[1], color='palevioletred', marker='D', label='Goal', markersize=10)

for obstacle in obstacles:
    plt.plot(obstacle[[0, 1, 3, 2, 0], 0], obstacle[[0, 1, 3, 2, 0], 1], 'gray')
    plt.fill(obstacle[:, 0], obstacle[:, 1], 'gray', alpha=0.5)

colors = ['darkorange', 'blueviolet', 'crimson']
for i, c in enumerate(curve, 1):
    plt.plot(c[0], c[1], colors[i - 1], alpha=0.9, linewidth=1.5, label=f'Curve {i}')

plt.axis([0, 10, 0, 5])
plt.grid(linewidth=0.5, linestyle='dashed', alpha=0.5)
plt.xticks(np.arange(0, 11, 1))
plt.yticks(np.arange(0, 6, 1))
plt.title('Bézier Curve Toy Example')
plt.xlabel('x', size=15)
plt.ylabel('y', size=15)
plt.legend(loc='upper right')
plt.savefig('opt_toy.pdf')
plt.show()

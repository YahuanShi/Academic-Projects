import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import math

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('env', help='input YAML file with environment')
    parser.add_argument('pdf_name', help='output PDF file name')
    parser.add_argument('--export-car', dest='export_car_plan', metavar='car_plan_file', help='Enable export car case')
    parser.add_argument('--export-arm', dest='export_arm_plan', metavar='arm_plan_file', help='Enable export arm case')
    return parser.parse_args()

def load_environment(file_path):
    with open(file_path, "r") as stream:
        return yaml.safe_load(stream)

def get_obstacles(environment):
    obstacles = []
    if environment["obstacles"]:
        for o in environment["obstacles"]:
            p = o["pos"]
            l, w, h = o["size"]
            corner0 = np.array([p[0], p[1]]) + np.array([-l/2, -w/2])  
            corner1 = np.array([p[0], p[1]]) + np.array([l/2, -w/2])  
            corner2 = np.array([p[0], p[1]]) + np.array([l/2, w/2])  
            corner3 = np.array([p[0], p[1]]) + np.array([-l/2, w/2])  
            Box = [corner0, corner1, corner2, corner3]
            obstacles.append(np.array(Box))
    return obstacles

def compute_hyperplane(segment, obstacle, color, line_type):
    w = cp.Variable((2, 1))
    b = cp.Variable()
    X = np.concatenate((obstacle, segment), axis=0)

    constraints = [w.T @ endpoint - b >= 1 for endpoint in segment] + [w.T @ corner - b <= -1 for corner in obstacle]
    objective = cp.Minimize(cp.sum_squares(w))
    problem = cp.Problem(objective, constraints)
    problem.solve()

    hyperplane = np.vstack((w.value, b.value))
    slope = -hyperplane[0] / hyperplane[1]
    intercept = hyperplane[2] / hyperplane[1]
    x_vals = np.array([min(X[:, 0]) - 0.5, max(X[:, 0]) + 0.5])
    y_vals = slope * x_vals + intercept
    plt.plot(x_vals, y_vals, color=color, linewidth=1.5, alpha=0.5, linestyle=line_type)

    return hyperplane

def draw_curves(segments, arguments, start, goal, x_min, x_max, y_min, y_max):
    control_points = cp.Variable((4 * len(segments), 2))
    constraints = [control_points[0] == start, control_points[-1] == goal]

    for i in range(len(segments) - 1):
        constraints += [
            control_points[4 * i + 3] == control_points[4 * i + 4],
            control_points[4 * i + 3] - control_points[4 * i + 2] == control_points[4 * i + 5] - control_points[4 * i + 4],
            control_points[4 * i + 1] - 2 * control_points[4 * i + 2] + control_points[4 * i + 3] == control_points[4 * i + 4] - 2 * control_points[4 * i + 5] + control_points[4 * i + 6]
        ]

    for i in range(len(segments)):
        hyperplanes = arguments[i]
        for hyperplane in hyperplanes:
            m, n, b = hyperplane
            constraints += [
                m * control_points[4 * i + j, 0] + n * control_points[4 * i + j, 1] - b >= 0 for j in range(4)
            ]

    for i in range(4 * len(segments)):
        constraints += [
            control_points[i, 0] >= x_min, control_points[i, 0] <= x_max,
            control_points[i, 1] >= y_min, control_points[i, 1] <= y_max
        ]

    lengths = [cp.norm(control_points[i * 4 + j + 1] - control_points[i * 4 + j]) for i in range(len(segments)) for j in range(3)]
    objective = cp.sum(lengths)

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve()

    control_points_opt = control_points.value
    print("Problem status:", problem.status)
    print("Optimal value:", problem.value)
    print(control_points_opt)
    
    plt.plot(control_points_opt[:, 0], control_points_opt[:, 1], color='lightgrey', marker='o', alpha=0.9, label='Control Points', linewidth=1.5)
    t = np.linspace(0, 1, 100)
    curve_colors = ['dodgerblue', 'yellow', 'lime']
    
    for i, segment in enumerate(segments):
        p0, p1, p2, p3 = control_points_opt[i * 4:(i + 1) * 4, :]
        curve = (1 - t) ** 3 * p0[:, np.newaxis] + 3 * t * (1 - t) ** 2 * p1[:, np.newaxis] + 3 * t ** 2 * (1 - t) * p2[:, np.newaxis] + t ** 3 * p3[:, np.newaxis]
        plt.plot(curve[0], curve[1], curve_colors[i % 3], alpha=0.9, linewidth=1.5)

    return control_points_opt

def compute_safe_regions(segments, obstacles, x_min, x_max, y_min, y_max, pdf_name, start, goal):
    plt.figure(figsize=(10, 5))
    plt.xlim(-2, 10)
    plt.ylim(-2, 5)

    colors = ["orange", "limegreen", "crimson", "blueviolet", "khaki"]
    map_colors = ["summer", "GnBu", "coolwarm", "cool", "Pastel1"]
    line_types = ['dotted', 'dashed', (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5, 1, 5)), (0, (3, 10, 1, 10))]
    arguments = []

    for i, segment in enumerate(segments):
        color = colors[i % len(colors)]
        map_color = map_colors[i % len(map_colors)]
        line_type = line_types[i % len(line_types)]
        hyperplanes = [compute_hyperplane(segment, obstacle, color, line_type) for obstacle in obstacles]
        arguments.append(hyperplanes)

        x = np.arange(-4, 10.02, 0.02)
        y = np.arange(-4, 5.02, 0.02)
        X, Y = np.meshgrid(x, y)

        zs = [m * X + n * Y - b for m, n, b in hyperplanes]
        valid_points = np.logical_and.reduce([z > 0 for z in zs])
        fill_array = np.full_like(X, np.nan)
        fill_array[valid_points] = 1
        plt.imshow(fill_array, extent=[-4, 10, -4, 5], origin='lower', cmap=map_color, vmin=0, vmax=1, alpha=0.6)

    control_points_opt = draw_curves(segments, arguments, start, goal, x_min, x_max, y_min, y_max)

    for obstacle in obstacles:
        plt.plot([obstacle[0, 0], obstacle[1, 0]], [obstacle[0, 1], obstacle[1, 1]], 'gray')
        plt.plot([obstacle[1, 0], obstacle[2, 0]], [obstacle[1, 1], obstacle[2, 1]], 'gray')
        plt.plot([obstacle[2, 0], obstacle[3, 0]], [obstacle[2, 1], obstacle[3, 1]], 'gray')
        plt.plot([obstacle[3, 0], obstacle[0, 0]], [obstacle[3, 1], obstacle[0, 1]], 'gray')
        plt.fill([obstacle[0, 0], obstacle[1, 0], obstacle[2, 0], obstacle[3, 0]], [obstacle[0, 1], obstacle[1, 1], obstacle[2, 1], obstacle[3, 1]], 'dimgray', alpha=0.5)

    plt.grid(linewidth=0.5, linestyle='dashed', alpha=0.5)
    plt.xticks(np.arange(-2, 10, 1))
    plt.yticks(np.arange(-2, 5, 1))
    plt.title('Computing Safe Regions')
    plt.xlabel('x', size=15)
    plt.ylabel('y', size=15)
    plt.legend(loc='upper right')
    plt.savefig(pdf_name)
    plt.show()

    return control_points_opt

def car_case(segments, control_points_opt, export_car_plan):
    state_x, state_y, thetas, diff_xs, diff_ys, diff_2_xs, diff_2_ys, s, phis = [], [], [], [], [], [], [], [], []

    for i in range(len(segments)):
        T = 500
        P0, P1, P2, P3 = control_points_opt[i * 4:(i + 1) * 4, :]
        x0, y0 = P0.tolist()
        x1, y1 = P1.tolist()
        x2, y2 = P2.tolist()
        x3, y3 = P3.tolist()
        for t in range(0, T):
            x = (3 * x1 - x0 - 3 * x2 + x3) * t**3 / T**3 + (3 * x0 - 6 * x1 + 3 * x2) * t**2 / T**2 + (3 * x1 - 3 * x0) * t / T + x0
            y = (3 * y1 - y0 - 3 * y2 + y3) * t**3 / T**3 + (3 * y0 - 6 * y1 + 3 * y2) * t**2 / T**2 + (3 * y1 - 3 * y0) * t / T + y0
            state_x.append(x)
            state_y.append(y)
        if i == len(segments) - 1:
            t = T
            x = (3 * x1 - x0 - 3 * x2 + x3) * t**3 / T**3 + (3 * x0 - 6 * x1 + 3 * x2) * t**2 / T**2 + (3 * x1 - 3 * x0) * t / T + x0
            y = (3 * y1 - y0 - 3 * y2 + y3) * t**3 / T**3 + (3 * y0 - 6 * y1 + 3 * y2) * t**2 / T**2 + (3 * y1 - 3 * y0) * t / T + y0
            state_x.append(x)
            state_y.append(y)

        for t in range(0, T):
            diff_x = (3 * x1 - 3 * x0) / T + 2 * t * (3 * x0 - 6 * x1 + 3 * x2) / T**2 - 3 * t**2 * (x0 - 3 * x1 + 3 * x2 - x3) / T**3
            diff_y = (3 * y1 - 3 * y0) / T + 2 * t * (3 * y0 - 6 * y1 + 3 * y2) / T**2 - 3 * t**2 * (y0 - 3 * y1 + 3 * y2 - y3) / T**3
            diff_xs.append(diff_x)
            diff_ys.append(diff_y)
        if i == len(segments) - 1:
            t = T
            diff_x = (3 * x1 - 3 * x0) / T + 2 * t * (3 * x0 - 6 * x1 + 3 * x2) / T**2 - 3 * t**2 * (x0 - 3 * x1 + 3 * x2 - x3) / T**3
            diff_y = (3 * y1 - 3 * y0) / T + 2 * t * (3 * y0 - 6 * y1 + 3 * y2) / T**2 - 3 * t**2 * (y0 - 3 * y1 + 3 * y2 - y3) / T**3
            diff_xs.append(diff_x)
            diff_ys.append(diff_y)

        for t in range(0, T):
            diff_2_x = (6 * x0 - 12 * x1 + 6 * x2) / T**2 - 6 * t * (x0 - 3 * x1 + 3 * x2 - x3) / T**3
            diff_2_y = (6 * y0 - 12 * y1 + 6 * y2) / T**2 - 6 * t * (y0 - 3 * y1 + 3 * y2 - y3) / T**3
            diff_2_xs.append(diff_2_x)
            diff_2_ys.append(diff_2_y)
        if i == len(segments) - 1:
            t = T
            diff_2_x = (6 * x0 - 12 * x1 + 6 * x2) / T**2 - 6 * t * (x0 - 3 * x1 + 3 * x2 - x3) / T**3
            diff_2_y = (6 * y0 - 12 * y1 + 6 * y2) / T**2 - 6 * t * (y0 - 3 * y1 + 3 * y2 - y3) / T**3
            diff_2_xs.append(diff_2_x)
            diff_2_ys.append(diff_2_y)

    for i in range(len(diff_xs)):
        theta = math.atan2(diff_ys[i], diff_xs[i])
        thetas.append(theta)

    states = [[state_x[i], state_y[i], thetas[i]] for i in range(len(state_x))]
    s = [(diff_xs[i]**2 + diff_ys[i]**2)**0.5 for i in range(len(diff_xs))]
    phis = [math.atan(3 * (diff_xs[i] * diff_2_ys[i] - diff_ys[i] * diff_2_xs[i]) / (diff_xs[i]**2 + diff_ys[i]**2)**1.5) for i in range(len(diff_xs) - 1)]
    u = [[s[i], phis[i]] for i in range(len(diff_xs) - 1)]

    result_car = {
        'plan': {
            'type': 'car',
            'dt': 1,
            'L': 3,
            'W': 1.5,
            'H': 1,
            'states': states,
            'actions': u
        }
    }

    with open(export_car_plan, "w") as stream:
        yaml.dump(result_car, stream, default_flow_style=None, sort_keys=False)

    plt.figure(figsize=(10, 5))
    plt.xlim(0, 10)
    plt.ylim(0, 5)
    plt.plot(state_x, state_y, 'yellow', alpha=0.9, linewidth=1.5)
    plt.show()

def arm_case(segments, control_points_opt, export_arm_plan):
    state_x, state_y, alphas, thetas = [], [], [], []
    err = False

    def computing_states(x, y, phi):
        nonlocal err
        x2 = x - np.cos(phi)
        y2 = y - np.sin(phi)
        c2 = (x2**2 + y2**2 - 2) / 2
        cb = (x2**2 + y2**2)**0.5 / 2
        theta2 = np.arccos(c2)
        if 0 <= cb <= 1:
            beta = np.arccos(cb)
            theta1 = np.arctan2(y2, x2) - beta
            theta3 = phi - theta1 - theta2
            thetas.append([theta1.tolist(), theta2.tolist(), theta3.tolist()])
        else:
            err = True

    for i in range(len(segments)):
        T = 500
        P0, P1, P2, P3 = control_points_opt[i * 4:(i + 1) * 4, :]
        x0, y0 = P0.tolist()
        x1, y1 = P1.tolist()
        x2, y2 = P2.tolist()
        x3, y3 = P3.tolist()
        for t in range(0, T):
            x = (3 * x1 - x0 - 3 * x2 + x3) * t**3 / T**3 + (3 * x0 - 6 * x1 + 3 * x2) * t**2 / T**2 + (3 * x1 - 3 * x0) * t / T + x0
            y = (3 * y1 - y0 - 3 * y2 + y3) * t**3 / T**3 + (3 * y0 - 6 * y1 + 3 * y2) * t**2 / T**2 + (3 * y1 - 3 * y0) * t / T + y0
            state_x.append(x)
            state_y.append(y)
        if i == len(segments) - 1:
            t = T
            x = (3 * x1 - x0 - 3 * x2 + x3) * t**3 / T**3 + (3 * x0 - 6 * x1 + 3 * x2) * t**2 / T**2 + (3 * x1 - 3 * x0) * t / T + x0
            y = (3 * y1 - y0 - 3 * y2 + y3) * t**3 / T**3 + (3 * y0 - 6 * y1 + 3 * y2) * t**2 / T**2 + (3 * y1 - 3 * y0) * t / T + y0
            state_x.append(x)
            state_y.append(y)

        for t in range(0, T):
            diff_x = (3 * x1 - 3 * x0) / T + 2 * t * (3 * x0 - 6 * x1 + 3 * x2) / T**2 - 3 * t**2 * (x0 - 3 * x1 + 3 * x2 - x3) / T**3
            diff_y = (3 * y1 - 3 * y0) / T + 2 * t * (3 * y0 - 6 * y1 + 3 * y2) / T**2 - 3 * t**2 * (y0 - 3 * y1 + 3 * y2 - y3) / T**3
            alphas.append(math.atan2(diff_y, diff_x))
        if i == len(segments) - 1:
            t = T
            diff_x = (3 * x1 - 3 * x0) / T + 2 * t * (3 * x0 - 6 * x1 + 3 * x2) / T**2 - 3 * t**2 * (x0 - 3 * x1 + 3 * x2 - x3) / T**3
            diff_y = (3 * y1 - 3 * y0) / T + 2 * t * (3 * y0 - 6 * y1 + 3 * y2) / T**2 - 3 * t**2 * (y0 - 3 * y1 + 3 * y2 - y3) / T**3
            alphas.append(math.atan2(diff_y, diff_x))

    states = [[state_x[i], state_y[i], alphas[i]] for i in range(len(state_x))]

    result_arm = {
        'plan': {
            'type': 'arm',
            'L': [1, 1, 1],
            'states': thetas,
        }
    }

    with open(export_arm_plan, "w") as stream:
        yaml.dump(result_arm, stream, default_flow_style=None, sort_keys=False)

def main():
    args = parse_arguments()
    env = load_environment(args.env)
    
    x_min, y_min, z_min = env["environment"]["min"]
    x_max, y_max, z_max = env["environment"]["max"]
    
    obstacles = get_obstacles(env["environment"])
    solutionpath = env["motionplanning"]["solutionpath"]
    start = np.array(env["motionplanning"]["start"][:2])
    goal = np.array(env["motionplanning"]["goal"][:2])
    segments = [np.array([solutionpath[i][:2], solutionpath[i + 1][:2]]) for i in range(len(solutionpath) - 1)]

    control_points_opt = compute_safe_regions(segments, obstacles, x_min, x_max, y_min, y_max, args.pdf_name, start, goal)

    if args.export_car_plan:
        car_case(segments, control_points_opt, args.export_car_plan)
    
    if args.export_arm_plan:
        arm_case(segments, control_points_opt, args.export_arm_plan)

if __name__ == '__main__':
    main()

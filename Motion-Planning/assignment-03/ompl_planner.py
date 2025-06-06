import argparse
import numpy as np
import yaml
import time
import fcl
import math
import sys
import ompl.base as base
import ompl.geometric as geometric
import ompl.control as control

parser = argparse.ArgumentParser()
parser.add_argument('action', help='input YAML file with action')
parser.add_argument('output_path', help='output YAML file with solution')
parser.add_argument('--export-planner-data', help='Export planner data')
args = parser.parse_args()

with open(args.action, 'r') as steam:
    object = yaml.safe_load(steam)

if args.export_planner_data is not None:
    out_path = args.export_planner_data

timelimit = object["hyperparameters"]["timelimit"]
goal_bias = object["hyperparameters"]["goal_bias"]
goal_eps = object["hyperparameters"]["goal_eps"]
obs_objs = []
env_min = object["environment"]["min"]
env_max = object["environment"]["max"]

if object["motionplanning"]["type"] == 'car':
    dt = object["motionplanning"]["dt"]


def creat_Obs():
    for k, o in enumerate(object["environment"]["obstacles"]):
        if o["type"] == "box":
            p = o["pos"]
            s = o["size"]
            L = s[0]
            W = s[1]
            H = s[2]
            b = fcl.Box(L, W, H)
            T = np.array(p)
            t = fcl.Transform(T)
            obs_objs.append(fcl.CollisionObject(b, t))

        elif o["type"] == "cylinder":
            p = o["pos"]
            q = o["q"]
            r = o["r"]
            lz = o["lz"]
            b = fcl.Cylinder(r, lz)
            T = np.array(p)
            t = fcl.Transform(q, T)
            obs_objs.append(fcl.CollisionObject(b, t))


def getCost(si):
    return base.PathLengthOptimizationObjective(si)


def statePropagator(start, control, duration, target):
    u1 = control[0]
    u2 = control[1]
    x = start.getX()
    y = start.getY()
    theta = start.getYaw()
    L = object["motionplanning"]["L"]
    xn = u1 * np.cos(theta)
    yn = u1 * np.sin(theta)
    thetan = u1 * np.tan(u2) / L
    target.setX(x + duration * xn)
    target.setY(y + duration * yn)
    target.setYaw(theta + duration * thetan)
    return target


def Broadphase(objs, obs_objs):
    manager1 = fcl.DynamicAABBTreeCollisionManager()
    manager2 = fcl.DynamicAABBTreeCollisionManager()

    manager1.registerObjects(objs)
    manager2.registerObjects(obs_objs)

    manager1.setup()
    manager2.setup()

    cdata = fcl.CollisionData()
    manager1.collide(cdata, fcl.defaultCollisionCallback)

    req = fcl.CollisionRequest(num_max_contacts=100, enable_contact=True)
    rdata = fcl.CollisionData(request=req)
    rdata = fcl.CollisionData(request=req)
    manager1.collide(manager2, rdata, fcl.defaultCollisionCallback)

    return rdata.result.is_collision


def create_car(x, y, theta):
    L = object["motionplanning"]["L"]
    W = object["motionplanning"]["W"]
    H = object["motionplanning"]["H"]
    b = fcl.Box(L, W, H)
    T = np.array([x, y, 0])
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    t = fcl.Transform(R, T)
    car_model = fcl.CollisionObject(b, t)
    return [car_model]


def car_check(x, y, theta):
    objs = create_car(x, y, theta)
    return Broadphase(objs, obs_objs)


def Rmatrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return matrix


def get_Transform(theta1, theta2, theta3):
    z = np.array([1, 0, 0])
    R1 = Rmatrix(theta1)
    R2 = Rmatrix(theta1 + theta2)
    R3 = Rmatrix(theta1 + theta2 + theta3)

    t1 = np.matmul(R1, z) / 2
    t2 = t1 * 2 + np.matmul(R2, z) / 2
    t3 = t1 * 2 + np.matmul(R2, z) + np.matmul(R3, z) / 2

    T1 = fcl.Transform(R1, t1)
    T2 = fcl.Transform(R2, t2)
    T3 = fcl.Transform(R3, t3)

    return T1, T2, T3


def create_Links(theta1, theta2, theta3):
    L1, L2, L3 = object["motionplanning"]["L"]
    l1 = fcl.Box(L1, 0.1, 0.1)
    l2 = fcl.Box(L2, 0.1, 0.1)
    l3 = fcl.Box(L3, 0.1, 0.1)

    T1, T2, T3 = get_Transform(theta1, theta2, theta3)

    Link1 = fcl.CollisionObject(l1, T1)
    Link2 = fcl.CollisionObject(l2, T2)
    Link3 = fcl.CollisionObject(l3, T3)

    return [Link1, Link2, Link3]


def arm_check(theta1, theta2, theta3):
    objs = create_Links(theta1, theta2, theta3)
    return Broadphase(objs, obs_objs)


def isValid(state):
    theta1 = state[0].value
    theta2 = state[1].value
    theta3 = state[2].value
    if arm_check(theta1, theta2, theta3):
        return False
    else:
        return True


def isValid_car(state):
    x = state.getX()
    y = state.getY()
    theta = state.getYaw()
    if car_check(x, y, theta):
        return False
    else:
        return True


def getStateSampler(si):
    return base.ObstacleBasedValidStateSampler(si)


def main():
    creat_Obs()
    si = None  # Initialize si to None

    if object["motionplanning"]["type"] == 'arm':
        spaces = []
        for _ in range(3):
            sp = base.SO2StateSpace()
            spaces.append(sp)
        space = base.CompoundStateSpace()
        for sp in spaces:
            space.addSubspace(sp, 1.0)

        si = base.SpaceInformation(space)
        si.setStateValidityChecker(base.StateValidityCheckerFn(isValid))
        si.setValidStateSamplerAllocator(base.ValidStateSamplerAllocator(getStateSampler))

        print(si.settings())

    elif object["motionplanning"]["type"] == 'car':  # Use elif here
        space = base.SE2StateSpace()
        bounds = base.RealVectorBounds(2)
        bounds.setLow(0, env_min[0])
        bounds.setHigh(0, env_max[0])
        bounds.setLow(1, env_min[1])
        bounds.setHigh(1, env_max[1])
        space.setBounds(bounds)

        control_space = control.RealVectorControlSpace(space, 2)
        control_bounds = base.RealVectorBounds(2)
        control_bounds.setLow(0, -0.5)
        control_bounds.setHigh(0, +2)
        control_bounds.setLow(1, -np.pi / 6)
        control_bounds.setHigh(1, +np.pi / 6)

        control_space.setBounds(control_bounds)

        si = control.SpaceInformation(space, control_space)
        si.setStateValidityChecker(base.StateValidityCheckerFn(isValid_car))
        si.setStatePropagator(control.StatePropagatorFn(statePropagator))
        si.setPropagationStepSize(0.1)
        si.setMinMaxControlDuration(1, 2)

    if si is None:
        raise ValueError("SpaceInformation (si) is not initialized.")

    pdef = base.ProblemDefinition(si)

    start = base.State(space)
    start[0] = object["motionplanning"]["start"][0]
    start[1] = object["motionplanning"]["start"][1]
    start[2] = object["motionplanning"]["start"][2]

    goal = base.State(space)
    goal[0] = object["motionplanning"]["goal"][0]
    goal[1] = object["motionplanning"]["goal"][1]
    goal[2] = object["motionplanning"]["goal"][2]

    pdef.setStartAndGoalStates(start, goal, goal_eps)

    if object["motionplanning"]["type"] == 'car':
        pdef.setOptimizationObjective(getCost(si))

    if object["motionplanning"]["type"] == 'arm':
        time_begin = time.time()
        time_now = time.time()
        path = None
        length = 0
        Approximate = False
        while time_now - time_begin <= timelimit:
            planner = geometric.RRT(si)
            planner.setProblemDefinition(pdef)
            planner.setGoalBias(goal_bias)
            step_size = 0.03
            planner.setRange(step_size)

            planner.solve(10)
            tentative_path = pdef.getSolutionPath()
            tentative_path_length = tentative_path.length()

            if (not pdef.hasApproximateSolution()) and (path is None or tentative_path_length < length):
                path = tentative_path
                length = tentative_path_length

            time_now = time.time()

        if path is None and pdef.hasApproximateSolution():
            path = tentative_path
            Approximate = True

    if object["motionplanning"]["type"] == 'car':
        time_begin = time.time()
        time_now = time.time()
        path = None
        length = 0
        Approximate = False
        Durations = []
        steps = []
        while time_now - time_begin <= timelimit:
            planner = control.SST(si)
            planner.setProblemDefinition(pdef)
            planner.setGoalBias(goal_bias)
            planner.setup()
            planner.solve(timelimit)
            planner_data = base.PlannerData(si)
            tentative_path = pdef.getSolutionPath()
            tentative_path_length = tentative_path.length()

            if (not pdef.hasApproximateSolution()) and (path is None or tentative_path_length < length):
                path = tentative_path
                length = tentative_path_length
                planner.getPlannerData(planner_data)

                num_controls = path.getControlCount()

                s = list(map(lambda Controls: Controls[0], path.getControls()))
                phi = list(map(lambda Controls: Controls[1], path.getControls()))
                for i in range(num_controls):
                    duration = path.getControlDuration(i)
                    Durations.append(duration)
                    step = int(duration * 10)
                    steps.append(step)

            time_now = time.time()
        if path is None and pdef.hasApproximateSolution():
            path = tentative_path
            Approximate = True
            planner.getPlannerData(planner_data)

            num_controls = path.getControlCount()

            s = list(map(lambda Controls: Controls[0], path.getControls()))
            phi = list(map(lambda Controls: Controls[1], path.getControls()))
            for i in range(num_controls):
                duration = path.getControlDuration(i)
                Durations.append(duration)
                step = int(duration * 10)
                steps.append(step)

    if args.export_planner_data is not None:
        graphml_content = planner_data.printGraphML()
        with open(args.export_planner_data, 'w') as f:
            f.write(graphml_content)

    if object["motionplanning"]["type"] == 'arm':
        theta1 = list(map(lambda state: state[0].value, path.getStates()))
        theta2 = list(map(lambda state: state[1].value, path.getStates()))
        theta3 = list(map(lambda state: state[2].value, path.getStates()))

    if object["motionplanning"]["type"] == 'car':
        x = list(map(lambda state: state.getX(), path.getStates()))
        y = list(map(lambda state: state.getY(), path.getStates()))
        theta = list(map(lambda state: state.getYaw(), path.getStates()))

    with open(args.output_path, 'w') as f:
        f.write("plan:" + '\n')
        if object["motionplanning"]["type"] == 'arm':
            f.write("  type: arm" + '\n')
            f.write("  L: " + str(object["motionplanning"]["L"]) + '\n')
        if object["motionplanning"]["type"] == 'car':
            f.write("  type: car" + '\n')
            f.write("  dt: " + str(object["motionplanning"]["dt"]) + '\n')
            f.write("  L: " + str(object["motionplanning"]["L"]) + '\n')
            f.write("  W: " + str(object["motionplanning"]["W"]) + '\n')
            f.write("  H: " + str(object["motionplanning"]["H"]) + '\n')
        f.write("  states:" + '\n')
        if object["motionplanning"]["type"] == 'arm':
            for i in range(len(theta1)):
                f.write("    - " + str([theta1[i], theta2[i], theta3[i]]) + '\n')

        if object["motionplanning"]["type"] == 'car':
            for i in range(len(x)):
                f.write("    - " + str([x[i], y[i], theta[i]]) + '\n')
            f.write("  actions: " + '\n')
            for i in range(len(steps)):
                num = steps[i]
                for k in range(num):
                    f.write("    -" + ' ' + str([s[i], phi[i]]) + '\n')


if __name__ == "__main__":
    main()

import sys
import yaml
from ompl import base as ob
from ompl import control as oc
from ompl import geometric as og
from ompl import tools as ot
import math
import numpy as np
import meshcat.transformations as mtf
import fcl
from fcl import DynamicAABBTreeCollisionManager
import copy
import subprocess

sys.setrecursionlimit(10**6)

# Optimization objective for the path
def getCost(si):
    return ob.PathLengthOptimizationObjective(si)

def is_state_valid(state, env):

    space = ob.SE2StateSpace()
    bounds = ob.RealVectorBounds(2)
    env_min = env["environment"]["min"]
    env_max = env["environment"]["max"]
    bounds.setLow(0, env_min[0])
    bounds.setLow(1, env_min[1])
    bounds.setHigh(0, env_max[0])
    bounds.setHigh(1, env_max[1])
    space.setBounds(bounds)
    uspace = oc.RealVectorControlSpace(space, 2)
    ubounds = ob.RealVectorBounds(2)
    ubounds.setLow(0, -0.5)
    ubounds.setHigh(0, 2)
    ubounds.setLow(1, -math.pi / 6)
    ubounds.setHigh(1, math.pi / 6)
    uspace.setBounds(ubounds)

    si = oc.SpaceInformation(space, uspace)

    if not si.satisfiesBounds(state):
        return False

    x = state.getX()
    y = state.getY()
    theta = state.getYaw()
    collision = check_collision_car(x, y, theta, env)
    return not collision


def check_collision_car(x, y, theta, env):
    car_size = [env['motionplanning']['L'], env['motionplanning']['W'], env['motionplanning']['H']]
    car_pos = [x, y, env['motionplanning']['H'] / 2]
    car = create_car(car_pos, car_size, theta)

    obstacles = create_obstacles(env)
    obstacle_manager = DynamicAABBTreeCollisionManager()
    obstacle_manager.registerObjects(obstacles)
    obstacle_manager.setup()

    car_manager = DynamicAABBTreeCollisionManager()
    car_manager.registerObject(car)
    car_manager.setup()

    req = fcl.CollisionRequest(enable_contact=True)
    rdata = fcl.CollisionData(request=req)
    car_manager.collide(obstacle_manager, rdata, fcl.defaultCollisionCallback)
    collision = copy.copy(rdata.result.is_collision)
    return collision

def create_car(pos, size, theta):
    box = fcl.Box(size[0], size[1], size[2])
    rotation_matrix = mtf.rotation_matrix(theta, [0, 0, 1])[:3, :3]
    transform = fcl.Transform(rotation_matrix, np.array(pos))
    return fcl.CollisionObject(box, transform)

def create_obstacles(env):
    obstacles = []
    for obs in env['environment']['obstacles']:
        if obs['type'] == 'box':
            size = obs['size']
            transform = fcl.Transform(np.array(obs["pos"]))
            obstacles.append(fcl.CollisionObject(fcl.Box(size[0], size[1], size[2]), transform))
        elif obs['type'] == 'cylinder':
            p = obs["pos"]
            q = obs["q"]
            r = obs["r"]
            lz = obs["lz"]
            cylinder = fcl.Cylinder(r, lz)
            transform = fcl.Transform(q, p)
            obstacles.append(fcl.CollisionObject(cylinder, transform))
        else:
            raise ValueError("Unknown obstacle type")
    return obstacles

def load_yaml(filename):
    with open(filename, 'r') as f:
        return yaml.safe_load(f)

def configure_planner(config, si):
    planner_type = config['hyperparameters'].get('planner', 'rrt')
    if planner_type == "rrt":
        planner = oc.RRT(si) if config['motionplanning']['type'] == 'car' else og.RRT(si)
    elif planner_type == "rrt*":
        planner = og.RRTstar(si)
    elif planner_type == "rrt-connect":
        planner = og.RRTConnect(si)
    elif planner_type == "sst":
        planner = oc.SST(si)
    else:
        raise ValueError("Unknown planner type")

    if hasattr(planner, 'setGoalBias'):
        planner.setGoalBias(config['hyperparameters']['goal_bias'])
    if hasattr(planner, 'setRange'):
        planner.setRange(1)
    
    return planner

def setup_benchmark(config_file, log_file):
    config = load_yaml(config_file)

    space = ob.SE2StateSpace()
    bounds = ob.RealVectorBounds(2)
    env_min = config["environment"]["min"]
    env_max = config["environment"]["max"]
    bounds.setLow(0, env_min[0])
    bounds.setLow(1, env_min[1])
    bounds.setHigh(0, env_max[0])
    bounds.setHigh(1, env_max[1])
    space.setBounds(bounds)

    uspace = oc.RealVectorControlSpace(space, 2)
    ubounds = ob.RealVectorBounds(2)
    ubounds.setLow(0, -0.5)
    ubounds.setHigh(0, 2)
    ubounds.setLow(1, -math.pi / 6)
    ubounds.setHigh(1, math.pi / 6)
    uspace.setBounds(ubounds)

    # # Space Information
    ss = oc.SimpleSetup(uspace)
    si = ss.getSpaceInformation()
    si.setPropagationStepSize(config['motionplanning']['dt'])
    si.setMinMaxControlDuration(1, 1)
    def statePropagator(start, control, duration, target):
        u1 = control[0]
        u2 = control[1]
        x = start.getX()
        y = start.getY()
        theta = start.getYaw()

        xn = u1 * math.cos(theta)
        yn = u1 * math.sin(theta)
        thetan = (u1/config['motionplanning']['L']) * math.tan(u2)

        target.setX(x + duration * xn)
        target.setY(y + duration * yn)
        target.setYaw(theta + duration * thetan)

        return target
    ss.setStatePropagator(oc.StatePropagatorFn(statePropagator))
    ss.setOptimizationObjective(getCost(si))
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(lambda state: is_state_valid(state, config)))

    # Start and Goal States
    start = ob.State(space)
    goal = ob.State(space)
    start[0] = config['motionplanning']['start'][0]
    start[1] = config['motionplanning']['start'][1]
    start[2] = config['motionplanning']['start'][2]
    goal[0] = config['motionplanning']['goal'][0]
    goal[1] = config['motionplanning']['goal'][1]
    goal[2] = config['motionplanning']['goal'][2]
    ss.setStartAndGoalStates(start, goal, config['hyperparameters']['goal_eps'])

    benchmark = ot.Benchmark(ss)

    samplers = [
        lambda si: ob.UniformValidStateSampler(si),
        lambda si: ob.BridgeTestValidStateSampler(si),
        lambda si: ob.ObstacleBasedValidStateSampler(si),
        lambda si: ob.GaussianValidStateSampler(si)
    ]
    sampler_names = ['Uniform', 'Bridge','ObstacleBased','Gaussian']

    for i,sampler in enumerate(samplers):
        si.setValidStateSamplerAllocator(ob.ValidStateSamplerAllocator(sampler))
        planner = configure_planner(config, si)
        planner.setName(str(sampler_names[i]))
        benchmark.addPlanner(planner)

    query = benchmark.Request()
    query.maxTime = config['hyperparameters']['timelimit']
    # query.maxTime = 3.0
    query.runCount = 1
    benchmark.benchmark(query)    
    benchmark.saveResultsToFile(log_file)

if __name__ == "__main__":
    
    config_file = sys.argv[1]
    log_file = sys.argv[2]
    setup_benchmark(config_file, log_file)
    
    #convert_command = ["python3", "ompl_benchmark_statistics.py", log_file, "-d", "car_1.db"]
    #subprocess.run(convert_command, check=True)

    #plot_command = ["python3", "ompl_benchmark_plotter.py", "car_1.db"]
    #subprocess.run(plot_command, check=True)
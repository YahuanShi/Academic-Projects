import argparse
import numpy as np
import yaml
import time
import fcl
import math
import sys

# Increase the recursion limit
sys.setrecursionlimit(10000)

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('action', help='input YAML file with action')
parser.add_argument('output_path', help='output YAML file with solution')
args = parser.parse_args()

# Load the action file
try:
    with open(args.action, 'r') as steam:
        object = yaml.safe_load(steam)
except Exception as e:
    print(f"Failed to load action file: {e}")
    sys.exit(1)

obs_objs = []

def Distance(q1, q2):
    if object["motionplanning"]["type"] == 'arm':
        dis1 = min(abs(q1[0] - q2[0]), 2 * np.pi - abs(q1[0] - q2[0]))
        dis2 = min(abs(q1[1] - q2[1]), 2 * np.pi - abs(q1[1] - q2[1]))
        dis3 = min(abs(q1[2] - q2[2]), 2 * np.pi - abs(q1[2] - q2[2]))
        distance = dis1 + dis2 + dis3

    elif object["motionplanning"]["type"] == 'car':
        p1 = np.array([q1[0], q1[1]])
        p2 = np.array([q2[0], q2[1]])
        distance = np.linalg.norm(p1 - p2) + min(abs(q1[2] - q2[2]), 2 * np.pi - abs(q1[2] - q2[2]))

    return distance

class KDNode:
    def __init__(self, point, left=None, right=None, parent=None, s=None, phi=None):
        self.point = point
        self.parent = parent
        self.left = left
        self.right = right
        self.s = s
        self.phi = phi

class KDTree:
    def __init__(self):
        self.root = None

    def addConfiguration(self, q, _parent, s, phi):
        if self.root is None:
            self.root = KDNode(q, parent=None)
        else:
            self.Add(q, self.root, 0, _parent, s, phi)

    def Add(self, q, node, depth, _parent, s, phi):
        axis = depth % len(q)
        if q[axis] < node.point[axis]:
            if node.left is None:
                node.left = KDNode(q, parent=_parent, s=s, phi=phi)
            else:
                self.Add(q, node.left, depth + 1, _parent, s, phi)
        else:
            if node.right is None:
                node.right = KDNode(q, parent=_parent, s=s, phi=phi)
            else:
                self.Add(q, node.right, depth + 1, _parent, s, phi)

    def nearestK(self, q, k):
        nearest = []
        self.calculate_nearestK(q, self.root, 0, k, nearest)
        return nearest

    def calculate_nearestK(self, q, node, depth, k, nearest):
        if node is None:
            return

        axis = depth % len(q)
        distance = Distance(q, node.point)

        if len(nearest) < k:
            nearest.append((node, distance))
            nearest.sort(key=lambda x: x[1])
        elif distance < nearest[-1][1]:
            nearest[-1] = (node, distance)
            nearest.sort(key=lambda x: x[1])

        if q[axis] < node.point[axis]:
            self.calculate_nearestK(q, node.left, depth + 1, k, nearest)
        else:
            self.calculate_nearestK(q, node.right, depth + 1, k, nearest)

        if abs(q[axis] - node.point[axis]) < nearest[-1][1]:
            if q[axis] < node.point[axis]:
                self.calculate_nearestK(q, node.right, depth + 1, k, nearest)
            else:
                self.calculate_nearestK(q, node.left, depth + 1, k, nearest)

    def nearestR(self, q, r):
        nearest = []
        self.calculate_nearestR(q, self.root, 0, r, nearest)
        return nearest

    def calculate_nearestR(self, q, node, depth, r, nearest):
        if node is None:
            return

        axis = depth % len(q)
        distance = Distance(q, node.point)

        if distance <= r:
            nearest.append((node.point, distance))
            nearest.sort(key=lambda x: x[1])

        if q[axis] - r < node.point[axis]:
            self.calculate_nearestR(q, node.left, depth + 1, r, nearest)

        if q[axis] + r >= node.point[axis]:
            self.calculate_nearestR(q, node.right, depth + 1, r, nearest)

def creat_Obs():
    for k, o in enumerate(object["environment"]["obstacles"]):
        if o["type"] == "box":
            p = o["pos"]
            s = o["size"]
            L, W, H = s[0], s[1], s[2]
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

def check_colli(p_start, p_goal, obs_objs):
    def Rmatrix(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

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

    if object["motionplanning"]["type"] == 'arm':
        L1, L2, L3 = object["motionplanning"]["L"]

        l1 = fcl.Box(L1, 0.1, 0.1)
        l2 = fcl.Box(L2, 0.1, 0.1)
        l3 = fcl.Box(L3, 0.1, 0.1)

        T1_start, T2_start, T3_start = get_Transform(p_start.point[0], p_start.point[1], p_start.point[2])

        Link1_start = fcl.CollisionObject(l1, T1_start)
        Link2_start = fcl.CollisionObject(l2, T2_start)
        Link3_start = fcl.CollisionObject(l3, T3_start)
        Link = [Link1_start, Link2_start, Link3_start]

        T1_goal, T2_goal, T3_goal = get_Transform(p_goal[0], p_goal[1], p_goal[2])
        T_goal = [T1_goal, T2_goal, T3_goal]

        request = fcl.ContinuousCollisionRequest()
        result = fcl.ContinuousCollisionResult()

        for i, l in enumerate(Link):
            for obs in obs_objs:
                ret = fcl.continuousCollide(l, T_goal[i], obs, obs.getTransform(), request, result)
                if ret != 1:
                    return ret
        return ret

    elif object["motionplanning"]["type"] == 'car':
        L = object["motionplanning"]["L"]
        W = object["motionplanning"]["W"]
        H = object["motionplanning"]["H"]
        b = fcl.Box(L, W, H)
        T = np.array([p_start.point[0], p_start.point[1], 0])
        c, s = np.cos(p_start.point[2]), np.sin(p_start.point[2])
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        t = fcl.Transform(R, T)
        car_model = fcl.CollisionObject(b, t)

        T_goal = np.array([p_goal[0], p_goal[1], 0])
        c_goal, s_goal = np.cos(p_goal[2]), np.sin(p_goal[2])
        R_goal = np.array([[c_goal, -s_goal, 0], [s_goal, c_goal, 0], [0, 0, 1]])
        t_goal = fcl.Transform(R_goal, T_goal)

        request = fcl.ContinuousCollisionRequest()
        result = fcl.ContinuousCollisionResult()

        for obs in obs_objs:
            ret = fcl.continuousCollide(car_model, t_goal, obs, obs.getTransform(), request, result)
            if ret != 1:
                return ret
        return ret

def main():
    nearest_point = None
    nearest_node = None
    nearest_dis = np.inf
    action_s = None
    action_phi = None

    def dynamics(dt, L, p):
        s = np.random.uniform(-0.5, 2)
        phi = np.random.uniform(-np.pi / 6, np.pi / 6)
        x, y, theta = p
        x_offset = s * math.cos(theta) * dt
        y_offset = s * math.sin(theta) * dt
        theta_offset = s * math.tan(phi) * dt / L
        return [x + x_offset, y + y_offset, theta + theta_offset], s, phi

    def rrt(object):
        env_min = np.array(object["environment"]["min"])
        env_max = np.array(object["environment"]["max"])
        start = object["motionplanning"]["start"]
        goal = object["motionplanning"]["goal"]
        timelimit = object["hyperparameters"]["timelimit"]
        goal_bias = object["hyperparameters"]["goal_bias"]
        goal_eps = object["hyperparameters"]["goal_eps"]
        dt = object["motionplanning"].get("dt", None)
        L = object["motionplanning"]["L"]

        KD_tree = KDTree()
        KD_tree.addConfiguration(start, None, None, None)

        best_solution = []
        best_action = []
        time_start = time.time()
        find_path = False
        n_recursion = 0

        def Expand(KD_tree, goal, mu):
            nonlocal nearest_dis, nearest_node, nearest_point, action_s, action_phi

            if object["motionplanning"]["type"] == 'arm':
                x_rand = goal if np.random.rand() < goal_bias else np.random.uniform([0, 0, 0], [2 * np.pi, 2 * np.pi, 2 * np.pi])
                x_near = KD_tree.nearestK(x_rand, 1)[0][0]
                length = Distance(x_rand, x_near.point)
                vector = (x_rand - np.array(x_near.point)) / length
                x_new = np.clip(np.array(x_near.point) + mu * vector, [0, 0, 0], [2 * np.pi, 2 * np.pi, 2 * np.pi]).tolist()

                if check_colli(x_near, x_new, obs_objs) == 1:
                    KD_tree.addConfiguration(x_new, x_near, None, None)
                    if Distance(x_new, goal) < goal_eps:
                        return [x_new, x_near]
                    elif not find_path:
                        dis_to_goal = Distance(x_new, goal)
                        if dis_to_goal < nearest_dis:
                            nearest_dis = dis_to_goal
                            nearest_point = x_new
                            nearest_node = x_near
                return False

            elif object["motionplanning"]["type"] == 'car':
                x_rand = goal if np.random.rand() < goal_bias else np.random.uniform([env_min[0], env_min[1], 0], [env_max[0], env_max[1], 2 * np.pi])
                x_new = None
                dis = np.inf
                x_near = KD_tree.nearestK(x_rand, 1)[0][0]

                for _ in range(5):
                    x_generated, s_generated, phi_generated = dynamics(dt, L, x_near.point)
                    dista = Distance(x_generated, x_rand)
                    if dista < dis:
                        x_new, dis, s, phi = x_generated, dista, s_generated, phi_generated

                if check_colli(x_near, x_new, obs_objs) == 1:
                    KD_tree.addConfiguration(x_new, x_near, s, phi)
                    if Distance(x_new, goal) < goal_eps:
                        return [x_new, x_near, s, phi]
                    elif not find_path:
                        dis_to_goal = Distance(x_new, goal)
                        if dis_to_goal < nearest_dis:
                            nearest_dis, nearest_point, nearest_node = dis_to_goal, x_new, x_near
                            action_s, action_phi = s, phi
                return False

        while True:
            n_recursion += 1
            path = []
            action = []
            ret = Expand(KD_tree, goal, 0.1)
            if ret:
                x_end, node = ret[:2]
                if object["motionplanning"]["type"] == 'car':
                    s, phi = ret[2], ret[3]
                path.append(x_end)
                if object["motionplanning"]["type"] == 'car':
                    action.append([s, phi])
                while node:
                    path.append(node.point)
                    if object["motionplanning"]["type"] == 'car' and node.s and node.phi:
                        action.append([node.s, node.phi])
                    node = node.parent
                path.reverse()
                if object["motionplanning"]["type"] == 'car':
                    action.reverse()

                if not best_solution or (len(path) < len(best_solution) and path):
                    best_solution, best_action = path, action

            if object["motionplanning"]["type"] == 'arm' and n_recursion > 1000:
                KD_tree = KDTree()
                KD_tree.addConfiguration(start, None, None, None)
                n_recursion = 0

            if best_solution:
                KD_tree = KDTree()
                KD_tree.addConfiguration(start, None, None, None)
                find_path = True

            if timelimit > 0 and time.time() - time_start >= timelimit:
                break

        if not best_solution:
            x_end, node = nearest_point, nearest_node
            best_solution.append(x_end)
            if object["motionplanning"]["type"] == 'car':
                best_action.append([action_s, action_phi])
            while node:
                best_solution.append(node.point)
                if object["motionplanning"]["type"] == 'car' and node.s and node.phi:
                    best_action.append([node.s, node.phi])
                node = node.parent
            best_solution.reverse()
            if object["motionplanning"]["type"] == 'car':
                best_action.reverse()

        if object["motionplanning"]["type"] == 'car':
            result = {
                'plan': {
                    'type': 'car',
                    'dt': dt,
                    'L': object["motionplanning"]["L"],
                    'W': object["motionplanning"]["W"],
                    'H': object["motionplanning"]["H"],
                    'states': best_solution,
                    'actions': best_action,
                }
            }
        else:
            result = {
                'plan': {
                    'type': 'arm',
                    'L': object["motionplanning"]["L"],
                    'states': best_solution
                }
            }

        result['results'] = [[[0.0, 1.5]], [[]]]

        try:
            with open(args.output_path, "w") as stream:
                yaml.dump(result, stream, default_flow_style=None, sort_keys=False)
        except Exception as e:
            print(f"Failed to write output file: {e}")
            sys.exit(1)

    creat_Obs()
    rrt(object)

if __name__ == "__main__":
    main()

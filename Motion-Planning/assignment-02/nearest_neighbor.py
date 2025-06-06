import argparse
import yaml
import numpy as np
import copy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('nn_0', help='input YAML file with nn_0')
    parser.add_argument('nn_sol', help='input YAML file with solution')
    args = parser.parse_args()

    with open(args.nn_0, "r") as stream:
        nn_0 = yaml.safe_load(stream)

    configurations = nn_0["configurations"]

    class KDNode:
        def __init__(self, point, left=None, right=None):
            self.point = point
            self.left = left
            self.right = right

    class KDTree:
        def __init__(self):
            self.root = None

        def addConfiguration(self, q):
            if self.root is None:
                self.root = KDNode(q)
            else:
                self._add(q, self.root, 0)

        def _add(self, q, node, depth):
            axis = depth % len(q)
            if q[axis] < node.point[axis]:
                if node.left is None:
                    node.left = KDNode(q)
                else:
                    self._add(q, node.left, depth + 1)
            else:
                if node.right is None:
                    node.right = KDNode(q)
                else:
                    self._add(q, node.right, depth + 1)

        def nearestK(self, q, k):
            nearest = []
            self._calculate_nearestK(q, self.root, 0, k, nearest)
            return [x[0] for x in nearest]

        def _calculate_nearestK(self, q, node, depth, k, nearest):
            if node is None:
                return
            axis = depth % len(q)
            distance = self._distance(q, node.point)

            if len(nearest) < k:
                nearest.append((node.point, distance))
                nearest.sort(key=lambda x: x[1])
            elif distance < nearest[-1][1]:
                nearest[-1] = (node.point, distance)
                nearest.sort(key=lambda x: x[1])

            if q[axis] < node.point[axis]:
                self._calculate_nearestK(q, node.left, depth + 1, k, nearest)
            else:
                self._calculate_nearestK(q, node.right, depth + 1, k, nearest)

            if abs(q[axis] - node.point[axis]) < nearest[-1][1]:
                if q[axis] < node.point[axis]:
                    self._calculate_nearestK(q, node.right, depth + 1, k, nearest)
                else:
                    self._calculate_nearestK(q, node.left, depth + 1, k, nearest)

        def nearestR(self, q, r):
            nearest = []
            self._calculate_nearestR(q, self.root, 0, r, nearest)
            return [x[0] for x in nearest]

        def _calculate_nearestR(self, q, node, depth, r, nearest):
            if node is None:
                return
            axis = depth % len(q)
            distance = self._distance(q, node.point)

            if distance <= r:
                nearest.append((node.point, distance))
                nearest.sort(key=lambda x: x[1])

            if q[axis] - r < node.point[axis]:
                self._calculate_nearestR(q, node.left, depth + 1, r, nearest)
            if q[axis] + r >= node.point[axis]:
                self._calculate_nearestR(q, node.right, depth + 1, r, nearest)

        def _distance(self, q1, q2):
            if nn_0["distance"] == 'l2':
                return np.linalg.norm(np.array(q1) - np.array(q2))
            elif nn_0["distance"] == 'angles':
                return sum(min(abs(a - b), 2 * np.pi - abs(a - b)) for a, b in zip(q1, q2))
            elif nn_0["distance"] == 'se2':
                xy_dist = np.linalg.norm(np.array(q1[:2]) - np.array(q2[:2]))
                theta_dist = min(abs(q1[2] - q2[2]), 2 * np.pi - abs(q1[2] - q2[2]))
                return xy_dist + theta_dist
            return float('inf')

    tree = KDTree()
    for config in configurations:
        tree.addConfiguration(config)

    resultK, resultR = [], []
    for query in nn_0["queries"]:
        q = query["q"]
        if query["type"] == "nearestK":
            k = query["k"]
            resultK.append([list(x) for x in tree.nearestK(q, k)])
        elif query["type"] == "nearestR":
            r = query["r"]
            resultR.append([list(x) for x in tree.nearestR(q, r)])

    results = {'results': resultK + resultR}

    with open(args.nn_sol, "w") as stream:
        yaml.dump(results, stream, default_flow_style=None, sort_keys=False)


if __name__ == "__main__":
    main()

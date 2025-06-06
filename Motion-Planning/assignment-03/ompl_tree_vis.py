import argparse
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import networkx as nx
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('export_planner_data', help='Export planner data')
    args = parser.parse_args()

    in_path = args.export_planner_data
    graph = nx.read_graphml(in_path)

    vis = meshcat.Visualizer()
    vis.open()

    for node, node_data in graph.nodes(data=True):
        position = [float(coord) for coord in node_data['coords'].split(',')]
        position[2] = 0
        sphere = g.Sphere(0.01)
        vis["tree"]["node"][str(node)].set_object(sphere)
        vis["tree"]["node"][str(node)].set_transform(tf.translation_matrix(position))

    for source_node, target_node in graph.edges():
        source_position = [float(coord) for coord in graph.nodes[source_node]['coords'].split(',')]
        target_position = [float(coord) for coord in graph.nodes[target_node]['coords'].split(',')]

        source_position[2] = 0
        target_position[2] = 0

        sphere = g.Sphere(0.01)

        vis["tree"]["node"][str(source_node)].set_object(sphere)
        vis["tree"]["node"][str(source_node)].set_transform(tf.translation_matrix(source_position))

        vis["tree"]["node"][str(target_node)].set_object(sphere)
        vis["tree"]["node"][str(target_node)].set_transform(tf.translation_matrix(target_position))

        vertices = np.column_stack([source_position, target_position])
        vis["tree"]["edge"][f"{source_node}-{target_node}"].set_object(g.Line(g.PointsGeometry(vertices)))

    input("Press Enter to exit...")

if __name__ == "__main__":
    main()

import argparse
import yaml
import time
import numpy as np
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from meshcat.animation import Animation

# Function to compute forward kinematics
def forward_kinematics(thetas, link_lengths):
    cumulative_theta = 0
    x_centers = []
    y_centers = []
    for theta, length in zip(thetas, link_lengths):
        cumulative_theta += theta
        x_center = length * np.cos(cumulative_theta)
        y_center = length * np.sin(cumulative_theta)
        x_centers.append(x_center)
        y_centers.append(y_center)
    return x_centers, y_centers, np.cumsum(thetas)

# Main function
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('env', help='input YAML file with environment')
    parser.add_argument('plan', help='input YAML file with plan')
    parser.add_argument('--dt', type=float, default=0.1, help='sleeping time between frames')
    parser.add_argument('--output', default=None, help='output file with animation')
    args = parser.parse_args()

    # Load environment parameters
    with open(args.env, "r") as stream:
        env = yaml.safe_load(stream)

    # Load plan parameters
    with open(args.plan, "r") as stream:
        plan = yaml.safe_load(stream)

    # Initialize MeshCat visualizer
    vis = meshcat.Visualizer()

    # Set up visualization of obstacles in the environment
    for k, obstacle in enumerate(env["environment"]["obstacles"]):
        if obstacle["type"] == "box":
            pos = obstacle["pos"]
            size = obstacle["size"]
            vis["obstacles"][str(k)].set_object(g.Box(size))
            vis["obstacles"][str(k)].set_transform(tf.translation_matrix(pos))
        elif obstacle["type"] == "cylinder":
            pos = obstacle["pos"]
            quat = obstacle["q"]
            radius = obstacle["r"]
            length = obstacle["lz"]
            vis["obstacles"][str(k)].set_object(g.Cylinder(length, radius))
            vis["obstacles"][str(k)].set_transform(
                tf.translation_matrix(pos).dot(
                    tf.quaternion_matrix(quat)).dot(
                        tf.euler_matrix(np.pi / 2, 0, 0)))
        else:
            raise RuntimeError("Unknown obstacle type " + obstacle["type"])

    # Set up visualization of robot links
    link_lengths = plan["plan"]["L"]
    for i, length in enumerate(link_lengths):
        vis[f"link_{i+1}"].set_object(g.Cylinder(length, 0.05), g.MeshLambertMaterial(
            color=0xff0000, opacity=0.5))

    # Create animation object
    anim = Animation()

    # Update poses of robot links based on plan states
    for k, thetas in enumerate(plan["plan"]["states"]):
        x_centers, y_centers, thetas = forward_kinematics(thetas, link_lengths)
        with anim.at_frame(vis, k * args.dt) as frame:
            for i, (x_center, y_center, theta) in enumerate(zip(x_centers, y_centers, thetas)):
                transform_matrix = tf.translation_matrix([x_center, y_center, 0]).dot(
                    tf.euler_matrix(np.pi/2, 0, np.pi/2 + theta)).dot(tf.euler_matrix(np.pi/2, 0, 0))
                frame[f"link_{i+1}"].set_transform(transform_matrix)

    # Set up the animation
    vis.set_animation(anim)

    # Display or save the animation
    if args.output is None:
        vis.open()
        time.sleep(1e9)
    else:
        with open(args.output, "w") as f:
            f.write(vis.static_html())

if __name__ == "__main__":
    main()

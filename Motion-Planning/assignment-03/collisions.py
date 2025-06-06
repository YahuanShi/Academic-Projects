import argparse
import yaml
import numpy as np
import fcl
import meshcat.transformations as tf

# Function to create collision manager for environment obstacles
def create_collision_manager_env(env_config):
    manager = fcl.DynamicAABBTreeCollisionManager()
    obstacles = []
    for obstacle in env_config["environment"]["obstacles"]:
        position = obstacle["pos"]
        if obstacle["type"] == "box":
            size = obstacle["size"]
            obstacles.append(fcl.CollisionObject(fcl.Box(size[0], size[1], size[2]), fcl.Transform(position)))
        elif obstacle["type"] == "cylinder":
            rotation = obstacle["q"]
            radius = obstacle["r"]
            length = obstacle["lz"]
            obstacles.append(fcl.CollisionObject(fcl.Cylinder(radius, length), fcl.Transform(rotation, position)))
        else:
            raise ValueError("Unknown obstacle type: " + obstacle["type"])

    manager.registerObjects(obstacles)
    manager.setup()
    return manager

# Function to create collision manager for car
def create_collision_manager_for_car():
    manager = fcl.DynamicAABBTreeCollisionManager()
    length = 3.0
    width = 1.5
    height = 1.0
    car_obj = [fcl.CollisionObject(fcl.Box(length, width, height))]
    manager.registerObjects(car_obj)
    manager.setup()

    # Function to update car manager with new state
    def update_manager(manager, state):
        objects = manager.getObjects()
        car = objects[0]
        car.setTranslation(state[0:3])
        manager.update(car)

    return manager, update_manager

# Function to create collision manager for robotic arm
def create_collision_manager_for_arm():
    manager = fcl.DynamicAABBTreeCollisionManager()
    radius = 0.04
    length = 1.0
    arm_parts = [fcl.CollisionObject(fcl.Cylinder(radius, length)),
                 fcl.CollisionObject(fcl.Cylinder(radius, length)),
                 fcl.CollisionObject(fcl.Cylinder(radius, length))]
    manager.registerObjects(arm_parts)
    manager.setup()

    # Function to update arm manager with new state
    def update_manager(manager, state):
        objects = manager.getObjects()
        theta_1, theta_2, theta_3 = state
        lengths = [1, 1, 1]

        # Calculate positions of arm parts based on joint angles
        x1 = lengths[0] / 2 * np.cos(theta_1)
        y1 = lengths[0] / 2 * np.sin(theta_1)

        x2 = lengths[0] * np.cos(theta_1) + lengths[1] / 2 * np.cos(theta_1 + theta_2)
        y2 = lengths[0] * np.sin(theta_1) + lengths[1] / 2 * np.sin(theta_1 + theta_2)

        x3 = lengths[0] * np.cos(theta_1) + lengths[1] * np.cos(theta_1 + theta_2) + lengths[2] / 2 * np.cos(theta_1 + theta_2 + theta_3)
        y3 = lengths[0] * np.sin(theta_1) + lengths[1] * np.sin(theta_1 + theta_2) + lengths[2] / 2 * np.sin(theta_1 + theta_2 + theta_3)

        offset = np.pi / 2
        # Update transformation matrices for arm parts
        T1 = tf.translation_matrix([x1, y1, 0]).dot(tf.euler_matrix(np.pi / 2, 0, offset + theta_1))
        objects[0].setTransform(fcl.Transform(T1[0:3, 0:3], T1[0:3, 3]))

        T2 = tf.translation_matrix([x2, y2, 0]).dot(tf.euler_matrix(np.pi / 2, 0, offset + theta_1 + theta_2))
        objects[1].setTransform(fcl.Transform(T2[0:3, 0:3], T2[0:3, 3]))

        T3 = tf.translation_matrix([x3, y3, 0]).dot(tf.euler_matrix(np.pi / 2, 0, offset + theta_1 + theta_2 + theta_3))
        objects[2].setTransform(fcl.Transform(T3[0:3, 0:3], T3[0:3, 3]))

        manager.update()

    return manager, update_manager

# Function to create collision manager based on plan type
def create_collision_manager(plan_config):
    plan_type = plan_config["plan"]["type"]
    if plan_type == "car":
        return create_collision_manager_for_car()
    elif plan_type == "arm":
        return create_collision_manager_for_arm()
    else:
        raise ValueError("Unknown plan type: " + plan_type)

# Main function
def main():
    parser = argparse.ArgumentParser(description='Collision check for given environment and plan.')
    parser.add_argument('env', help='Input YAML file with environment configuration')
    parser.add_argument('plan', help='Input YAML file with plan configuration')
    parser.add_argument('output', help='Output YAML file with collision results')
    args = parser.parse_args()

    # Load environment configuration from YAML file
    with open(args.env, 'r') as env_file:
        env_config = yaml.safe_load(env_file)

    # Load plan configuration from YAML file
    with open(args.plan, 'r') as plan_file:
        plan_config = yaml.safe_load(plan_file)

    # Create collision manager for environment obstacles
    env_manager = create_collision_manager_env(env_config)
    # Create collision manager for plan based on plan type
    plan_manager, update_plan_manager = create_collision_manager(plan_config)

    collision_results = []
    # Iterate through states in the plan configuration
    for state in plan_config["plan"]["states"]:
        # Perform collision check for each state
        collision_request = fcl.CollisionRequest()
        collision_data = fcl.CollisionData(request=collision_request)
        update_plan_manager(plan_manager, np.array(state))
        env_manager.collide(plan_manager, collision_data, fcl.defaultCollisionCallback)
        collision_results.append(collision_data.result.is_collision)

    # Write collision results to output YAML file
    result = {'collisions': collision_results}
    with open(args.output, 'w') as output_file:
        yaml.dump(result, output_file)

if __name__ == "__main__":
    main()

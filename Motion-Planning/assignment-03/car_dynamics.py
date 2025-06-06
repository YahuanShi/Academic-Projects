import yaml
import sys
import math

def load_yaml(filename):
    """Load YAML data from a file."""
    with open(filename, 'r') as file:
        return yaml.safe_load(file)

def save_yaml(data, filename):
    """Save data to a YAML file."""
    with open(filename, 'w') as file:
        yaml.dump(data, file)

def car_dynamics(actions_file, plan_file):
    """Simulate car dynamics based on actions and save the result to a plan file."""
    # Load actions data
    actions_data = load_yaml(actions_file)

    dt = actions_data['dt']
    L = actions_data['L']
    W = actions_data['W']  # Adding car width information
    H = actions_data['H']  # Adding car height information
    start = actions_data['start']
    actions = actions_data['actions']

    # Initialize state
    x, y, theta = start
    states = [[x, y, theta]]  # Adding initial state

    for s, phi in actions:
        # Euler integration to update position and orientation
        x += s * math.cos(theta) * dt
        y += s * math.sin(theta) * dt
        theta += (s / L) * math.tan(phi) * dt
        states.append([x, y, theta])

    # Prepare plan data
    plan_data = {
        'plan': {
            'type': 'car',
            'dt': dt,
            'L': L,
            'W': W,
            'H': H,
            'states': states,
            'actions': actions
        }
    }

    # Save the result to the plan file
    save_yaml(plan_data, plan_file)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python car_dynamics.py <actions_file> <plan_file>")
    else:
        car_dynamics(sys.argv[1], sys.argv[2])

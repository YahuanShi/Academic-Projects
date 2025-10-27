import numpy as np
import matplotlib.pyplot as plt

def simulate_spring_mass(x_init, v_init, x_eq, k, m, b, t_max):
    """
    Discrete-time equations:
        v_{t+1} = v_t - (b/m)*v_t - (k/m)*(x_t - x_eq)
        x_{t+1} = x_t + v_{t+1}
    
    Parameters:
        x_init, v_init : Initial position and velocity
        x_eq           : Equilibrium position of the spring
        k, m, b        : Spring constant, mass, damping coefficient
        t_max          : Maximum number of time steps for simulation
    
    Returns:
        ts, xs, vs : Arrays of time, position, and velocity
    """
    xs = [x_init]
    vs = [v_init]
    ts = [0]
    x, v = x_init, v_init
    for t in range(1, t_max + 1):
        v_next = v - (b/m) * v - (k/m) * (x - x_eq)
        x_next = x + v_next
        x, v = x_next, v_next
        xs.append(x)
        vs.append(v)
        ts.append(t)
    return np.array(ts), np.array(xs), np.array(vs)

def plot_trajectory(ts, xs, vs, title, filename):
    """
    Plot the position and velocity trajectories over time.
    """
    plt.figure()
    plt.plot(ts, xs, label="Position")
    plt.plot(ts, vs, label="Velocity")
    plt.title(title)
    plt.xlabel("Time step (t)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    plt.close()
    print(f"Saved: {filename}")

# Parameters
K = 7.0
M = 30.0
X0 = 161.0
T_MAX = 500

# Undamped system
ts, xs, vs = simulate_spring_mass(
    x_init=200.0, v_init=0.0,
    x_eq=X0, k=K, m=M, b=0.0, t_max=T_MAX
)
plot_trajectory(ts, xs, vs, "Spring-Mass System (No Damper)", "mass.png")

# Damped system (b=1)
ts_d, xs_d, vs_d = simulate_spring_mass(
    x_init=200.0, v_init=0.0,
    x_eq=X0, k=K, m=M, b=1.0, t_max=T_MAX
)
plot_trajectory(ts_d, xs_d, vs_d, "Spring-Mass-Damper System (b=1)", "mass_damper.png")


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style='ticks', palette='Set2')


def compute_spline(start_pos, final_pos, start_velocity, final_velocity, start_time, final_time):
    stepsize = 0.01
    t = np.arange(start_time, final_time + stepsize, stepsize)

    dt = final_time - start_time
    a0 = start_pos
    a1 = start_velocity
    a2 = 3 / (dt**2) * (final_pos - start_pos) - 2 / dt * start_velocity - 1 / dt * final_velocity
    a3 = -2 / (dt**3) * (final_pos - start_pos) + 1 / (dt**2) * (start_velocity + final_velocity)

    f = a0 + a1*(t - start_time) + a2*(t - start_time)**2 + a3*(t - start_time)**3
    return t, f


def velocity_heuristic(start, via, end):
    if start <= via <= end or start >= via >= end:
        return 0.5 * ((via - start)/2.5 + (end - via)/2.5)
    return 0


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(10, 6))

    start = [0, 0, 0]
    via = [-np.pi/4, np.pi/2, 0]
    end = [-np.pi/2, np.pi/4, 0]

    for i in range(3):
        via_vel = velocity_heuristic(start[i], via[i], end[i])

        t1, f1 = compute_spline(start[i], via[i], 0, via_vel, 0, 2.5)
        t2, f2 = compute_spline(via[i], end[i], via_vel, 0, 2.5, 5.0)

        sns.lineplot(x=t1, y=f1, ax=ax, label=f"q{i+1} (segment 1)")
        sns.lineplot(x=t2, y=f2, ax=ax, label=f"q{i+1} (segment 2)")

        dotsize = 80
        ax.scatter(0.0, start[i], s=dotsize, zorder=3)
        ax.scatter(2.5, via[i], s=dotsize, zorder=3)
        ax.scatter(5.0, end[i], s=dotsize, zorder=3)

    ax.set_title("Joint Trajectories")
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("Joint angle q")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()

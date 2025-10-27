import numpy as np
import matplotlib.pyplot as plt

def simulate_spring_mass(x_init, v_init, x_eq, k, m, b, t_max):
    """
    离散方程：
        v_{t+1} = v_t - (b/m)*v_t - (k/m)*(x_t - x_eq)
        x_{t+1} = x_t + v_{t+1}
    参数：
        x_init, v_init : 初始位置与速度
        x_eq           : 弹簧平衡位置
        k, m, b        : 弹簧常数、质量、阻尼系数
        t_max          : 模拟的最大时间步数
    返回：
        ts, xs, vs ：时间、位置、速度的数组
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

# 参数设置
K = 7.0
M = 30.0
X0 = 161.0
T_MAX = 500

# 无阻尼系统
ts, xs, vs = simulate_spring_mass(
    x_init=200.0, v_init=0.0,
    x_eq=X0, k=K, m=M, b=0.0, t_max=T_MAX
)
plot_trajectory(ts, xs, vs, "Spring-Mass System (No Damper)", "mass.png")

# 带阻尼系统（b=1）
ts_d, xs_d, vs_d = simulate_spring_mass(
    x_init=200.0, v_init=0.0,
    x_eq=X0, k=K, m=M, b=1.0, t_max=T_MAX
)
plot_trajectory(ts_d, xs_d, vs_d, "Spring-Mass-Damper System (b=1)", "mass_damper.png")

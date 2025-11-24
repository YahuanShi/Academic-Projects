import numpy as np
import matplotlib.pyplot as plt


radius = 0.2
x_center = [0.6, 0.35]
circle_iters = 3.0

vel_max = (2.0 * np.pi) / 5.0
acc_max = (2.0 * np.pi) / 25.0

total_traj_len = 2 * np.pi * circle_iters

t_blend = vel_max / acc_max
  
blend_dist = acc_max * t_blend**2
const_dist = total_traj_len - blend_dist

t_const = const_dist / vel_max
total_time = 2 * t_blend + t_const
  
ts = np.linspace(0, total_time, 5000)

beta = []
beta_dot = []
for t in ts:
  
    if t < t_blend:
        target_vel = acc_max * t
        target_pos = 0.5 * acc_max * t**2

    elif t < total_time - t_blend:
        target_vel = vel_max
        target_pos = 0.5 * acc_max * t_blend**2 + vel_max * (t - t_blend)
    
    elif t < total_time:
        target_vel = vel_max - acc_max * (t - (total_time - t_blend))
        target_pos = total_traj_len - 0.5 * acc_max * (total_time - t)**2
    else:
        target_vel = 0.0
        target_pos = total_traj_len

    beta.append(target_pos)
    beta_dot.append(target_vel)

print(ts.shape)
plt.plot(ts, beta, label="beta")
plt.plot(ts, beta_dot, label="beta_dot")
plt.legend()
plt.grid()
plt.savefig("betas")

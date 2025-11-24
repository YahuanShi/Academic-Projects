import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

t_start = 0
t1 = 2.5 
tf = 5.0  

q_a = np.array([0.0, 0.0, 0.0])
q_b = np.array([-pi / 4, pi / 2, 0.0])
q_c = np.array([-pi / 2, pi / 4, 0.0])

v_avg_1 = (q_b - q_a) / t1
v_avg_2 = (q_c - q_b) / (tf - t1)

q_dot_a = np.zeros(3)
q_dot_b = np.zeros(3)
q_dot_c = np.zeros(3)

for i in range(3):
    v1 = v_avg_1[i]
    v2 = v_avg_2[i]

    if (v1 * v2 < 0):
        q_dot_b[i] = 0.0
    else:
        q_dot_b[i] = (v1 + v2) / 2

A_params = np.zeros((3, 2, 4)) 



# What is known: q(0), q(2,5), q'(0), q'(2,5) (first segment)

M_s = np.zeros((3, 2, 4, 4))
V_s = np.zeros((3,2,4))
S_s = np.zeros((3,2,4))

def q(t):
    # q(t)  = a1 + a2 * t + a3*t**2 + a4*t**3
    return np.array([1, t, t**2, t**3])

def q_dot(t):
    # q'(t) = 0 * a1 + a2 + 2*a3*t + 3*a4*t**2
    return np.array([0, 1, 2*t, 3*t**2])



for j in range(3):
    M_s[j][0] = np.array([q(0),q(2.5),q_dot(0),q_dot(2.5)])
    V_s[j][0] = np.array([q_a[j], q_b[j], q_dot_a[j], q_dot_b[j]])    

for j in range(3):
    M_s[j][1] = np.array([q(2.5),q(5),q_dot(2.5),q_dot(5)])
    V_s[j][1] = np.array([q_b[j], q_c[j], q_dot_b[j], q_dot_c[j]])  
    

# solve the equations:
for j in range(3):
    for s in range(2):
        M_s_inv = np.linalg.inv(M_s[j][s])
        S_s[j][s] = M_s_inv  @ V_s[j][s]

t_vals = np.linspace(0, tf, 400)
q_full = np.zeros((3, len(t_vals)))

for idx, t in enumerate(t_vals):
    if t <= t1:
        seg = 0
    else:
        seg = 1
    a = S_s[:, seg, :]
    q_full[:, idx] = a[:,0] + a[:,1]*t + a[:,2]*t*t + a[:,3]*t*t*t

# Plot final trajectories
plt.figure(figsize=(10,6))
plt.plot(t_vals, q_full[0], label="Joint 1")
plt.plot(t_vals, q_full[1], label="Joint 2")
plt.plot(t_vals, q_full[2], label="Joint 3")

plt.xlabel("Time (s)")
plt.ylabel("Joint Angle (rad)")
plt.title("Joint Space Trajectories (One Continuous Curve per Joint)")
plt.legend()
plt.grid(True)
plt.show()
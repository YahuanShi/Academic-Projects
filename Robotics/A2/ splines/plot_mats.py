import numpy as np
import matplotlib.pyplot as plt

# === Load the data ===
# Replace 'data.txt' with your filename
data = np.loadtxt('data_proj2_700_500_250.mat')

# === Parse columns ===
# Adjust the indices according to your file format
# Example structure based on your description:
# q(3) dq(3) qd(3) tau(3)
# If your file repeats these sets for multiple joints, adjust accordingly.
cutoff = 4200
# cutoff = -1
time = data[:cutoff, 0]
q = data[:cutoff, 1:4]       # columns 0,1,2
dq = data[:cutoff, 4:7]      # columns 3,4,5
qd = data[:cutoff, 7:10]      # columns 6,7,8
tau = data[:cutoff, 10:13]    # columns 9,10,11
x = data[:cutoff, 13:16]
xd = data[:cutoff, 16:19]
xd = data[:cutoff, 19:22]
print(data.shape)

# === Plotting ===
e = x - xd

for i in range(5):
    plt.figure(figsize=(10, 5))

    if i == 0:
        name = "qs"
        plt.plot(time, q[:,0], marker='', linestyle='-', linewidth=1.5, label="q1")
        plt.plot(time, q[:,1], marker='', linestyle='-', linewidth=1.5, label="q2")
        plt.plot(time, q[:,2], marker='', linestyle='-', linewidth=1.5, label="q3")
        plt.plot(time, qd[:,0], marker='', linestyle='-', linewidth=1.5, label="qd1")
        plt.plot(time, qd[:,1], marker='', linestyle='-', linewidth=1.5, label="qd2")
        plt.plot(time, qd[:,2], marker='', linestyle='-', linewidth=1.5, label="qd3")
    if i == 1:
        name = "taus"
        plt.plot(time, tau[:,0], marker='', linestyle='-', linewidth=1.5, label="tau1")
        plt.plot(time, tau[:,1], marker='', linestyle='-', linewidth=1.5, label="tau2")
        plt.plot(time, tau[:,2], marker='', linestyle='-', linewidth=1.5, label="tau3")
    if i == 2:
        name = "x_xd"
        plt.plot(time, x[:,0], marker='', linestyle='-', linewidth=1.5, label="x_x")
        plt.plot(time, x[:,1], marker='', linestyle='-', linewidth=1.5, label="x_y")
        plt.plot(time, x[:,2], marker='', linestyle='-', linewidth=1.5, label="x_phi")
        plt.plot(time, xd[:,0], marker='', linestyle='-', linewidth=1.5, label="xd_x")
        plt.plot(time, xd[:,1], marker='', linestyle='-', linewidth=1.5, label="xd_y")
        plt.plot(time, xd[:,2], marker='', linestyle='-', linewidth=1.5, label="xd_phi")
    if i == 3:
        name = "e"
        plt.plot(time, e[:, 0], marker='', linestyle='-', linewidth=1.5, label="x_ex")
        plt.plot(time, e[:, 1], marker='', linestyle='-', linewidth=1.5, label="x_ey")
        plt.plot(time, e[:, 2], marker='', linestyle='-', linewidth=1.5, label="x_ephi")
    if i == 4:
        name = "traj"
        plt.plot(x[:, 0], x[:, 1], marker='', linestyle='-', linewidth=1.5, label="x")
        plt.plot(xd[:, 0], xd[:, 1], marker='', linestyle='-', linewidth=1.5, label="xd")
        plt.axis("equal")

    plt.legend()
    plt.title('Values Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{name}.png")
    # plt.show()

"""Solutions to Deep Learning 2 Sheet 7: Neural ODEs"""

# Task 1
class EulerSolver(ODESolver):
    def __init__(self, dt):
        super().__init__()
        self.dt = dt

    def step(self, f, t, u):
        t_next = t + self.dt
        u_next = u + self.dt * f(t, u)
        return t_next, u_next

# Task 2
class RungeKuttaSolver(ODESolver):
    def __init__(self, dt):
        super().__init__()
        self.dt = dt

    def step(self, f, t, u):
        dt = self.dt
        k1 = f(t, u)
        k2 = f(t + dt / 2, u + dt / 2 * k1)
        k3 = f(t + dt / 2, u + dt / 2 * k2)
        k4 = f(t + dt, u + dt * k3)

        t_next = t + self.dt
        u_next = u + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return t_next, u_next

# Task 3
def loss_fn(u, u_pred):
    return torch.mean(torch.norm(u - u_pred, dim=1))
    
# Task 4
for i in range(1, n_iters + 1):
    optimizer.zero_grad()

    u_pred = odeint(neural_ode, u0_torch, t_torch)
    loss = loss_fn(u_target, u_pred)
    loss.backward()
    optimizer.step()

    if i == 1 or i % print_every == 0:
        print(f"Step: {i :4d} | Loss: {loss.item() :.6f}")
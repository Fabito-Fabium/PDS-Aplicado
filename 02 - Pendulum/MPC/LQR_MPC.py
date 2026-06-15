import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import control
import scipy.linalg as sl

# %%
### ml*(D^2 theta) + cl(D theta) + mg sin(theta) = 0
M = .5                  # Cart mass [kg]
k = .1                  # Cart friction [N.s/m]

l = .3                  # Length [m]
m = .2                  # Mass [Kg] (little mass)
c = .01                 # Friction [N.s/rad] (Perda no eixo)
g = 9.81                # Gravity [m/s^2]

Lt = 60                 # Simulation time [s]
dt = 0.01               # Discretization [s]
Nt = round(Lt/dt)

# %%
def fd_step(x_1, x_2, theta_1, theta_2, F):
    xd  = (x_1     - x_2)     / dt
    thd = (theta_1 - theta_2) / dt

    denom = M + m * np.sin(theta_1)**2
    xdd   = (F - k*xd
               - m*g*np.sin(theta_1)*np.cos(theta_1)
               + c*thd*np.cos(theta_1)
               + m*l*thd**2*np.sin(theta_1)) / denom

    thdd  = (m*g*np.sin(theta_1) - c*thd - m*xdd*np.cos(theta_1)) / (m*l)

    x_new  = xdd*dt**2 + 2*x_1     - x_2
    th_new = thdd*dt**2 + 2*theta_1 - theta_2
    return float(x_new), float(th_new)

# %%
Ac = np.array([
    [0,  1,              0,                    0               ],
    [0, -k/M,           -m*g/M,               +c/M             ],
    [0,  0,              0,                    1               ],
    [0, +k/(M*l),  +(M+m)*g/(M*l),       -(M+m)*c/(M*m*l)     ]
])

Bc = np.array([
    [ 0      ],
    [ 1/M    ],
    [ 0      ],
    [-1/(M*l)]
])

n = Ac.shape[0]
m = Bc.shape[1]
Z = np.zeros((n+m, n+m))

Z[:n, :n] = Ac
Z[:n, n:] = Bc

eZ = sl.expm(Z * dt)

Ad = eZ[:n, :n]   # bloco superior esquerdo
Bd = eZ[:n, n:]   # bloco superior direito

# %%
#        [x, \dot x, \theta, \dot\theta]
Q = np.diag([20.0, 1.0, 500.0, 20.0])
R = np.array([[0.005]])

# scipy.linalg.solve_discrete_are : Solves the discrete-time algebraic Riccati equation (DARE).
K_lqr, P_lqr, _ = control.dlqr(Ad, Bd, Q, R)

# %%
DISTURB_MAX = 100
USE_MPC = True
# %% Definição do MPC
N_mpc = 20
F_max = np.min([80, DISTURB_MAX])

# define as variáveis do problema
X_var    = cp.Variable((4, N_mpc + 1))
U_var    = cp.Variable((1, N_mpc))
x0_param = cp.Parameter(4)

#### Dada a função custo J
J = 0
for t in range(N_mpc):
    J += cp.quad_form(X_var[:, t], Q) + cp.quad_form(U_var[:, t], R)

J += cp.quad_form(X_var[:, N_mpc], P_lqr)

#### e as restrições Cons
constraints = [X_var[:, 0] == x0_param]
for t in range(N_mpc):
    constraints += [X_var[:, t+1] == Ad @ X_var[:, t] + Bd @ U_var[:, t]]
    constraints += [U_var[:, t] <= F_max, U_var[:, t] >= -F_max]

#### minimize J sujeita a Cons
mpc_prob = cp.Problem(cp.Minimize(J), constraints)
# %%
tht0 = 1e-9
tht  = np.zeros(Nt); tht[:3] = tht0
x    = np.zeros(Nt)
F_hist = np.zeros((Nt, 2))

# %%
fig, ax = plt.subplots(figsize=(12, 5))
xlim = 5 * l; ylim = 1.1 * l
ax.set_xlim(-xlim, xlim); ax.set_ylim(-ylim, ylim)
ax.set_aspect('equal', 'box')

def coords(cx, th):
    return ((cx - l*np.sin(th), cx, cx - l/2, cx - l/2, cx + l/2, cx + l/2, cx),
            ( l*np.cos(th),     0,  0,        -l/2,      -l/2,      0,        0))

xx, yy = coords(0, tht0)
line = ax.plot(xx[0], yy[0], "bo", xx, yy, "b")

# %%
from time import time

t2 = time()
txt_t = ax.text(0.0, 1.02, '', transform=ax.transAxes, fontsize=12)
txt_th = ax.text(0.40, 1.02, '', transform=ax.transAxes, fontsize=12)
txt_F = ax.text(0.75, 1.02, '', transform=ax.transAxes, fontsize=12)

if USE_MPC: ctrl = "MPC"
else: ctrl = "LQR"
fig.suptitle(ctrl, fontsize=15, fontweight='bold')

for nt in range(3, Nt):
    t1 = t2

    # Estado atual
    x_d  = (x[nt-1]   - x[nt-3]) / (2*dt)
    th_d = (tht[nt-1] - tht[nt-3]) / (2*dt)
    xk   = np.array([x[nt-1], x_d, tht[nt-1], th_d])


    if USE_MPC:
        x0_param.value = xk
        mpc_prob.solve(solver=cp.CLARABEL, warm_start=True)
        F = (U_var.value[0, 0])
    else:
        F = -(K_lqr @ xk)[0]

    F_hist[nt, 0] = F

    disturb = DISTURB_MAX * np.random.uniform(-1, 1)
    F_hist[nt, 1] = disturb
    F_total = F + disturb

    x[nt], tht[nt] = fd_step(x[nt-1], x[nt-2], tht[nt-1], tht[nt-2], F_total)


    if x[nt] > xlim:
        x -= 2 * xlim
    elif x[nt] < -xlim:
        x += 2 * xlim

    xx, yy = coords(x[nt], tht[nt])
    line[0].set_data([xx[0]], [yy[0]])
    line[1].set_data(xx, yy)

    ctrl = "MPC" if USE_MPC else "LQR"
    txt_t.set_text(f"t={nt * dt:.2f}s")
    txt_th.set_text(f"abs theta max={np.max(np.abs(np.degrees(tht))):.2f}°")
    txt_F.set_text(f"abs F control max={np.max(np.abs(F_hist[:, 0])):.1f}N")

    t2 = time()
    if nt % 10 == 0:
        plt.pause(max(1e-12, dt - (t2 - t1)))

plt.close('all')

# %%
t_vec = np.arange(Nt) * dt
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
ctrl = "MPC" if USE_MPC else "LQR"
fig.suptitle(f"{ctrl} — Inverted Pendulum", fontsize=13)

axes[0].plot(t_vec, np.degrees(tht));          axes[0].set_ylabel("theta [deg]")
axes[1].plot(t_vec, x, 'tab:orange');          axes[1].set_ylabel("x [m]")
axes[2].plot(t_vec, F_hist[:,0], 'tab:red');   axes[2].set_ylabel("F control [N]")
axes[3].plot(t_vec, F_hist[:,1], 'tab:green'); axes[3].set_ylabel("Disturbance [N]")

for ax in axes: ax.grid(alpha=.3); ax.axhline(0, color='k', lw=.5, ls='--')
plt.tight_layout()
plt.show()
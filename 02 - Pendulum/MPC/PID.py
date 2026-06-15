import matplotlib
matplotlib.use('TkAgg')
# %%
import numpy as np
import matplotlib.pyplot as plt
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

# %% forward
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
tht0 = 1e-9
tht = np.zeros(Nt)
tht[:3] = tht0

x = np.zeros(Nt)
x0 = 0

e = np.zeros(Nt)

F_hist = np.zeros([Nt, 2]) # 0 => força rand, 1 => força contr
# %%
fig, ax = plt.subplots(figsize=(12, 5))
xlim = 5 * l
ylim = 1.1 * l
ax.set_xlim(-xlim, xlim)
ax.set_ylim(-ylim,ylim)
ax.set_aspect('equal', 'box')

def coords(x, tht):
    return ((x - l * np.sin(tht), x, x - l / 2, x - l / 2, x + l / 2, x + l / 2, x),
            (l * np.cos(tht), 0, 0, -l / 2, -l / 2, 0, 0))

xx, yy = coords(x0, tht0)
line = ax.plot(xx[0], yy[0], "bo", xx, yy, "b")
# %%
from time import time
t2 = time()

txt_t = ax.text(0.0, 1.02, '', transform=ax.transAxes, fontsize=12)
txt_th = ax.text(0.40, 1.02, '', transform=ax.transAxes, fontsize=12)
txt_F = ax.text(0.75, 1.02, '', transform=ax.transAxes, fontsize=12)
fig.suptitle('PID', fontsize=15, fontweight='bold')

DISTURB_MAX = 15

P = 15
I = 25
D = 1
E = 0

for nt in range(3, Nt):
    t1 = t2
    thtd = (tht[nt - 1] - tht[nt - 2]) / ( dt)
    thtdd = (tht[nt - 1] - 2 * tht[nt - 2] + tht[nt - 3]) / (dt ** 2)
    set_point = 0

    e[nt] = (tht[nt - 1])
    E += e[nt] * dt
    de = (e[nt] - e[nt - 1]) / (dt)
    F = P * e[nt] + I * E + D * de
    F_hist[nt, 0] = F
    disturb = DISTURB_MAX * (np.random.uniform(-1, 1))

    F_hist[nt, 1] = disturb
    F += disturb

    x[nt], tht[nt] = fd_step(x[nt-1], x[nt-2], tht[nt-1], tht[nt-2], F)

    if x[nt] > xlim:
        x -= 2 * xlim
    elif x[nt] < -xlim:
        x += 2 * xlim

    xx, yy = coords(x[nt], tht[nt])
    line[0].set_data([xx[0]], [yy[0]])
    line[1].set_data(xx, yy)

    ctrl = 'PID'
    txt_t.set_text(f"t={nt * dt:.2f}s")
    txt_th.set_text(f"abs theta max={np.max(np.abs(np.degrees(tht))):.2f}°")
    txt_F.set_text(f"abs F control max={np.max(np.abs(F_hist[:, 0])):.1f}N")

    t2 = time()
    delay = max(1e-12, dt - (t2 - t1))
    if nt % 10 == 0:
        plt.pause(delay)

plt.close('all')
# %%
t_vec = np.arange(Nt) * dt
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
ctrl = 'PID'
fig.suptitle(f"{ctrl} — Inverted Pendulum", fontsize=13)

axes[0].plot(t_vec, np.degrees(tht));          axes[0].set_ylabel("theta [deg]")
axes[1].plot(t_vec, x, 'tab:orange');          axes[1].set_ylabel("x [m]")
axes[2].plot(t_vec, F_hist[:,0], 'tab:red');   axes[2].set_ylabel("F control [N]")
axes[3].plot(t_vec, F_hist[:,1], 'tab:green'); axes[3].set_ylabel("Disturbance [N]")

for ax in axes: ax.grid(alpha=.3); ax.axhline(0, color='k', lw=.5, ls='--')
plt.tight_layout()
plt.show()
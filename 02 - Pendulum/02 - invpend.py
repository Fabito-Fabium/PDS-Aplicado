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

tht0 = 1e-9
tht = np.zeros(Nt)
tht[:3] = tht0

x = np.zeros(Nt)
x0 = 0

e = np.zeros(Nt)
# %% forward
a0 = (M + m)/ dt ** 2 + k / dt
a1 = (2 * (M + m)/ dt**2 + k / dt ) / a0
a2 = - ((M+m) / dt**2) / a0
a3 = m * l / a0
a4 = - m * l / a0

b0 = m * l / dt**2 + c/dt
b1 = (2 * m * l / dt**2 + c / dt) / b0
b2 = -(m * l / dt**2) / b0
b3 = m * g / b0
b4 = m / b0

def fd_step_x(x_1, x_2, theta_1, D1theta_1, D2theta_1, F=0):
    return a1*x_1 + a2*x_2 + a3*D2theta_1*np.cos(theta_1) + a4*(D1theta_1**2)*np.sin(theta_1) + F/a0

def fd_step_theta(theta_1, theta_2, D2x_1):
    return b1*theta_1 + b2*theta_2 + b3*np.sin(theta_1) + b4*(D2x_1)*np.cos(theta_1)
# %%
fig, ax = plt.subplots()
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


P = 15
I = 25
D = 1
E = 0

for nt in range(3, Nt):
    t1 = t2
    thtd = (tht[nt - 1] - tht[nt - 3]) / (2 * dt)
    thtdd = (tht[nt - 1] - 2 * tht[nt - 2] + tht[nt - 3]) / (dt ** 2)
    set_point = 0

    e[nt] = (set_point - tht[nt - 1])
    E += e[nt] * dt
    de = (e[nt] - e[nt - 1]) / (dt)
    F = P * e[nt] + I * E + D * de
    F += 10*np.random.randn()

    x[nt] = fd_step_x(x[nt - 1], x[nt - 2], tht[nt-1], thtd, thtdd, F)

    if x[nt] > xlim:
        x -= 2 * xlim
    elif x[nt] < -xlim:
        x += 2 * xlim

    xdd = (x[nt] - 2 * x[nt - 1] + x[nt - 2]) / (dt**2)
    tht[nt] = fd_step_theta(tht[nt - 1], tht[nt - 2], xdd)

    xx, yy = coords(x[nt], tht[nt])
    line[0].set_data([xx[0]], [yy[0]])
    line[1].set_data(xx, yy)

    ax.set_title(f"t = {nt * dt:.2f}, e = {e[nt]}")
    t2 = time()
    delay = max(1e-12, dt - (t2 - t1))
    if nt % 10 == 0:
        plt.pause(delay)

plt.close('all')
# %%
t = np.arange(Nt) * dt
plt.figure()
plt.plot(t, tht/(2*np.pi))
plt.show()
# %%

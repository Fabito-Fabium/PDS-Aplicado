import matplotlib
matplotlib.use('TkAgg')
# %%
import numpy as np
import matplotlib.pyplot as plt
# %%
### ml*(D^2 theta) + cl(D theta) + mg sin(theta) = 0
l = .5                  # Length [m]
m = 1                   # Mass [Kg]
c = .02                 # Friction [N.s/rad] (Perda no eixo)
g = 9.81               # Gravity [m/s^2]

Lt = 40                 # Simulation time [s]
dt = 0.05               # Discretization [s]
Nt = round(Lt/dt)

thet0 = np.pi - 1e-9
thet1 = np.pi - 1e-1


tht = np.zeros(Nt)
tht[0] = thet0
tht[1] = thet1
# %% forward
dem = 1 + (c*dt/(2*m))

a1 = 2/dem
a2 = - (1 - c*dt/(m*2))/dem
a3 = lambda x: -((g*dt**2)/(dem*l)) * np.sin(x)

def fd_step(theta_1, theta_2):
    return a1*theta_1 + a2*theta_2 + a3(theta_1)
# %%
fig, ax = plt.subplots()
ax.set_xlim(-1, 1)
ax.set_ylim(-1,1)
line = ax.plot([0, l*np.sin(thet0)], [0, -l*np.cos(thet0)])


from time import time
t2 = time()
for nt in range(2, Nt):
    t1 = t2
    tht[nt] = fd_step(tht[nt-1], tht[nt-2])
    line[0].set_data([0, l*np.sin(tht[nt])], [0, -l*np.cos(tht[nt])])
    ax.set_title(f"t = {nt * dt:.2f}")
    t2 = time()
    delay = max(1e-12, dt - (t2 - t1))
    plt.pause(delay)

plt.close('all')
# %%
t = np.arange(Nt) * dt
plt.figure()
plt.plot(t, tht/(2*np.pi))
plt.show()
# %%

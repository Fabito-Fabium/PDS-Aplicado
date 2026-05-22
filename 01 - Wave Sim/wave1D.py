# %%
import matplotlib
matplotlib.use('TkAgg')
# %%
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt
import sounddevice as sd
# %%
# Time marching
# %%
Lx = 12         # tamanho da sala [m]
Lt = 0.1        # tempo de simulação total [s]

c = 340         # velocidade do som no ar [m/s]

Dx = 50e-3      # [m]
Dt = 50e-6      # [s]
fc = 440        # [Hz]
bw = .5

CFL = (c)*(Dt)/(Dx)
print(f'Courant Number: {CFL}')

if CFL > 1:
    print('Warning: Courant number > 1')

Nx = round(Lx / Dx)
Nt = round(Lt / Dt)

print(f'Nx: {Nx}, Nt: {Nt}')
# %%
# Deslocamentos no tempo
u_0 = np.zeros(Nx)
u_1 = np.zeros(Nx)
u_2 = np.zeros(Nx)
t0 = 10e-3

x = (np.arange(Nx)) * Dx
t = (np.arange(Nt)) * Dt

S = ss.gausspulse(t - t0, fc, bw)
# plt.plot(S)

# laplaciano no tempo
def lap(u):
    tmp = -2 * u.copy()
    tmp[:-1] += u[1:]
    tmp[1:] += u[:-1]
    return tmp

fig, ax = plt.subplots()
lines = ax.plot(x, u_0)
ax.set_ylim(-10, 10)

for nt in range(Nt):
    u_1, u_2 = u_0, u_1
    u_0 = 2*u_1 - u_2 + (CFL**2) * lap(u_1)
    u_0[0] += S[nt]

    if nt % 5 == 0:
        lines[0].set_ydata(u_0)
        ax.set_title(f't = {nt}')
        plt.pause(1e-5)

plt.close('all')
# %%
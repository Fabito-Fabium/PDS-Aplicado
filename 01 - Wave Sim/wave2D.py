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
Ly = 8
Lt = 0.1        # tempo de simulação total [s]

ca = 340         # velocidade do som no ar [m/s]
cw = 680        # velocidade do som na agua [m/s]
beta = 100

Dx = 50e-3      # discretização em x [m]
Dy = 50e-3      # discretização em y [m]
Dt = 20e-6      # [s]
fc = 440        # [Hz]
bw = .5

Nx = round(Lx / Dx)
Ny = round(Ly / Dy)
Nt = round(Lt / Dt)

print(f'Nx: {Nx}, Ny: {Ny},, Nt: {Nt}')


c = np.zeros((Ny, Nx))
c[:, :Ny//2] = ca
c[:, Ny//2:] = ca

CFL = (np.max(c)*(Dt)*(1/Dx + 1/Dy))
print(f'Courant Number: {CFL}')

if CFL > 2:
    print('Warning: Courant number > 2')


c2 = (c * Dt/Dx )**2
# %%
# Deslocamentos no tempo
u_0 = np.zeros((Ny, Nx))
u_1 = np.zeros_like(u_0)
u_2 = np.zeros_like(u_0)

# x = (np.arange(Nx)) * Dx
t = (np.arange(Nt)) * Dt

t0 = 10e-3
S = ss.gausspulse(t - t0, fc, bw)
# plt.plot(S)

# laplaciano no tempo
def lap(u):
    tmp = -4 * u.copy()

    # with respect to y
    tmp[:-1, :] += u[1:, :]
    tmp[1:, :] += u[:-1, :]

    # with respect to x
    tmp[:, :-1] += u[:, 1:]
    tmp[:, 1:] += u[:, :-1]
    return tmp

vmm = 1

fig, ax = plt.subplots()
im = plt.imshow(u_0, extent=[0, Lx, Ly, 0], aspect='auto', vmin=-vmm, vmax=vmm, cmap='bwr')
# plt.axvline(x=Ly//2, color='r', linestyle='--')



A = (1 + beta*Dt)
B = (2 + beta*Dt)

A_ = (1 + Dt*beta/2)
B_ = (Dt*beta/2 - 1)

for nt in range(Nt):
    u_1, u_2 = u_0, u_1
    # without damping
    # u_0 = 2. * u_1 - u_2 + c2*lap(u_1)

    # with damping
    # u_0 = (B*u_1 - u_2 + (c2) * lap(u_1))/A
    u_0 = (2*u_1 + B_*u_2 + (c2) * lap(u_1))/A_

    u_0[1, 1] += S[nt]
    if not nt % 10:
        im.set_data(u_0)

        E = np.sum(u_0 ** 2)
        ax.set_title(f'Nt = {nt}, E = {E:.2f}')
        plt.pause(1e-5)

plt.close('all')
# %%
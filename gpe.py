from scipy.fftpack import fft2, ifft2
import numpy as np
import external as ext
import warnings
warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")

time_steps = 100000
every = 500
i1 = 50000
i2 = time_steps
lengthwindow = i2-i1
import matplotlib.pyplot as pl

class gpe:
    def __init__(self, Kc, Kd, Kc2, rc, rd, uc, ud, sigma, z,
                 L, N, dx, dkx, x, kx, hatpsi, dt, psi_x=0):
# =============================================================================
#       Initialitze
# =============================================================================
        self.x = x
        self.kx =kx
        self.X, self.Y = np.meshgrid(self.x,self.x)
        self.KX, self.KY = np.meshgrid(self.kx, self.kx)
        self.L = L
        self.N = N
        self.dx = dx
        self.dkx = dkx
        self.hatpsi = hatpsi
# =============================================================================
#       Time
# =============================================================================
        self.dt = dt
        self.t = ext.time(self.dt, time_steps, i1, i2, every)
# =============================================================================
#       Params
# =============================================================================
        self.Kc = Kc
        self.Kd = Kd
        self.Kc2 = Kc2
        self.rc = rc
        self.rd = rd
        self.uc = uc
        self.ud = ud
        self.sigma = sigma
        self.z = z
# =============================================================================
# Initialize ψ
# =============================================================================
        self.psi_x = psi_x
        self.psi_x = np.ones((self.N, self.N))
        self.psi_x /= self.hatpsi
        self.psi_mod_k = fft2(self.psi_mod_x)

# =============================================================================
# Vortices
# =============================================================================
        '''
        self.initcond = np.full((N,N),np.sqrt(n_s))
        self.initcond[int(N/2),int(N/4)] = 0
        self.initcond[int(N/2),int(3*N/4)] = 0
        rot = []
        for i in range(N):
            for j in range(N):
                if i <= int(N/2):
                    rot.append(np.exp(-1*1j*math.atan2(x[i], y[j])))
                elif i>int(N/2):
                    rot.append(np.exp(1*1j*math.atan2(x[i], y[j])))
        self.psi_x = np.array(rot).reshape(N,N) * self.initcond
        density = (self.psi_x * np.conjugate(self.psi_x)).real
        fig,ax = pl.subplots(1,1, figsize=(8,8))
        c = ax.pcolormesh(X, Y, density, cmap='viridis')
        ax.set_title('Density')
        ax.axis([x.min(), x.max(), y.min(), y.max()])
        fig.colorbar(c, ax=ax)
        pl.show()
        '''
        '''
        self.psi_x = np.ones((N,N))
        self.psi_mod_k = fft2(self.psi_mod_x)
        print(self.psi_mod_x[5,5])
        print(ifft2(fft2(self.psi_mod_x))[5,5])

        fig,ax = pl.subplots(1,1, figsize=(8,8))
        c = ax.pcolormesh(self.KX, self.KY, np.abs(self.psi_k), cmap='viridis')
        ax.set_title('FT')
        ax.axis([kx.min(), kx.max(), ky.min(), ky.max()])
        fig.colorbar(c, ax=ax)
        pl.show()
        
        self.psi_mod_x = ifft2(self.psi_mod_k)
        fig,ax = pl.subplots(1,1, figsize=(8,8))
        c = ax.pcolormesh(self.X, self.Y, np.abs(self.psi_x), cmap='viridis')
        ax.set_title('IFFT')
        ax.axis([kx.min(), kx.max(), ky.min(), ky.max()])
        fig.colorbar(c, ax=ax)
        pl.show()
        '''
# =============================================================================
# Discrete Fourier pairs
# =============================================================================
    def _set_fourier_psi_x(self, psi_x):
        self.psi_mod_x = psi_x * np.exp(-1j * self.KX[0,0] * self.X - 1j * self.KY[0,0] * self.Y) * self.dx * self.dx / (2 * np.pi)

    def _get_psi_x(self):
        return self.psi_mod_x * np.exp(1j * self.KX[0,0] * self.X + 1j * self.KY[0,0] * self.Y) * 2 * np.pi / (self.dx * self.dx)

    def _set_fourier_psi_k(self, psi_k):
        self.psi_mod_k = psi_k * np.exp(1j * self.X[0,0] * self.dkx * np.arange(self.N) + 1j * self.Y[0,0] * self.dkx * np.arange(self.N))

    def _get_psi_k(self):
        return self.psi_mod_k * np.exp(-1j * self.X[0,0] * self.dkx * np.arange(self.N) - 1j * self.Y[0,0] * self.dkx * np.arange(self.N))

    psi_x = property(_get_psi_x, _set_fourier_psi_x)
    psi_k = property(_get_psi_k, _set_fourier_psi_k)
# =============================================================================
# Definition of the split steps
# =============================================================================
    def prefactor_x(self, wave_fn):
        n_red = wave_fn * np.conjugate(wave_fn) - 1/(2*self.dx**2)
        return np.exp(-1j*0.5*self.dt*((self.rc + 1j*self.rd) + (self.uc - 1j*self.ud)*n_red)/self.z)

    def prefactor_k(self):
        return np.exp(-1j*self.dt*((self.KX**2 + self.KY**2)*(self.Kc - 1j*self.Kd)-(self.KX**4 + self.KY**4)*self.Kc2)/self.z)

# =============================================================================
# Time evolution and Phase unwinding
# =============================================================================
    def time_evolution(self, realisation):
        #v = np.zeros(len(self.t))
        #av = np.zeros(len(self.t))
        #fuckoff = np.zeros(len(self.t))
        #n00 = np.zeros(len(self.t))
        cor_psi = np.zeros((len(t), int(N/2)), dtype=complex)
        cor_psi_row = np.zeros(int(self.N/2), dtype=complex)
        d1 = np.zeros((len(self.t), int(self.N/2)))
        d1_row = np.zeros(int(self.N/2))
        d2 = np.zeros((len(self.t), int(self.N/2)))
        d2_row = np.zeros(int(self.N/2))
        for i in range(time_steps+1):
            self.psi_x *= self.prefactor_x(self.psi_x)
            self.psi_mod_k = fft2(self.psi_mod_x)
            self.psi_k *= self.prefactor_k()
            self.psi_mod_x = ifft2(self.psi_mod_k)
            self.psi_x *= self.prefactor_x(self.psi_x)
            self.psi_x += np.sqrt(self.dt) * np.sqrt(self.sigma) * ext.noise((self.N,self.N)) / self.z
            if i>=i1 and i<=i2 and i%every==0:
                print(i)
                n = np.abs(self.psi_x * np.conjugate(self.psi_x)) - 1/(2*self.dx**2)
                #fuckoff[(i-i1)//every] = np.sum(n)/self.N**2
                #n00[(i-i1)//every] = n[int(self.N/2), int(self.N/2)]
                '''
                if i == i1:
                    ref_cor_psi = np.conjugate(self.psi_x[int(self.N/2), int(self.N/2)])
                    ref_d2 = np.sqrt(n[int(self.N/2), int(self.N/2)])
                for j in range(int(self.N/2)):
                    if j == 0:
                        cor_psi_row[j] = ref_cor_psi * self.psi_x[int(self.N/2), int(self.N/2)]
                        d2_row[j] = ref_d2 * np.sqrt(n[int(self.N/2), int(self.N/2)])
                        d1_row[j] = n[int(self.N/2), int(self.N/2)]
                    else:
                        cor_psi_row[j] = ref_cor_psi * (self.psi_x[int(self.N/2), int(self.N/2)+j] + self.psi_x[int(self.N/2), int(self.N/2)-j] + 
                                                        self.psi_x[int(self.N/2)+j, int(self.N/2)] + self.psi_x[int(self.N/2)-j, int(self.N/2)]) / 4
                        d2_row[j] = ref_d2 * (np.sqrt(n[int(self.N/2), int(self.N/2)+j]) + np.sqrt(n[int(self.N/2), int(self.N/2)-j]) 
                                              + np.sqrt(n[int(self.N/2)+j, int(self.N/2)]) + np.sqrt(n[int(self.N/2)-j, int(self.N/2)])) / 4
                        d1_row[j] = (n[int(self.N/2), int(self.N/2)+j] + n[int(self.N/2), int(self.N/2)-j] +
                                     n[int(self.N/2)+j, int(self.N/2)] + n[int(self.N/2)-j, int(self.N/2)]) / 4
                cor_psi[(i-i1)//every] = cor_psi_row
                d2[(i-i1)//every] = d2_row
                d1[(i-i1)//every] = d1_row
                '''
                '''
                count_p = 0
                count_n = 0
                theta = np.angle(self.psi_x)
                grad = np.gradient(theta, self.dx)
                for k in range(1, len(self.x)-1):
                    for l in range(1, len(self.x)-1):
                        loop = self.dx * np.sum(grad[0][k+1, l-1]*self.Y[k+1, l-1] + grad[1][k+1, l-1]*self.X[k+1, l-1]
                                                + grad[0][k+1, l]*self.Y[k+1, l] + grad[1][k+1, l]*self.X[k+1, l]
                                                + grad[0][k+1, l+1]*self.Y[k+1, l+1] + grad[1][k+1, l+1]*self.X[k+1, l+1]
                                                + grad[0][k-1, l-1]*self.Y[k-1, l-1] + grad[1][k-1, l-1]*self.X[k-1, l-1]
                                                + grad[0][k-1, l]*self.Y[k-1, l] + grad[1][k-1, l]*self.X[k-1, l]
                                                + grad[0][k-1, l+1]*self.Y[k-1, l+1] + grad[1][k-1, l+1]*self.X[k-1, l+1]
                                                + grad[0][k, l-1]*self.Y[k, l-1] + grad[1][k, l-1]*self.X[k, l-1]
                                                + grad[0][k, l+1]*self.Y[k, l+1] + grad[1][k, l+1]*self.X[k, l+1])
                        if loop >= 2 * np.pi:
                            count_p += 1
                        elif loop <= -2 * np.pi:
                            count_n += 1
                v[(i-i1)//every] = count_p
                av[(i-i1)//every] = count_n
                '''
        return cor_psi, d2, d1
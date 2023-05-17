import numpy as np
import matplotlib.pyplot as plt

# paramètres matériaux et milieux
rho_s = 3.78e-4             # 1.24*10**(-5) pour le nylon (wikipedia)
L = 1                     # longueur de la corde
T0 = 12                   # initial tensionning axial force (supposition 35)   ### 12 kg https://www.tabs4acoustic.com/calculateur-tension-cordes.html
# E = 210e9  # module Young acier
# S = 0.001

rho = 1.2
c = 342

# paramètres relatifs aux modes propres
N = 5                    # nombre de modes
n = np.arange(1, N+1, 1)

m_n = rho_s*L/2*np.ones(N)                    # masses modales
omega_n = np.sqrt(T0/(rho_s))*((n*np.pi)/L)   # pulsations propres
xi_n = 0.005*np.ones(N)                         # amortissement (entre 0 et 1 pour un mode oscillant)
k_n = m_n*omega_n**2

# discrétisation spatiale de la corde
Nx = 100
x = np.linspace(0, L, Nx)

#position de la source
xs = L/4
F0 = 0.1
i = 0
if xs<L:
    while x[i]<=xs:
        indS = i
        i+=1
else: print('ATTENTION!!!!! source hors de la corde')
    
# modes propres de la corde
phi_n = np.array([np.sin((ni*x*np.pi)/L) for ni in n])

# 'fct_mat' revoit le vecteur u_p à partir du vecteur u pour la fonction rk4

def fct_mat(tn, vec_qn): #, fn=np.zeros(N)
    
    A11 = np.eye(N)-np.eye(N)
    A12 = np.eye(N)
    A21 = -omega_n**2*np.eye(N)
    A22 = -2*omega_n*xi_n*np.eye(N)
    
    A_up = np.hstack((A11, A12))
    A_down = np.hstack((A21, A22))
    A = np.vstack((A_up, A_down))
    
    #vec_fn = np.concatenate(np.zeros(N), fn*np.ones(N)/m_n) 
    vec_fn = np.zeros(N)
    sys_vec_fn = np.concatenate((np.zeros(N),vec_fn)) #.reshape((N*2,1))
    
    vec_qn1 = A@vec_qn + sys_vec_fn
    #qn_p, qn_pp = vec_qn1[:N], vec_qn1[N:]
    #qn = (-qn_pp - 2*omega_n*xi_n*qn_p + vec_fn/m_n)/(omega_n**2)
    return vec_qn1

tn = np.linspace(0, 10, 441000)  # choisi pour Fs = 44100
q0 = [F0*phi_n[i][indS]/(m_n[i]*omega_n[i]**2) for i in range(N)]
fn1 = [F0*phi_n[i][indS]/(m_n[i]) for i in range(N)]

vect = fct_mat(tn, np.concatenate((q0, np.zeros(N))))

#%%

# rk4
def rk4s(fp, f0, x):
    """Solve ODE with 2nd order Runge-Kutta method."""
    h = np.abs(x[0]-x[1])
    f = np.zeros((len(f0), len(x)))
    
    f[:,0] = f0
    for xi in range(len(x)-1):
        k1 = fp(x[xi], f[:, xi])
        k2 = fp(x[xi]+h/2, f[:, xi]+h/2*k1)
        k3 = fp(x[xi]+h/2, f[:, xi]+h/2*k2)
        k4 = fp(x[xi]+h, f[:, xi]+h*k3)
        f[:, xi+1] = f[:, xi] + h/6*(k1+2*k2+2*k3+k4)
    return f

#tn = np.linspace(0, 3, 100)
vect_qn_qnp = rk4s(fct_mat, np.concatenate((q0, np.zeros(N))), tn)
qn_rk4, qn_p_rk4 = vect_qn_qnp[:N], vect_qn_qnp[N:]

#%%
'''
plt.figure()
for ind in n-1:
    plt.plot(tn, qn_rk4[ind], '*', label='$q_{}$'.format(ind+1))
plt.legend()
plt.xlabel('Temps (s)')
plt.ylabel('Poids $q_n$')
plt.title('Influence des modes sur le mouvement au cours du temps')
# plt.show()
'''

#%%

# on veut une fonction qui revoit les valeurs de w à partir des valeurs de qn

# exemple : tracer la corde pour t_0
qn_rk4.T # les lignes sont les temps discrets et les colonnes sont les poids des modes
qn_rk4.T[0] # les poids des modes à t=0
phi_n[0]*qn_rk4.T[0][0] #w_0 : déplacement de la corde pour le mode 1 pondéré par le poids à t=0
phi_n[1]*qn_rk4.T[0][1] #w_1

list_modes_ponderes = []
w_t0_rk4 = np.zeros(Nx)
for ind in range(N):
    list_modes_ponderes.append(phi_n[ind]*qn_rk4.T[0][ind])
for ind in range(N):
    w_t0_rk4 += list_modes_ponderes[ind]
'''
plt.figure()
plt.plot(x, w_t0_rk4)
plt.title('Corde à t=0')
plt.xlabel('Discrétisation spatiale')
# plt.show()
'''

#%%

# On généralise ce code en une fonction pour pouvoir représenter la corde pour tous les temps

def fct_w_rk4(tn, qn):
    list_w = []
    for indt in range (len(tn)):
        list_modes_pond_ti = []
        w_ti_rk4 = np.zeros(Nx)
        for ind in range (N):
            list_modes_pond_ti.append(phi_n[ind]*qn_rk4.T[indt][ind])
        for ind in range (N):
            w_ti_rk4 += list_modes_pond_ti[ind]
        list_w.append(w_ti_rk4)
    w_rk4 = np.array(list_w)
    return w_rk4

w_rk4 = fct_w_rk4(tn, qn_rk4)

'''
plt.figure()
for ind in np.arange(0, 100, 2):
    plt.plot(x, w_rk4[ind], label='$t_{}$'.format(ind))
plt.title('Corde')
plt.xlabel('Discrétisation spatiale')
plt.legend()
# plt.show()
'''

# %%
def diffO4(s, h):
    """ 4th order derivation using slices. """
    ds = np.zeros(s.shape)
    ds[0] = (-11*s[0] + 18*s[1] - 9*s[2] + 2*s[3])/(6*h)
    ds[1] = (-2*s[0] - 3*s[1] + 6*s[2] - s[3])/(6*h)
    ds[-1] = -(-11*s[-1] + 18*s[-2] - 9*s[-3] + 2*s[-4])/(6*h)
    ds[-2] = -(-2*s[-1] - 3*s[-2] + 6*s[-3] - s[-4])/(6*h)
    ds[2:-2] = (s[:-4] - 8*s[1:-3] + 8*s[3:-1] - 1*s[4:])/(12*h)
    return ds

#position de la source
xm = 0.42
ym = 20

dx = x[1]-x[0]

def wXfix(xi,wRK4=w_rk4):
    ixm = np.where(abs(x-xi)==min(abs(x-xi)))
    wt = wRK4[:,ixm][:,0,0]
    return wt

def Q(xi):  # A VERIFIER !!!
    return diffO4(wXfix(xi), dx)
# print(Q(0.5))

def r(r1,r2):
    return np.sqrt((r1[0]-r2[0])**2+(r1[1]-r2[1])**2)

def Pmono(xi,rm):
    ri = (xi,0)
    return sum(1j*om*rho*Q(xi)*np.exp(-1j*om/c*(r(rm,ri))/(4*np.pi*r(rm,ri))) for om in omega_n)

p = sum([Pmono(xi,(xm,ym)) for xi in x])

plt.figure('pression')
plt.plot(tn, np.real(p))
plt.xlabel('temps (s)')
plt.ylabel('Re(p)')

from scipy.io.wavfile import write
rate = int(len(tn)/(tn[-1]-tn[0]))
print(rate)
pAbs = abs(p)
# scaled = np.int16(data / np.max(np.abs(data)) * 32767)
pwav = (pAbs*2**15).astype(np.int16)
write('testY20F01N5.wav', rate, pwav)

plt.show()


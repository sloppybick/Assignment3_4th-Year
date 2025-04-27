import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def f(t,y):
    theta,omega=y
    dtheta_dt=omega
    domega_dt=-(32.17/2)*np.sin(theta)
    return [dtheta_dt,domega_dt]

theta0=np.pi/6
omega0=0
t_span=(0,2)
t_eval=np.arange(0,2.1,0.1)
solution=solve_ivp(f,t_span,(theta0,omega0),t_eval=t_eval,method='RK45')

t=solution.t
theta=solution.y[0]
omega=solution.y[1]

for ti,thetai,omegai in zip(t,theta,omega):
    print(f't={ti:.1f}s, theta={thetai:0.5f}rad, omega={omegai:0.5f}rad/s')

plt.figure(figsize=(12,6))

plt.subplot(1,3,1)
plt.plot(t,theta,'c--',label=r'$\theta(t)$')
plt.xlabel('t(s)')
plt.ylabel(r'$\theta$(radians)')
plt.title('Pendulum motion: Angle vs Time')
plt.legend()

plt.subplot(1,3,2)
plt.plot(t,omega,'m--',label=r'$\omega(t)$')
plt.xlabel('t(s)')
plt.ylabel(r'$\omega$(rad/s)')
plt.title('Pendulum motion:  Angular Velocity vs Time')
plt.legend()

plt.subplot(1,3,3)
plt.plot(theta,omega,'b--')
plt.xlabel('theta')
plt.ylabel('omega')
plt.title('Phase plot :theta vs omega')
plt.tight_layout()
plt.show()

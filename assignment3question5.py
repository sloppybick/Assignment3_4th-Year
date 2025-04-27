import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def system_of_odes(t,y):
    x1=y[0]
    x1_prime=y[1]
    x1_double_prime=y[2]
    x2=y[3]
    x2_prime=y[4]
    x2_double_prime=y[5]

    dx1_dt=x1_prime
    dx1_prime_dt=x1_double_prime
    dx1_double_prime_dt=-2*(x2_prime)**2+x2

    dx2_dt=x2_prime
    dx2_prime_dt=x2_double_prime
    dx2_double_prime_dt=-(x1_double_prime)**3+x2_prime+x1+np.sin(t)

    return [dx1_dt,dx1_prime_dt,dx1_double_prime_dt,dx2_dt,dx2_prime_dt,dx2_double_prime_dt]

tspan=(0,100)
teval=np.linspace(0,100,1000)
initial_condtion=[0,0,0,0,0,0]

solution=solve_ivp(system_of_odes,tspan,initial_condtion,t_eval=teval,method='RK45',rtol=1e-08,atol=1e-08)

t=solution.t
x1=solution.y[0]
x1_prime=solution.y[1]
x1_double_prime=solution.y[2]
x2=solution.y[3]
x2_prime=solution.y[4]
x2_double_prime=solution.y[5]


for ti,xi1,xi2 in zip(t,x1,x2):
    print(f"t={ti:.5f},x1={xi1:0.5f},x2={xi2:0.5f}")

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.plot(t,x1,'b-',label='x1(t)')
plt.plot(t,x1_prime,'r',label="x1'(t)")
plt.plot(t,x1_double_prime,'g--',label="x1''(t)")
plt.xlabel('Time(t)')
plt.ylabel('Value')
plt.title("x1(t) and its Derivatives")
plt.legend()
plt.grid(True)

plt.subplot(2,2,2)
plt.plot(t,x2,'b--',label='x2(t)')
plt.plot(t,x2_prime,'r',label="x2'(t)")
plt.plot(t,x2_prime,'g--',label="x2''(t)")
plt.xlabel('Time(t)')
plt.ylabel('Value')
plt.title("x2(t) and its Derivatives")
plt.legend()
plt.grid(True)

plt.subplot(2,2,3)
plt.plot(x1,x2,'c--')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Phase plot x1 vs x2')

plt.subplot(2,2,4)
plt.plot(x1_prime,x2_prime,'m--')
plt.xlabel('x1\'')
plt.ylabel('x2\'')
plt.title('Phase plot x1\' vs x2\'')

plt.tight_layout()
plt.show()
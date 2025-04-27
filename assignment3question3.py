import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def competition_model(t, z):
    x, y = z
    dxdt = x * (2 - 0.4 * x - 0.3 * y)
    dydt = y * (1 - 0.1 * y - 0.3 * x)
    return [dxdt, dydt]


t_span = (0, 1000)
t_eval = np.linspace(0, 1000, 10000)


initial_conditions = [
    [1.5, 3.5], 
    [1, 1],     
    [2, 7],          
    [4.5, 0.5]   
]


colors = ['red', 'blue', 'green', 'purple']
case_labels = ['a) x(0)=1.5, y(0)=3.5', 
               'b) x(0)=1, y(0)=1', 
               'c) x(0)=2, y(0)=7', 
               'd) x(0)=4.5, y(0)=0.5']


solutions = []
for z0 in initial_conditions:
    sol = solve_ivp(competition_model, t_span, z0, t_eval=t_eval, method='RK45')
    solutions.append(sol)


plt.figure(figsize=(12, 8))


plt.subplot(3,1, 1)
for sol, color, label in zip(solutions, colors, case_labels):
    plt.plot(sol.t, sol.y[0], color=color, label=label)
plt.xlabel('Time (years)')
plt.ylabel('Population x (thousands)')
plt.title('Population x over Time')
plt.legend()
plt.grid(True)


plt.subplot(3,1, 2)
for sol, color, label in zip(solutions, colors, case_labels):
    plt.plot(sol.t, sol.y[1], color=color, label=label)
plt.xlabel('Time (years)')
plt.ylabel('Population y (thousands)')
plt.title('Population y over Time')
plt.legend()
plt.grid(True)


plt.subplot(3,1,3)
for sol,color,label in zip(solutions,colors,case_labels):
    plt.plot(sol.y[0],sol.y[1],color=color,label=label)
plt.xlabel('Population x (thousands)')
plt.ylabel('Population y (thousands)')
plt.title('Phase Portrait')
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
# problem conditions
demand_1 = 110; demand_2 = 100; demand_3 = 80
prob_1 = 0.2; prob_2 = 0.6; prob_3 = 0.2
price_1 = 50; price_2 = 46; price_3 = 44
price_c = 45; contract_max = 90.0

demands = [demand_1, demand_2, demand_3]
probs = [prob_1, prob_2, prob_3]
prices = [price_1, price_2, price_3]
prices_matrix = np.identity(len(prices))
np.fill_diagonal(prices_matrix, prices)
# define objective function
def objective(p):
    p_c, p1, p2, p3 = p
    stg_1 = price_c*p_c
    stg_2 = np.dot(np.dot(probs, prices_matrix), [p1, p2, p3])
    # prob_1*price_1*p[1]+prob_2*price_2*p[2]+prob_3*price_3*p[3]
    obj = 24*7*(stg_1+stg_2)
    return obj
# define constraints
def constraint1(p):
    return p[0]+p[1]-demand_1
def constraint2(p):
    return p[0]+p[2]-demand_2
def constraint3(p):
    return p[0]+p[3]-demand_3
'''
def constraint2(x):
    sum_eq = 40.0
    for i in range(4):
        sum_eq = sum_eq - x[i]**2
    return sum_eq
'''
# initial guesses
n = 4
p0 = np.zeros(n)
for i in range(len(p0)):
    p0[i] = 80.0
# p0[0] = 60.0; p0[1] = 50.0; p0[2] = 50.0; p0[3] = 10.0

# show initial objective
print('Initial Objective: ' + str(objective(p0)))

# set constraints/bounds
b_c = (0.0, contract_max); b_pos = (0.0, None); b_open = (None, None)
con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'ineq', 'fun': constraint2}
con3 = {'type': 'ineq', 'fun': constraint3}
# con2 = {'type': 'eq', 'fun': constraint2}
cons = [con1,con2,con3]
bnds = (b_c, b_pos, b_pos, b_pos)
# optimize
solution = minimize(objective,p0,method='SLSQP',bounds=bnds, constraints=cons)
p = solution.x
for i in range(len(p)):
    p[i]=round(p[i],0)

# print solution
print('Solution Node-formulation')
print('Final Objective: ' + str(objective(p)))
print('p_c = ' + str(p[0]))
print('p1 = ' + str(p[1]))
print('p2 = ' + str(p[2]))
print('p3 = ' + str(p[3]))

## calculate VSS
# stochastic solution
z_s = objective(p)
# VSS problem
# stage 1 optimal value for the first stage variable p_c
def objective_avg(p):
    avg_price = np.dot(probs, prices)    # sigma_i(prob_i*price_i)
    stg_1 = price_c*p[0]
    stg_2 = avg_price*p[1]
    obj = 24*7*(stg_1+stg_2)
    return obj
# define constraints
avg_demand = np.dot(probs, demands)
def constraint1(p):
    return p[0]+p[1]-avg_demand
# initial guesses
n = 2
p0 = np.zeros(n)
for i in range(len(p0)):
    p0[i] = 80.0
# set constraints/bounds
con1 = {'type': 'ineq', 'fun': constraint1}
bnds = (b_c, b_pos)
# optimize
cons = [con1]
solution = minimize(objective_avg,p0,method='SLSQP',bounds=bnds, constraints=cons)
p = solution.x

# print solution
print('Solution Deterministic Problem 1')
print('p_cd = ' + str(p[0]))
p_cd = p[0]

# stage 2
def objective_VSS(p):
    p1, p2, p3 = p
    stg_1 = price_c*p_cd
    stg_2 = np.dot(np.dot(probs, prices_matrix), [p1, p2, p3])
    obj = 24 * 7 * (stg_1 + stg_2)
    return obj
# define constraints
def constraint1(p):
    return p_cd + p[0] - demand_1
def constraint2(p):
    return p_cd + p[1] - demand_2
def constraint3(p):
    return p_cd + p[2] - demand_3
# initial guesses
n = 3
p0 = np.zeros(n)
for i in range(len(p0)):
    p0[i] = 80.0
# set constraints/bounds
con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'ineq', 'fun': constraint2}
con3 = {'type': 'ineq', 'fun': constraint3}
cons = [con1,con2,con3]
bnds = (b_pos, b_pos, b_pos)
# optimize
solution = minimize(objective_VSS,p0,method='SLSQP',bounds=bnds, constraints=cons)
p = solution.x
for i in range(len(p)):
    p[i]=round(p[i],0)

# print solution
print('Solution Deterministic Problem 2')
print('Final Objective: ' + str(objective_VSS(p)))
print('p1 = ' + str(p[0]))
print('p2 = ' + str(p[1]))
print('p3 = ' + str(p[2]))

# deterministic solution
z_d = objective_VSS(p)

# calculate VSS value
VSS_min = z_d - z_s
print('VSS value: ' + str(VSS_min))

## calculate CVaR
def objective_CVaR(p, a=0, b=0):
    p_c, p1, p2, p3, s1, s2, s3, yita = p
    stg_1 = price_c*p_c
    stg_2 = np.dot(np.dot(probs, prices_matrix), [p1, p2, p3])
    profit = 0-24*7*(stg_1 + stg_2)
    max_obj = (1-b)*profit+b*(yita-(1/(1-a))*np.dot(probs, [s1, s2, s3]))
    min_obj = -max_obj
    return min_obj
# define constraints
def constraint1(p):
    return p[0] + p[1] - demand_1
def constraint2(p):
    return p[0] + p[2] - demand_2
def constraint3(p):
    return p[0] + p[3] - demand_3

def constraint_s1(p):
    profit_1 = 0-24*7*np.dot([price_c,price_1],[p[0],p[1]])
    return profit_1 - p[7] + p[4]
def constraint_s2(p):
    profit_2 = 0-24*7*np.dot([price_c,price_2],[p[0],p[2]])
    return profit_2 - p[7] + p[5]
def constraint_s3(p):
    profit_3 = 0-24*7*np.dot([price_c,price_3],[p[0],p[3]])
    return profit_3 - p[7] + p[6]
# initial guesses
n = 8
p0 = np.zeros(n)
for i in range(len(p0)):
    p0[i] = 70.0

# set constraints/bounds
con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'ineq', 'fun': constraint2}
con3 = {'type': 'ineq', 'fun': constraint3}
con4 = {'type': 'ineq', 'fun': constraint_s1}
con5 = {'type': 'ineq', 'fun': constraint_s2}
con6 = {'type': 'ineq', 'fun': constraint_s3}
cons = [con1,con2,con3,con4,con5,con6]
bnds = (b_c, b_pos, b_pos, b_pos, b_pos, b_pos, b_pos, b_open)

# set alpha and beta
alpha = 0.6; beta = 1
# optimize
solution = minimize(objective_CVaR, p0, args=(alpha, beta), method='SLSQP', bounds=bnds, constraints=cons)
p = solution.x
p_c, p1, p2, p3, s1, s2, s3, yita = p

# print solution
print('Solution CVaR')
print('Final Objective: ' + str(objective_CVaR(p)))
print('alpha = '+str(alpha)+'\tbeta='+str(beta))
print('yita = '+str(yita))

def print_scenario(i):
    profit = 0-24*7*np.dot([price_c, prices[i-1]], [p_c, p[i]])
    print('scenario {0}: '.format(i)+'p{0}='.format(i)+str(round(p[i],0))
          +'\tprob.='+str(probs[i-1])
          +'\tprofit='+str(round(profit,0))
          +'\ts{0}='.format(i)+str(max(yita-profit,0)))
    return max(yita-profit,0),profit

n=4
s = np.zeros(n-1)
profits = np.zeros(n-1)
for i in range(1,n):
    s[i-1], profits[i-1] = print_scenario(i)

CVaR = yita-1/(1-alpha)*np.dot(probs,s)
Exp_profit = float(np.dot(probs, profits))
print('CVaR='+str(round(CVaR,0))+',\tExp_profit='+str(round(Exp_profit,0)))

# plot figure
# ref:
# https://stackoverflow.com/questions/49661247/plotting-cdf-for-discrete-variable-step-plot-with-alternating-lines
# https://matplotlib.org/gallery/statistics/histogram_cumulative.html
x = np.append(profits, -600000)
y = np.array([0]+probs).cumsum()
fig, ax = plt.subplots()
ax.set_facecolor('white')
# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.hlines.html
ax.hlines(y=y[1:], xmin=x[:-1], xmax=x[1:], zorder=1)
# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.vlines.html
ax.vlines(x=x[:-1], ymin=y[:-1], ymax=y[1:], zorder=1)
ax.scatter(x[:-1], y[1:], s=18, color='black', zorder=1)
ax.scatter(Exp_profit, 0, s=20, color='black', zorder=0)
ax.annotate('Expected profit', (Exp_profit, 0), xytext=(Exp_profit, 0.02))
ax.scatter(CVaR, 0, s=20, color='b', zorder=0)
ax.annotate('CVaR(alpha={0})'.format(alpha), (CVaR, 0), xytext=(CVaR, 0.08), arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
# tidy up the figure
ax.grid(False)
ax.set_xlim(-900000, x[-1])
ax.set_ylim(y[0], 1.1)

ax.set_title('cdf of scenario profits, beta={0}'.format(beta))
ax.set_xlabel('Profit(euro)')
ax.set_ylabel('Probability')

plt.show()
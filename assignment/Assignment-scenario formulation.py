import numpy as np
from scipy.optimize import minimize
# problem conditions
demand_1 = 120; demand_2 = 100; demand_3 = 80
prob_1 = 0.15; prob_2 = 0.75; prob_3 = 0.1
price_1 = 59; price_2 = 46; price_3 = 44
price_c = 45; contract_max = 90.0
# define objective function
def objective(p):
    p_c1, p_c2, p_c3, p1, p2, p3 = p
    sce_1 = np.dot([price_c,price_1],[p_c1,p1])
    sce_2 = np.dot([price_c,price_2],[p_c2,p2])
    sce_3 = np.dot([price_c,price_3],[p_c3,p3])
    obj = 24*7*np.dot([prob_1, prob_2, prob_3],[sce_1,sce_2,sce_3])
    return obj
# define constraints
def constraint1(p):
    return p[0]+p[3]-demand_1
def constraint2(p):
    return p[1]+p[4]-demand_2
def constraint3(p):
    return p[2]+p[5]-demand_3

# p_c1=p_c2=p_c3
def constraint_c12(p):
    return p[0]-p[1]
def constraint_c23(p):
    return p[1]-p[2]

# initial guesses
n = 6
p0 = np.zeros(n)
for i in range(len(p0)):
    p0[i] = 80.0
# show initial objective
print('Initial Objective: ' + str(round(objective(p0),0)))

# set constraints/bounds
b_c = (0.0, contract_max)
b_pos = (0.0, None)
con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'ineq', 'fun': constraint2}
con3 = {'type': 'ineq', 'fun': constraint3}
con_c12 = {'type': 'eq', 'fun': constraint_c12}
con_c23 = {'type': 'eq', 'fun': constraint_c23}
cons = [con1,con2,con3,con_c12,con_c23]
bnds = (b_c, b_c, b_c, b_pos, b_pos, b_pos)
# optimize
solution = minimize(objective,p0,method='SLSQP',bounds=bnds, constraints=cons)
p = solution.x
for i in range(len(p)):
    p[i]=round(p[i],0)
# show final objective
print('Final Objective: ' + str(round(objective(p),0)))

# print solution
print('Solution')
print('p_c1 = ' + str(p[0]))
print('p1 = ' + str(p[3]))
print('p_c2 = ' + str(p[1]))
print('p2 = ' + str(p[4]))
print('p_c3 = ' + str(p[2]))
print('p3 = ' + str(p[5]))

# calculate EVPI

# stochastic solution
z_s = objective(p)
# EVPI problem (with perfect infomation) -- relax the non-anticipativity constrainsts
cons = [con1,con2,con3]
# optimize
solution = minimize(objective,p0,method='SLSQP',bounds=bnds, constraints=cons)
p = solution.x
for i in range(len(p)):
    p[i]=round(p[i],0)
print('Final Objective: ' + str(round(objective(p),0)))
print('Solution given perfect information')
print('p_c1 = ' + str(p[0]))
print('p1 = ' + str(p[3]))
print('p_c2 = ' + str(p[1]))
print('p2 = ' + str(p[4]))
print('p_c3 = ' + str(p[2]))
print('p3 = ' + str(p[5]))

# perfect info. solution
z_p = objective(p)

# calculate EVPI value
EVPI_min = z_s - z_p
print('EVPI value: ' + str(EVPI_min))
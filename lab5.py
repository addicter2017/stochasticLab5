import random
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# question 1
def f(x):
    return (0.5*(x**4+x**2))
theta,error = integrate.quad(f,-1,1)
print(theta)

# question 2
def crude_monte(M):
    y = 0
    for i in range(int(M)):
        x = random.uniform(0,1)
        y = y +  0.5*((x-1)**4+(x-1)**2+x**4+x**2)
    return y/M

# question 3
def HoM_monte(M):
    count = 0
    for i in range(int(M)):
        u1 = random.uniform(0,1)
        u2 = random.uniform(0,1)
        if u2 <= 0.5*((u1-1)**4+u1**4+(u1-1)**2+u1**2):
            count += 1
    return count/M



# question 5
N = 20
sample_size = np.zeros(N)
crude_var = []
hom_var = []
crude_list = []
hom_list = []
# creating different sample sizes from 2 to 2^20
for n in range(int(N)):
    sample_size[n] = 2**(n+1)
for i in sample_size:
    for j in range(int(i)):
        x = random.uniform(0,1)
        y1 = 0.5 * ((x - 1) ** 4 + (x - 1) ** 2 + x ** 4 + x ** 2)
        y2 = random.uniform(0,1)
        crude_list.append(y1)
        if y2 <= 0.5*((x-1)**4+x**4+(x-1)**2+x**2):
            hom_list.append(1)
        else:
            hom_list.append(0)
# Split the crude list and hom list to obtain all variances with different sample sizes
for num in range(int(N)):
    crude_var.append(np.var(crude_list[:2**(num+1)]))
    hom_var.append(np.var(hom_list[:2**(num+1)]))

# Calculate root mean square error
def root(fun):
    return fun**0.5
print(list(map(root,crude_var)))
print(list(map(root, hom_var)))


# question 6
N  = 20
sample_size = np.zeros(N)
crude_differ = []
hom_differ = []
# creating different sample sizes from 2 to 2^20
for n in range(int(N)):
    sample_size[n] = 2**(n+1)
for i in sample_size:
    theta_crude = crude_monte(i)
    theta_hom = HoM_monte(i)
    crude_differ.append(abs(theta_crude-theta))
    hom_differ.append(abs(theta_hom-theta))

plt.loglog(sample_size,1/np.sqrt(sample_size))
plt.loglog(sample_size,crude_differ)
plt.loglog(sample_size,hom_differ)
plt.legend(labels = ['Slope','Crude Monte Carlo','HoM Monte Carlo'],loc = 'best')

plt.savefig("Comparison.pdf")

plt.show()

# question 7
#  Root mean squared error of Crude Monte Carlo
def crude_rmse(M,N = 10):
    sum = 0
    for i in range(int(N)):
        sum += (crude_monte(M)-theta)**2
    return (1/N*sum)**0.5
#  Root mean squared error of HoM Monte Carlo
def hom_rmse(M,N = 10):
    sum = 0
    for i in range(int(N)):
        sum += (HoM_monte(M)-theta)**2
    return (1/N*sum)**0.5

# The M is set from 1 to 1000 to plot the data
crude_rmse_list = []
hom_rmse_list = []
M_list = [x for x in range(1,1000)]
for i in M_list:
    crude_rmse_list.append(crude_rmse(i))
    hom_rmse_list.append(hom_rmse(i))
plt.plot(M_list,crude_rmse_list)
plt.plot(M_list,hom_rmse_list)
plt.xlabel('M sample sizes')
plt.ylabel('RMSE')
plt.legend(labels = ['Crude Monte Carlo rmse','HoM Monte Carlo rmse'],loc = 'best')
plt.savefig('Rmse')
plt.show()

# If N increases from 10 to 20
crude_rmse_list_20 = []
hom_rmse_list_20 = []
for i in M_list:
    crude_rmse_list_20.append(crude_rmse(i,20))
    hom_rmse_list_20.append(hom_rmse(i,20))
plt.plot(M_list,crude_rmse_list)
plt.plot(M_list,hom_rmse_list)
plt.plot(M_list,crude_rmse_list_20)
plt.plot(M_list,hom_rmse_list_20)

plt.xlabel('M sample sizes')
plt.ylabel('RMSE')
plt.legend(labels = ['Crude Monte Carlo rmse N=10','HoM Monte Carlo rmse N=10','Crude Monte Carlo rmse N=20','Crude Monte Carlo rmse N=20'],loc = 'best')
plt.savefig('Rmse_20')
plt.show()
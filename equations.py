import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import math

def line(x,a,b):
    return a*math.cos(2*math.pi+b)

x =np.random.uniform(0.,10,100)
y =3.*x+2 +np.random.uniform(0.,10,100)
popt ,pconv =curve_fit(line,x,y,)

data =np.loadtxt('CanadaTemp.csv', delimiter=',',dtype=str,skiprows=1)
# print(data)

date =data[:,1]
min_temperature=data[:,2]
avg_temperature=data[:,3]

def line_func(x,a,b):
    return a*math.cos(2*math.pi+b)

x =np.arange(len(date)).astype(float)
y =avg_temperature.astype(float)

popt,pconv =curve_fit(line,x,y)

plt.plot(x,y,'-',label = 'Original temperature')
plt.legend(loc='best')
plt.title('Temperature', fontweight='bold')
plt.xlabel('Date', fontweight='bold')
plt.ylabel('Average Temperature', fontweight='bold')
plt.show()

y1 = x* line_func(x,*popt)
plt.plot(x,y1,'-',label = 'Best Fit Curve', color='red')
plt.legend(loc='best')
plt.title('BEST FIT', fontweight='bold')
plt.xlabel('Date', fontweight='bold')
plt.ylabel('Average Temperature', fontweight='bold')
plt.show()

from scipy.stats import norm
from csv import writer
import numpy as np

#Cz.1

distribution_x = norm(loc=0, scale=200)
distribution_y = norm(loc=0, scale=200)

num_points = 5000

x = distribution_x.rvs(size=num_points)
y = distribution_y.rvs(size=num_points)
z = np.zeros(num_points)

points = zip(x,y,z)

with open('Lab01_1.xyz', 'w', encoding='utf-8', newline='\n') as csvfile:
    csvwriter = writer(csvfile)
    for p in points:
        csvwriter.writerow(p)

#Cz.2


distribution_y = norm(loc=0, scale=200)
distribution_z = norm(loc=0, scale=200)

num_points = 5000

x = np.zeros(num_points)
y = distribution_y.rvs(size=num_points)
z = distribution_y.rvs(size=num_points)

points = zip(x, y, z)

with open('Lab01_2.xyz', 'w', encoding='utf-8', newline='\n') as csvfile:
    csvwriter = writer(csvfile)
    for p in points:
        csvwriter.writerow(p)

#Cz.3
angleCount = 1000
angleMin = 0
angleMax = 2 * np.pi

angles = np.linspace(angleMin, angleMax, angleCount)

distribution_z = norm(loc=0, scale=200)

num_points = 5000

radius = 100
distribution_x = radius * np.cos(angles)
distribution_y = radius * np.sin(angles)

z = distribution_z.rvs(size=num_points)

x = np.tile(distribution_x, num_points)
y = np.tile(distribution_y, num_points)

points = zip(x, y, z)

with open('Lab01_3.xyz', 'w', encoding='utf-8', newline='\n') as csvfile:
    csvwriter = writer(csvfile)
    for p in points:
        csvwriter.writerow(p)

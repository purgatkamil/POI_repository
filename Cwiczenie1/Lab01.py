from csv import writer
import numpy as np

# Part 1
num_points = 5000

# Define range for x, y, and z coordinates
x_range = (-500, 500)
y_range = (-500, 500)
z_range = (-500, 500)

x = np.random.uniform(*x_range, size=num_points)
y = np.random.uniform(*y_range, size=num_points)
z = np.zeros(num_points)

points = zip(x, y, z)

with open('Lab01_1.xyz', 'w', encoding='utf-8', newline='\n') as csvfile:
    csvwriter = writer(csvfile)
    for p in points:
        csvwriter.writerow(p)

# Part 2
num_points = 5000

y_range = (-500, 500)
z_range = (-500, 500)

x = np.zeros(num_points)
y = np.random.uniform(*y_range, size=num_points)
z = np.random.uniform(*z_range, size=num_points)

points = zip(x, y, z)

with open('Lab01_2.xyz', 'w', encoding='utf-8', newline='\n') as csvfile:
    csvwriter = writer(csvfile)
    for p in points:
        csvwriter.writerow(p)

# Part 3
angleCount = 1000
angleMin = 0
angleMax = 2 * np.pi

angles = np.linspace(angleMin, angleMax, angleCount)

num_points = 5000

radius = 100

z_range = (-100, 100)
distribution_x = radius * np.cos(angles)
distribution_y = radius * np.sin(angles)

z = np.random.uniform(*z_range, size=num_points)

x = np.tile(distribution_x, num_points)
y = np.tile(distribution_y, num_points)

points = zip(x, y, z)

with open('Lab01_3.xyz', 'w', encoding='utf-8', newline='\n') as csvfile:
    csvwriter = writer(csvfile)
    for p in points:
        csvwriter.writerow(p)
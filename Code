# USEFUL LIBRARIES

import numpy as np
import math 
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial.distance as ssd
import itertools
from sklearn.metrics.pairwise import euclidean_distances

# Universale Constants
k      =   8.900000000 * 10**(  9)    # Colomb constant
q      =   1.602176621 * 10**(-19)  # Proton Charge 
m      =   1.672621898 * 10**(-27)  # Proton mass
Radius =   8.768000000 * 10**(-16)  # Proton Radius sould be -16
boltzman_constant = 1.38064852 * 10**(-23)
strong_force_distance = 2.5 * 10**(-15)

# TANK VARIABLES
Temperature = 100000000 # Temperature in Kelvin. 
time        = 100
delta_t     = 1*10**(-10)

## Tank Dimensions
Lx = 10 # x-length
Ly = 10 # y-length
Lz = 10 # z-lenght

FUSION = 0


N = 10000

position = np.random.uniform(-Lx, Lx, (N, 3))
velocity = np.array([1.6*np.sqrt(boltzman_constant * Temperature / m) * a for a in np.random.uniform(-Lx, Lx, (N, 3))])


start_x = position[:,0]
start_y = position[:,1]
start_z = position[:,2]

start_vx = velocity[:,0]
start_vy = velocity[:,1]
start_vz = velocity[:,2]

xs = []
ys = []
zs = []

vxs = []
vys = []
vzs = []

for i in range(time):
    
    x = position[:,0]
    y = position[:,1]
    z = position[:,2]
    
    vx = velocity[:,0]
    vy = velocity[:,1]
    vz = velocity[:,2]

    distance = euclidean_distances(position) + 1*10**(-150)
    
    newx = [0] * len(x)
    newy = [0] * len(y)
    newz = [0] * len(x)
    
    x_force = [0] * len(x)
    y_force = [0] * len(y) 
    z_force = [0] * len(z)
    
    a_x = [0] * len(x)
    a_y = [0] * len(y)
    a_z = [0] * len(z)
    
    x_differences = np.matrix(x) - np.matrix(x).T
    y_differences = np.matrix(y) - np.matrix(y).T 
    z_differences = np.matrix(z) - np.matrix(z).T
    
    x_differences = np.squeeze(np.array(x_differences))
    y_differences = np.squeeze(np.array(y_differences))
    z_differences = np.squeeze(np.array(z_differences))
    
    v_x = [0] * len(x)
    v_y = [0] * len(y)
    v_z = [0] * len(z)
    
    for particle in range(0, len(x)):
        x_force[particle] = - ((k * q**2 / (distance[particle]**2)) * (x_differences[particle] / (distance[particle]))).sum()
        y_force[particle] = - ((k * q**2 / (distance[particle]**2)) * (y_differences[particle] / (distance[particle]))).sum()
        z_force[particle] = - ((k * q**2 / (distance[particle]**2)) * (z_differences[particle] / (distance[particle]))).sum()

        a_x[particle] = x_force[particle] / (2*m)
        a_y[particle] = y_force[particle] / (2*m)
        a_z[particle] = z_force[particle] / (2*m)
        
        v_x[particle] = vx[particle] + a_x[particle] * delta_t
        v_y[particle] = vy[particle] + a_y[particle] * delta_t
        v_z[particle] = vz[particle] + a_z[particle] * delta_t
            
        ## WALL COLLISIONS ##
        if (x[particle] + v_x[particle] >= Lx) or (x[particle] + v_x[particle] <= -Lx):
            v_x[particle] = -1 * v_x[particle]            
        else:
            v_x[particle] = v_x[particle] 
            
        if (y[particle] + v_y[particle] >= Ly) or (y[particle] + v_y[particle] <= -Ly):
            v_y[particle] = -1 * v_y[particle]            
        else:
            v_y[particle] = v_y[particle]  
                
        if (z[particle] + v_z[particle] >= Lz) or (z[particle] + v_z[particle] <= -Lz):
            v_z[particle] = -1 * v_z[particle]            
        else:
            v_z[particle] = v_z[particle]   

    
    for i, j in itertools.combinations_with_replacement(range(len(x)), 2):
        if i!=j :
            if np.abs(x[j] + v_x[j] * delta_t - x[i] + v_x[i] * delta_t) <= 2* Radius:
                    v_x[i] = -1 * v_x[i]
                    v_x[j] = -1 * v_x[j]
                    
            if np.abs(y[j] + v_y[j] * delta_t - y[i] + v_y[i] * delta_t) <= 2* Radius:
                    v_y[i] = -1 * v_y[i]
                    v_y[j] = -1 * v_y[j]
                    
            if np.abs(z[j] + v_y[j] * delta_t - y[i] + v_y[i] * delta_t) <= 2* Radius:
                    v_z[i] = -1 * v_y[i]
                    v_z[j] = -1 * v_y[j]                    
                       
    for particle in range(0, len(x)):
        newx[particle] = x[particle] + v_x[particle] * delta_t
        newy[particle] = y[particle] + v_y[particle] * delta_t 
        newz[particle] = z[particle] + v_z[particle] * delta_t     
                
    # APPEND FOR RECORDING AND PLOTTING
    xs.append(np.squeeze(newx))
    ys.append(np.squeeze(newy))
    zs.append(np.squeeze(newz))
    
    vxs.append(np.squeeze(v_x)), vys.append(np.squeeze(v_y)), vzs.append(np.squeeze(v_z))
                  
    # REDEFINE POSITION AND VELOCITY TO RESTART RUN 
    position = np.squeeze(np.array([newx, newy, newz])).T
    velocity = np.array([v_x, v_y, v_z]).T
    
    particle_distances = euclidean_distances(position)

    for i, j in itertools.combinations_with_replacement(range(len(x)),2):
        if i!= j:
            if particle_distances[i][j] <= strong_force_distance:
                FUSION += 1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(start_x, start_y, start_z, color = 'r')

ax.scatter(xs, ys, zs, c='b',alpha = 0.2, marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# plt.xlim(-1, 1)
# plt.ylim(-1, 1)

plt.show();


plt.plot(np.array(xs))
# plt.plot(np.array(vxs))

plt.hlines([-Lx, Lx], 0, time)

plt.legend()
plt.show();                

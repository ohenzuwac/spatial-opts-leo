#!/usr/bin/env python
# coding: utf-8

# In[15]:

#print("I got to line 6");
"""
#Simulating agent/robot spatial trajectories for 2 options/targets

"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import math
import time
from matplotlib.animation import FuncAnimation

#I_0 = [0.3,0.4,2,np.pi/2,0,0]; #initial conditions for all equations [z10,z20,B0,theta0,x0,y0]
I_0 = [0.33,0.3,2.1,np.pi/2,0,0];
t0 = 0;
tn = 600;
h = 0.2;






def Euler(t0,I_0,h,tn):
    t = np.arange(t0,tn,h); #start time to end time with a spacing h between values
    
    #prepping each array which will hold solutions for a particular variable with zeros
    z1 = np.zeros(t.size);
    z2 = np.zeros(t.size);
    B = np.zeros(t.size);
    theta = np.zeros(t.size);
    x = np.zeros(t.size);
    y = np.zeros(t.size)
    xplot = []
    yplot = []
    
    z1[0] = I_0[0];
    z2[0]= I_0[1];
    B[0] = I_0[2];
    theta[0] = I_0[3];
    x[0] = I_0[4];
    y[0] = I_0[5];
    
    #the vector I holds the solutions to the coupled ODEs
    I = [z1,z2,B,theta,x,y]
    
    fig,ax = plt.subplots()



    for i in np.arange(0,t.size-1,1):
        # y(i + 1) = y(i) + h * f(y(i), t(i));
        B[i+1] = B[i] + h*updateConnection(x[i],y[i],B[i]);
        z1[i+1] = z1[i] + h*updateScore1(z1[i],z2[i],B[i]);
        z2[i+1] = z2[i] + h*updateScore2(z1[i],z2[i],B[i]);
        theta[i+1] = theta[i] + h*updateHeading(z1[i],z2[i],x[i],y[i],theta[i]);
        x[i+1] = x[i] + h*(updatePosi(theta[i]))[0];
        y[i+1] = y[i] + h*(updatePosi(theta[i]))[1];

        if i % 100 == 0:
            ax.cla()
            ax.plot(-10, 8, 'ks')
            ax.plot(10, 8, 'ks')
            ax.plot(0,0,'gs')
            ax.set_title("Animated Trajectory")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.scatter(x[i],y[i])
            plt.pause(0.1)

            

            
        


def IODE(I,t):
    
    """
    # t: scalar time
    # z: two-dimensional array [z1, z2, B, theta, x, y] 
    """
    
    #define state variables
    z1 = I[0];
    z2 = I[1];
    B = I[2];
    theta = I[3];
    x = I[4];
    y = I[5];
    
    #define constants
    d = 1; #damping coefficien
    u = 1; #attention
    b = 0; #input/bias about certain option(s)
    
    k = 1; #constant
    kp = 4; #constant (K_prime)
    
    v = 0.02; #velocity
    
    xt1 = 10; #x coordinate of target 1
    yt1 = 8; #y coordinate of target 1
    xt2 = -10; #x coordinate of target 2
    yt2 = 8; #y coordinate of target 2
    
    phi = np.array([math.atan2((yt1-y),(xt1-x)),math.atan2((yt2-y),(xt2-x))]);
    
    tau = 2; #timescale
      
    return np.array([-d*z1 + u*np.sum(np.tanh(B*z2))+b, 
                     -d*z2 + u*np.sum(np.tanh(B*z1))+b, 
                     (-B+kp*np.cos(k*(phi[0]-phi[1])))/tau, 
                     (np.sum(np.array([np.max(np.array([z1,0]))*np.sin((phi[0]-theta)/2),
                                      np.max(np.array([z2,0]))*np.sin((phi[1]-theta)/2)])))/tau, 
                     v*np.cos(theta), 
                     v*np.sin(theta)]);




def updateScore1 (z1,z2,B): #function to update z1

     #define constants
    d = 1; #damping coefficien
    u = 1; #attention
    b = 0; #input/bias about certain option(s)
    
    k = 1; #constant
    kp = 4; #constant (K_prime)
    
    v = 0.02; #velocity
    
    xt1 = 10; #x coordinate of target 1
    yt1 = 8; #y coordinate of target 1
    xt2 = -10; #x coordinate of target 2
    yt2 = 8; #y coordinate of target 2
    
    return -d*z1 + u*np.sum(np.tanh(B*z2)) +b



def updateScore2 (z1,z2,B): #function to update s2
     #define constants
    d = 1; #damping coefficien
    u = 1; #attention
    b = 0; #input/bias about certain option(s)
    
    k = 1; #constant
    kp = 4; #constant (K_prime)
    
    v = 0.02; #velocity
    
    xt1 = 10; #x coordinate of target 1
    yt1 = 8; #y coordinate of target 1
    xt2 = -10; #x coordinate of target 2
    yt2 = 8; #y coordinate of target 2
    
    return -d*z2 + u*np.sum(np.tanh(B*z1))+b



def updateConnection(x,y,B):
         #define constants
    d = 1; #damping coefficien
    u = 1; #attention
    b = 0; #input/bias about certain option(s)
    
    k = 1; #constant
    kp = 4; #constant (K_prime)
    
    v = 0.02; #velocity
    
    xt1 = 10; #x coordinate of target 1
    yt1 = 8; #y coordinate of target 1
    xt2 = -10; #x coordinate of target 2
    yt2 = 8; #y coordinate of target 2
    
    phi = np.array([math.atan2((yt1-y),(xt1-x)),math.atan2((yt2-y),(xt2-x))]);
    tau = 2; #timescale
    
    return -B+kp*np.cos(k*(phi[0]-phi[1]))/tau



def updateHeading (z1,z2,x,y,theta):
    #define constants
    d = 1; #damping coefficien
    u = 1; #attention
    b = 0; #input/bias about certain option(s)
    
    k = 1; #constant
    kp = 4; #constant (K_prime)
    
    v = 0.02; #velocity
    
    xt1 = 10; #x coordinate of target 1
    yt1 = 8; #y coordinate of target 1
    xt2 = -10; #x coordinate of target 2
    yt2 = 8; #y coordinate of target 2
    
    phi = np.array([math.atan2((yt1-y),(xt1-x)),math.atan2((yt2-y),(xt2-x))]);
    tau = 2;

    return np.sum(np.array([np.max(np.array([z1,0]))*np.sin((phi[0]-theta)/2),
                                      np.max(np.array([z2,0]))*np.sin((phi[1]-theta)/2)]))/tau;
                  
                  


def updatePosi(theta):
    
    d = 1; #damping coefficien
    u = 1; #attention
    b = 0; #input/bias about certain option(s)
    
    k = 1; #constant
    kp = 4; #constant (K_prime)
    
    v = 0.02; #velocity
    
    xt1 = 10; #x coordinate of target 1
    yt1 = 8; #y coordinate of target 1
    xt2 = -10; #x coordinate of target 2
    yt2 = 8; #y coordinate of target 2
    
    return np.array([ v*np.cos(theta), 
                     v*np.sin(theta)])

Euler(t0,I_0,h,tn)





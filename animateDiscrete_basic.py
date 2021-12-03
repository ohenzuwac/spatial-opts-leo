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
import imageio
import os

#I_0 = [0.3,0.4,2,np.pi/2,0,0]; #initial conditions for all equations [z10,z20,B0,theta0,x0,y0]

#I need to make the default attention u = 1!!!

#I_0 = [0.33,0.3,2.1,np.pi/2,0,0]; #gets a pretty standard plot where z1 is favored [no noise]
#_0 = [0.3,0.33,2.1,np.pi/2,0,0]; #gets a pretty standard plot where z2 is favored [no noise]
#I_0 = [0.3,0.3,2.1,np.pi/2,0,0]; #equal values of scores
I_0 = [0.34,0.3,2.1,np.pi/2,0,0]; #equal values of scores
I_0 = [0.34,0.3,2.1,np.pi/2,0,0]; #set u = 4 --> tune how quickly decisions are made
I_0 = [0.4,0.3,2.1,np.pi/2,0,0]; # set kp =2;
#I_0 = [0.4,0.3,2.1,np.pi/2,0,0]; # adding noise with default params, z1 favored
I_0 = [0.34,0.3,2.1,np.pi/2,0,0]; # adding noise with default params, z1 favored and distance bias included with g = 1
I_0 = [0.34,0.3,2.1,np.pi/2,0,0]; # adding noise with default params, z1 favored and distance bias included with g = 0.01 and heading noise



t0 = 0;
tn = 600;
h = 0.1; #its hard to choose an appropriate step size when changing the parameters
#be careful when chaging the parameters. If the step size is not tuned correctly, you may get incorrect results

#try changing time scales to see if this corrects the issue.

# define constants
d = 1;  # damping coefficien
u = 1;  # attention
#b = 0;  # input/bias about certain option(s)
g = 0.01; #gain for the distance bias

k = 1;  # constant
kp = 4;  # constant (K_prime)

v = 0.02;  # velocity

xt1 = 10;  # x coordinate of target 1
yt1 = 8;  # y coordinate of target 1
xt2 = -10;  # x coordinate of target 2
yt2 = 8;  # y coordinate of target 2




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
    
   # fig,ax = plt.subplots()



    for i in np.arange(0,t.size-1,1):
        # y(i + 1) = y(i) + h * f(y(i), t(i));
        B[i+1] = B[i] + h*updateConnection(x[i],y[i],B[i]);
        z1[i+1] = z1[i] + h*updateScore1(z1[i],z2[i],B[i],x[i],y[i]);
        z2[i+1] = z2[i] + h*updateScore2(z1[i],z2[i],B[i],x[i],y[i]);
        theta[i+1] = theta[i] + h*updateHeading(z1[i],z2[i],x[i],y[i],theta[i]);
        x[i+1] = x[i] + h*(updatePosi(theta[i])[0]);
        y[i+1] = y[i] + h*(updatePosi(theta[i])[1]);
    return I

            

            
        
def plotTraj(x,y):
    fig,ax = plt.subplots()
    filenames = []
    for i in range(len(x)):
        if i % 200 == 0:
            ax.cla()
            ax.plot(-10, 8, 'ks')
            ax.plot(10, 8, 'ks')
            ax.plot(I_0[4],I_0[5],'gs')
            ax.set_title("Animated Trajectory")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.scatter(x[i],y[i])
            plt.pause(h)

#saves the animated plots as a gif
"""
            filename = f'{i}.png'
            filenames.append(filename)
            # save frame
            plt.savefig(filename)

    # build gif
    with imageio.get_writer('defaultparams_z1favored034_smalldistBias_headingnoisy.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Remove files
    for filename in set(filenames):
        os.remove(filename)

"""
def plotScores(z1,z2):
    t = np.arange(t0, tn, h)
    fig,ax = plt.subplots()
    tplot = []
    z1plot = []
    z2plot = []
    #plt.xlim(0, 1)
    plt.ylim(-1, 1)
    for i in range(len(z1)):
        if i % 30 == 0:
            ax.cla()
            ax.set_title("Evolution of the Scores")
            ax.set_xlabel("time")
            #ax.plot(0,0,'gs')
            tplot.append(t[i])
            z1plot.append(z1[i])
            z2plot.append(z2[i])
            ax.plot(tplot,z1plot,'-g',label= 'z1')
            ax.plot(tplot,z2plot,'-b',label = "z2")
            plt.pause(h/10)

def plotConnect(B):
    t = np.arange(t0, tn, h)
    fig, ax = plt.subplots()
    tplot = []
    Bplot = []
    for i in range(len(B)):
        if i % 30 == 0:
            ax.cla()
            ax.set_title("Evolution of the Connection")
            ax.set_xlabel("time")
            ax.set_ylabel("Beta")
            tplot.append(t[i])
            Bplot.append(B[i])
            ax.plot(tplot, Bplot, '-g')
            plt.pause(h / 10)



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

    
    phi = np.array([math.atan2((yt1-y),(xt1-x)),math.atan2((yt2-y),(xt2-x))]);
    
    tau = 2; #timescale
      
    return np.array([-d*z1 + u*np.sum(np.tanh(B*z2))+b, 
                     -d*z2 + u*np.sum(np.tanh(B*z1))+b, 
                     (-B+kp*np.cos(k*(phi[0]-phi[1])))/tau, 
                     (np.sum(np.array([np.max(np.array([z1,0]))*np.sin((phi[0]-theta)/2),
                                      np.max(np.array([z2,0]))*np.sin((phi[1]-theta)/2)])))/tau, 
                     v*np.cos(theta), 
                     v*np.sin(theta)]);




def updateScore1 (z1,z2,B,x,y): #function to update z1
    r = np.array([np.sqrt(np.square(xt1 - x) + np.square((yt1 - y))), np.sqrt(np.square(xt2 - x) + np.square(yt2 - y))])
    b = g/np.square(r[0])
    return -d*z1 + u*np.sum(np.tanh(B*z2)) +b

def updateScore2 (z1,z2,B,x,y): #function to update s2
    r = np.array([np.sqrt(np.square(xt1 - x) + np.square((yt1 - y))), np.sqrt(np.square(xt2 - x) + np.square(yt2 - y))])
    b = g / np.square(r[1])
    
    return -d*z2 + u*np.sum(np.tanh(B*z1))+b



def updateConnection(x,y,B):
         #define constants

    phi = np.array([math.atan2((yt1-y),(xt1-x)),math.atan2((yt2-y),(xt2-x))]);
    tau = 2; #timescale
    
    return -B+kp*np.cos(k*(phi[0]-phi[1]))/tau



def updateHeading (z1,z2,x,y,theta):
    
    phi = np.array([math.atan2((yt1-y),(xt1-x)),math.atan2((yt2-y),(xt2-x))]);
    tau = 2;
    noise = np.random.normal()
    # noise = 0

    return np.sum(np.array([np.max(np.array([z1,0]))*np.sin((phi[0]-theta)/2)+0.05*noise,
                                      np.max(np.array([z2,0]))*np.sin((phi[1]-theta)/2)+0.05*noise]))/tau;
                  
                  


def updatePosi(theta):
    noise = np.random.normal()
    #noise = 0
    return np.array([v*np.cos(theta)+0.05*noise,
                     v*np.sin(theta)+0.05*noise])




I = Euler(t0,I_0,h,tn)

plotTraj(I[4],I[5])

#plotScores(I[0],I[1])
#plotConnect(I[2])



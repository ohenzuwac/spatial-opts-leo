#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
#Simulating agent/robot spatial trajectories for 2 options/targets

"""


# In[2]:


import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import math
import os
import imageio


# In[3]:


phi10 = math.atan2(8,10)
phi20 = math.atan2(8,-10);
#I_0 = [0.3,0.4,2,np.pi/2,0,0]; #initial conditions for all equations [z10,z20,B0,theta0,x0,y0]
I_0 = [0.30,0.30,2.1,np.pi/2,0,0];
T = 600; #time length

#one agent, two targets
#incorporating variation of robot movement and phi with respect to time as well as dynamics of theta


# In[4]:


def IODE(t,I):
    
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
    g =  1; #gain for the distance bias term

    k = 1; #constant
    kp = 4; #constant (K_prime)
    
    v = 0.02; #velocity
    
    xt1 = 10; #x coordinate of target 1
    yt1 = 8; #y coordinate of target 1
    xt2 = -10; #x coordinate of target 2
    yt2 = 8; #y coordinate of target 2
    
    phi = np.array([math.atan2((yt1-y),(xt1-x)),math.atan2((yt2-y),(xt2-x))]);

    r = np.array([np.sqrt(np.square(xt1-x)+ np.square((yt1-y))),np.sqrt(np.square(xt2-x) + np.square(yt2-y))])
    b1 = g/r[0];  # input/bias about certain option(s) --> for target 1
    b2 = g/r[1];

    
    tau = 2; #timescale
      
    return np.array([-d*z1 + u*np.sum(np.tanh(B*z2))+b1,
                     -d*z2 + u*np.sum(np.tanh(B*z1))+b2,
                     (-B+kp*np.cos(k*(phi[0]-phi[1])))/tau, 
                     (np.sum(np.array([np.max(np.array([z1,0]))*np.sin((phi[0]-theta)/2),
                                      np.max(np.array([z2,0]))*np.sin((phi[1]-theta)/2)])))/tau, 
                     v*np.cos(theta), 
                     v*np.sin(theta)]);


# In[5]:


I_out = solve_ivp(IODE,np.array([0,T]),I_0,dense_output=True); #obtain soln


# In[6]:


ts = np.linspace(0,T,500); #100 evenly spaced values from 0 to T
It = I_out.sol(ts); #solution of the integral at the specified time points

#assigning variable names to the solutions for coding ease
z1t = It[0]; #target 1 score
z2t = It[1]; #target 2 score
Bt = It[2]; #beta
thetat = It[3]; #theta
xt = It[4]; #x coordinate of robot
yt = It[5]; #y coordinate of robot


# In[7]:

def plotTraj(x,y):
    fig,ax = plt.subplots()
    filenames = []
    for i in range(len(x)):
        if i % 30 == 0:
            ax.cla()
            ax.plot(-10, 8, 'ks')
            ax.plot(10, 8, 'ks')
            ax.plot(I_0[4],I_0[5],'gs')
            ax.set_title("Animated Trajectory")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.scatter(x[i],y[i])
            plt.pause(0.1)
#saves the animated plot as a gif

            """
            filename = f'{i}.png'
            filenames.append(filename)
            # save frame
            plt.savefig(filename)

            # build gif
    with imageio.get_writer('defaultparams_z1favored_taueqs4.gif', mode='I') as writer:
           for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

            # Remove files
    for filename in set(filenames):
            os.remove(filename)
        """
plotTraj(xt,yt)


#plot of robot pathing
plt.plot(-10,8,'ks')
plt.plot(10,8,'ks')
plt.plot(xt,yt,label = 'Pathing')
plt.legend()
plt.xlabel('x position')


# In[8]:


#plot of score 1
plt.plot(ts,z1t,label = 'z1');
plt.xlabel('t');
plt.legend(bbox_to_anchor=(1, 1))
#plt.legend('z1');


# In[9]:


#plot of score 2
plt.plot(ts,z2t,color="orange", label = 'z2');
plt.xlabel('t');
plt.legend(bbox_to_anchor=(1, 1))
#plt.legend('z2 ');


# In[10]:


#Plot of target scores vs time

#line1 = plt.plot(ts,z1t,label='z1')
#line2 = plt.plot(ts,z2t,label='z2')


lines = plt.plot(ts,z1t,ts,z2t)
plt.legend(iter(lines), ('z1','z2'))
plt.xlabel('t')
plt.show()
#plt.legend()
#print(z1t);
#print(z2t);


# In[11]:


#plot of beta
plt.plot(ts,Bt,label = 'Beta')
plt.legend()
plt.xlabel('t')


# In[12]:


#plot of theta
plt.plot(ts,thetat,label = 'Theta')
plt.legend()
plt.xlabel('t')


# In[13]:


#plot of x position
plt.plot(ts,xt,label = 'x position')
plt.legend()
plt.xlabel('t')


# In[14]:


#plot of y position
plt.plot(ts,yt,label = 'y position')
plt.legend()
plt.xlabel('t')


# In[15]:


# compute phi(t)

#xt1 = 10; #x coordinate of target 1
#yt1 = 8; #y coordinate of target 1
#xt2 = -10; #x coordinate of target 2
#yt2 = 8; #y coordinate of target 2

#phi = np.array([math.atan2((yt1-yt),(xt1-xt)),math.atan2((yt2-yt),(xt2-xt))]);
#phi1t = phit[0,:]
#phi2t = phit[1,:]


# In[16]:


#plot of phi1
#plt.plot(ts,phi1t,label = 'phi1')
#plt.legend()
#plt.xlabel('t')


# In[17]:


#plot of phi2
#plt.plot(ts,phi2t,label = 'phi2')
#plt.legend()
#plt.xlabel('t')


# In[18]:


#xrange = np.linspace(-10,10,100)
#ys = math.atan2(xrange)
#plt.plot(xrange,ys)


# In[ ]:





# In[ ]:





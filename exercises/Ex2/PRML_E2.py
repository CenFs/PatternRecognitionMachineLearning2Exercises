# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 18:53:07 2017

@author: Jingyi
"""


""" Load CSV file """
import numpy as np
with open("D:\Hanke\Python\PRML\locationData.csv", "r") as f:
    lines = []

    for line in f:
        tmp = line.strip().split(' ')
#        print tmp
        tmp = [float(v) for v in tmp]
 #       print tmp
        lines.append(tmp)

lines = np.array(lines)
#print lines


fp = np.loadtxt("D:\Hanke\Python\PRML\locationData.csv")

if (np.any(lines == fp)):
    print "For np.any: Equal!"
else:
    print "For np.any: Not Equal!"
if (np.all(lines == fp)):
    print "For np.all: Equal!"
else:
    print "For np.all: Not Equal!"

#print lines
#print fp




""" Implement functions """
def gaussian(x, mu, sigma):
    p = 1/np.sqrt(2*np.pi*sigma**2) * np.exp((-1/2*sigma**2)*(x-mu)**2)
    return p

def log_gaussian(x, mu, sigma):
    p = 1/np.sqrt(2*np.pi*sigma**2) * np.exp((-1/2*sigma**2)*(x-mu)**2)
    lnp = np.log(p)
    return lnp

import matplotlib.pyplot as plt
mu = 0
sigma = 1
x = np.linspace(-5, 6, 50)
plt.plot(x, gaussian(x, mu, sigma))
plt.figure()
plt.plot(x, log_gaussian(x, mu, sigma))





""" Estimate sinusoidal parameters """
f = 0.017
w = np.sqrt(0.25) * np.random.randn(100)
n = range(0, 100)

x = np.sin(np.dot(2*np.pi*f, n)) + w
plt.figure()
plt.plot(x)

import cmath
scores = []
frequencies = []
for f in np.linspace(0, 0.5, 1000):
    n = np.arange(100)
    i = cmath.sqrt(-1)
    z = -2*np.pi*i*f*n
    e = np.exp(z)
    score = np.abs(np.dot(x, e))
    scores.append(score)
    frequencies.append(f)
fHat = frequencies[np.argmax(scores)]
print fHat


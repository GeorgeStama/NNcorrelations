from __future__ import print_function
import argparse
import numpy as np
import torch


import matplotlib.pyplot as plt

trnEBP = np.load('trnEBPs.npy')
trnMVG = np.load('trnMVGs.npy')
testEBP = np.load('testEBPs.npy')
testMVG = np.load('testMVGs.npy')

trnEBP = np.load('trnEBPn.npy')
trnMVG = np.load('trnMVGn.npy')
testEBP = np.load('testEBPn.npy')
testMVG = np.load('testMVGn.npy')
#
# trnEBP = np.load('trnEBP.npy')
# trnMVG = np.load('trnMVG.npy')
# testEBP = np.load('testEBP.npy')
# testMVG = np.load('testMVG.npy')


plt.plot(trnEBP,label = 'EBP_test')
plt.plot(trnMVG,label = 'MVG_test')
plt.legend()
plt.ylabel('test performance %')
plt.show()

plt.plot(testEBP,label = 'EBP_test')
plt.plot(testMVG,label = 'MVG_test')
plt.legend()
plt.ylabel('test performance %')
plt.show()
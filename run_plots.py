from __future__ import print_function
import argparse
import numpy as np
import torch


import matplotlib.pyplot as plt


trnEBP = np.load('trnEBPn.npy')
trnMVG = np.load('trnMVGn.npy')
testEBP = np.load('testEBPn.npy')
testMVG = np.load('testMVGn.npy')

trnEBP = np.load('trnEBPL2sh.npy')
trnMVG = np.load('trnMVGL2sh.npy')
testEBP = np.load('testEBPL2sh.npy')
testMVG = np.load('testMVGL2sh.npy')

trnEBP = np.load('trnEBPL2big_lr3.npy')
trnMVG = np.load('trnMVGL2big_lr3.npy')
testEBP = np.load('testEBPL2big_lr3.npy')
testMVG = np.load('testMVGL2big_lr3.npy')

# trnEBP = np.load('trnEBPL2sgd.npy')
# trnMVG = np.load('trnMVGL2sgd.npy')
# testEBP = np.load('testEBPL2sgd.npy')
# testMVG = np.load('testMVGL2sgd.npy')

trnEBP = np.load('trnEBPL2.npy')
trnMVG = np.load('trnMVGL2.npy')
testEBP = np.load('testEBPL2.npy')
testMVG = np.load('testMVGL2.npy')
#
# trnEBP = np.load('trnEBP.npy')
# trnMVG = np.load('trnMVG.npy')
# testEBP = np.load('testEBP.npy')
# testMVG = np.load('testMVG.npy')
# #
# trnEBP = np.load('trnEBPbigLR.npy')
# testEBP = np.load('testEBPbigLR.npy')


plt.plot(trnEBP,label = 'EBP_test')
plt.plot(trnMVG,label = 'MVG_test')
plt.legend()
plt.ylabel('training loss')
plt.show()

plt.plot(testEBP,label = 'EBP_test')
plt.plot(testMVG,label = 'MVG_test')
plt.legend()
plt.ylabel('test performance %')
plt.show()
#! /usr/bin/env python3

import chaospy
import numpy as np
import pickle
import time

from evalPoly import eval44Poly
from robotModel import RobotModel

# input options
initHeadingShape = [0., np.pi/18]
initRatioShape = [0., .1*.76]
# calculation options
Order = 4
# GPCMethod = ('regression', 10)
GPCMethod = 'quadrature'
NSamplesEval = 1000000
deltaSensitivity = False

# check if robot is inbounds
MaxHeadingDev = np.pi/6.
MaxRatioDev = .3*.76

# initial declerations
times = {}
outputNames = ['Heading', 'Distance']

# first and only GPC
print('Creating GPC model')
startTime = time.time()

gpcHeadingShape = [0., np.pi/18]
gpcRatioShape = [0., .1*.76]
distGPC = chaospy.J(chaospy.TruncNormal(-MaxHeadingDev, MaxHeadingDev, gpcHeadingShape[0], gpcHeadingShape[1]), chaospy.TruncNormal(-MaxRatioDev, MaxRatioDev, gpcRatioShape[0], gpcRatioShape[1]),
                    chaospy.Normal(0, 1), chaospy.Normal(0, 1))
orthoExpansion = chaospy.generate_expansion(Order, distGPC)

if GPCMethod == 'quadrature':
  gpcSamples, weights = chaospy.generate_quadrature(Order, distGPC, rule='gaussian')
else:
  NSamplesGPC = GPCMethod[1]*len(orthoExpansion)
  gpcSamples = distGPC.sample(NSamplesGPC, rule='sobol')
gpcEvals = RobotModel(gpcSamples)
if deltaSensitivity:
  gpcEvals -= gpcSamples[:2,:].T

if GPCMethod == 'quadrature':
  approxModel = chaospy.fit_quadrature(orthoExpansion, gpcSamples, weights, gpcEvals)
else:
  approxModel = chaospy.fit_regression(orthoExpansion, gpcSamples, gpcEvals)

times['GPCInit'] = time.time() - startTime

# common prob. distribution
distCombined = chaospy.J(chaospy.TruncNormal(-MaxHeadingDev, MaxHeadingDev, initHeadingShape[0], initHeadingShape[1]), chaospy.TruncNormal(-MaxRatioDev, MaxRatioDev, initRatioShape[0], initRatioShape[1]),
                         chaospy.Normal(0, 1), chaospy.Normal(0, 1))
distCombined2 = chaospy.J(chaospy.TruncNormal(-MaxHeadingDev, MaxHeadingDev, initHeadingShape[0], initHeadingShape[1]), chaospy.TruncNormal(-MaxRatioDev, MaxRatioDev, initRatioShape[0], initRatioShape[1]),
                          chaospy.Normal(0, 1), chaospy.Normal(0, 1),
                          chaospy.TruncNormal(-MaxHeadingDev, MaxHeadingDev, initHeadingShape[0], initHeadingShape[1]), chaospy.TruncNormal(-MaxRatioDev, MaxRatioDev, initRatioShape[0], initRatioShape[1]),
                          chaospy.Normal(0, 1), chaospy.Normal(0, 1))

print('Calculating GPC model sensitivity analytically')
startTime = time.time()
gpcAnSens = chaospy.Sens_m(approxModel, distCombined)
print(np.array2string(gpcAnSens,separator=','))
times['GPCSensAn'] = time.time() - startTime

print('Creating',NSamplesEval,'samples for empirical sensitivity analysis')
startTime = time.time()
samples = np.empty((4,NSamplesEval*6))
samples2 = distCombined2.sample(NSamplesEval, rule='sobol')
samples[:,:NSamplesEval] = samples2[:4,:]
samples[:,NSamplesEval:NSamplesEval*2] = samples2[4:,:]
for i in range(4):
  samples[:,NSamplesEval*(i+2):NSamplesEval*(i+3)] = samples[:,:NSamplesEval]
  samples[i,NSamplesEval*(i+2):NSamplesEval*(i+3)] = samples[i,NSamplesEval:NSamplesEval*2]
times['MCSSample'] = time.time() - startTime

def calcSens(evals):
  mean = np.mean(evals[NSamplesEval:NSamplesEval*2,:], axis=0)
  var = np.var(evals[NSamplesEval:NSamplesEval*2,:], axis=0)
  sens = np.empty((4,2))
  for i in range(4):
    prod = evals[NSamplesEval:NSamplesEval*2,:]*evals[NSamplesEval*(i+2):NSamplesEval*(i+3),:]
    prodmean = np.mean(prod, axis=0)
    vari = prodmean - mean**2
    sens[i,:] = vari/var
  return sens

print('Calculating MCS sensitivity empirically')
startTime = time.time()
evals = RobotModel(samples)
if deltaSensitivity:
  evals -= samples[:2,:].T
mcsEmSens = calcSens(evals)
times['MCSSensEm'] = time.time() - startTime
print(np.array2string(mcsEmSens,separator=','))

print('Calculating GPC sensitivity empirically')
startTime = time.time()
evals = eval44Poly([approxModel[0].coefficients, approxModel[1].coefficients], samples).T
# if deltaSensitivity:
#   evals -= samples[:2,:].T
gpcEmSens = calcSens(evals)
times['GPCSensEm'] = time.time() - startTime
print(np.array2string(gpcEmSens,separator=','))

print('Max diff:', np.max(np.abs(gpcAnSens-mcsEmSens)), np.max(np.abs(gpcEmSens-mcsEmSens)), np.max(np.abs(gpcAnSens-gpcEmSens)))

print(times)

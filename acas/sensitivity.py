#! /usr/bin/env python3

import chaospy
import numpy as np
import pickle
import time

from evalPoly import eval43Poly, eval63Poly
from robotModel import RobotModel

# state space dimensions - used in many places
StateDimensions = 5
OutputDimensions = 3

# input options
xShape = [-6000,6000]
yShape = [-5000,2500]
psiShape = [-np.pi/1,+np.pi/1]
# calculation options
Order = 4
# GPCMethod = ('regression', 10)
GPCMethod = 'quadrature'
useNN = False
NSamplesEval = 1000000
deltaSensitivity = False
excludeUnsafe = False

# check if robot is inbounds
MinSafeSep = 500 # feet
def inbounds(samples):
  return np.linalg.norm(samples[:2,:],ord=2,axis=0) >= MinSafeSep

# initial declerations
times = {}

# initialize model
robotModel = RobotModel()
model = robotModel.model

# first and only GPC
startTime = time.time()

print('Creating GPC models')

distGPC = chaospy.J(chaospy.Uniform(*xShape), chaospy.Uniform(*yShape), chaospy.Uniform(*psiShape))
orthoExpansion = chaospy.generate_expansion(Order, distGPC)
if GPCMethod == 'quadrature':
  gpcSamplesUnaug, weights = chaospy.generate_quadrature(Order, distGPC, rule='gaussian')
else:
  NSamplesGPC = GPCMethod[1]*len(orthoExpansion)
  gpcSamplesUnaug = distGPC.sample(NSamplesGPC, rule='sobol')
gpcSamples = np.vstack((gpcSamplesUnaug, np.zeros((2,gpcSamplesUnaug.shape[1]))))
gpcEvals, _ = model(gpcSamples, useNN=useNN)
if deltaSensitivity:
  gpcEvals -= gpcSamples[:3,:].T
if GPCMethod == 'quadrature':
  approxModel = chaospy.fit_quadrature(orthoExpansion, gpcSamplesUnaug, weights, gpcEvals)
else:
  approxModel = chaospy.fit_regression(orthoExpansion, gpcSamplesUnaug, gpcEvals)

print('Creating ancillary model')
robotModel.constructPerceptionModel()

times['GPCInit'] = time.time() - startTime

# common prob. distribution
distCombined = chaospy.J(chaospy.Uniform(*xShape), chaospy.Uniform(*yShape), chaospy.Uniform(*psiShape))
distCombined2 = chaospy.J(chaospy.Uniform(*xShape), chaospy.Uniform(*yShape), chaospy.Uniform(*psiShape),
                          chaospy.Uniform(*xShape), chaospy.Uniform(*yShape), chaospy.Uniform(*psiShape))

print('Calculating GPC model sensitivity analytically')
startTime = time.time()
gpcAnSens = chaospy.Sens_m(approxModel, distCombined)
print(np.array2string(gpcAnSens,separator=',',floatmode='fixed',suppress_small=True))
times['GPCSensAn'] = time.time() - startTime

print('Creating',NSamplesEval,'samples for empirical sensitivity analysis')
startTime = time.time()
samples = np.empty((3,NSamplesEval*5))
if excludeUnsafe:
  samples2 = distCombined2.sample(int(NSamplesEval*1.1), rule='latin_hypercube', seed=0)
  inboundsSamples = inbounds(samples2[:2,:]) & inbounds(samples2[3:5,:])
  samples2 = samples2[:,inboundsSamples]
  assert samples2.shape[1] >= NSamplesEval
  samples2 = samples2[:,:NSamplesEval]
else:
  samples2 = distCombined2.sample(NSamplesEval, rule='sobol')
samples[:,:NSamplesEval] = samples2[:3,:]
samples[:,NSamplesEval:NSamplesEval*2] = samples2[3:,:]
for i in range(3):
  samples[:,NSamplesEval*(i+2):NSamplesEval*(i+3)] = samples[:,:NSamplesEval]
  samples[i,NSamplesEval*(i+2):NSamplesEval*(i+3)] = samples[i,NSamplesEval:NSamplesEval*2]
samples = np.vstack((samples, np.zeros((2,samples.shape[1]))))
times['Samples'] = time.time() - startTime

def calcSens(evals):
  mean = np.mean(evals[NSamplesEval:NSamplesEval*2,:], axis=0)
  var = np.var(evals[NSamplesEval:NSamplesEval*2,:], axis=0)
  sens = np.empty((3,3))
  for i in range(3):
    prod = evals[NSamplesEval:NSamplesEval*2,:]*evals[NSamplesEval*(i+2):NSamplesEval*(i+3),:]
    prodmean = np.mean(prod, axis=0)
    vari = prodmean - mean**2
    sens[i,:] = vari/var
  return sens

print('Calculating MCS sensitivity empirically')
startTime = time.time()
evals, _ = robotModel.modelWithAncillary(samples)
if deltaSensitivity:
  evals -= samples[:3,:].T
minimum, maximum = np.min(evals, axis=0), np.max(evals, axis=0)
print('MCS min/max:', minimum, maximum)
mcsEmSens = calcSens(evals)
times['MCSSensEm'] = time.time() - startTime
print(np.array2string(mcsEmSens,separator=',',floatmode='fixed',suppress_small=True))

print('Calculating GPC sensitivity empirically')
startTime = time.time()
polyEvalr = {4:eval43Poly, 6:eval63Poly}[Order]
evals = polyEvalr([approxModel[0].coefficients, approxModel[1].coefficients, approxModel[2].coefficients], samples).T
# if deltaSensitivity:
#   evals -= samples[:3,:].T
minimum, maximum = np.min(evals, axis=0), np.max(evals, axis=0)
print('GPC min/max:', minimum, maximum)
gpcEmSens = calcSens(evals)
times['GPCSensEm'] = time.time() - startTime
print(np.array2string(gpcEmSens,separator=',',floatmode='fixed',suppress_small=True))

print('Max diff:', np.max(np.abs(gpcAnSens-mcsEmSens)[:2,:2]), np.max(np.abs(gpcEmSens-mcsEmSens)[:2,:2]), np.max(np.abs(gpcAnSens-gpcEmSens)[:2,:2]))

print(times)

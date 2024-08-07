#! /usr/bin/env python3

import chaospy
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import scipy.stats as stats
import time

from evalPoly import eval44Poly
from robotModel import *

# state space dimensions - used in many places
StateDimensions = 2

# controller to use
# controller = 'stanley'
controller = 'pursuit'

# input options
if curvedPerceptionModel:
  Radius = -97.9 # m
else:
  Radius = np.inf
initHeadingShape = [0,.075]
initDistanceShape = [0,.4]
mcTracesPickleFile = 'mcTraces/' + controller + ('' if Radius==np.inf else 'Curved') + 'TracesVeryBusy.pickle'
# calculation options
TimeSteps = 100
Order = 4
# GPCMethod = ('regression', 10)
GPCMethod = 'quadrature'
NSamplesEvalGPC = 10000
NSamplesEvalMC = 1000
seed = 0
# output options
zoom = 5
plotFreq = 10

# check if robot is inbounds
MaxHeadingDev = np.pi/12.
MaxDistanceDev = 1.2
def inbounds(samples):
  M, C = 8.51252, 1.86857 # line passing through (pi/12-3.5pi/60, 1.2) and (pi/12, 1.2-6.5*.24) (perception model bounds)
  return (abs(samples[0,:]) <= MaxHeadingDev) & (abs(samples[1,:]) <= MaxDistanceDev) #& (abs(samples[1,:] + M*samples[0,:]) <= C)

# initial declerations
runInfo = {
  'initStateParams': (initHeadingShape, initDistanceShape),
  'TimeSteps': TimeSteps,
  'Order': Order,
  'GPCMethod': GPCMethod,
  'NSamplesEval': (NSamplesEvalGPC, NSamplesEvalMC),
  'seed': seed,
  'totalGPCTime': 0,
  'totalMCTime': 0,
  'gpcInbounds': [],
  'mcInbounds': [],
  'gpcSamples': [],
  'mcSamples': [],
  'KS': np.ones((TimeSteps,StateDimensions)),
  'Wass': np.full((TimeSteps,StateDimensions),np.inf),
}

# initialize model
robotModel = RobotModel()
model = robotModel.relModel

# plotting data
fig, axs = plt.subplots(StateDimensions, TimeSteps//plotFreq+2, figsize=((TimeSteps//plotFreq+2)*zoom,StateDimensions*zoom), constrained_layout=True)
outputNames = ['Heading','Distance']

# first and only GPC
print('Creating GPC model')
startTime = time.time()

gpcHeadingShape = [0, np.pi/36.]
gpcDistanceShape = [0, .4]
distGPC = chaospy.J(\
                    # chaospy.Uniform(-MaxHeadingDev, MaxHeadingDev), chaospy.Uniform(-MaxDistanceDev, MaxDistanceDev),\
                    chaospy.TruncNormal(-MaxHeadingDev, MaxHeadingDev, gpcHeadingShape[0], gpcHeadingShape[1]), chaospy.TruncNormal(-MaxDistanceDev, MaxDistanceDev, gpcDistanceShape[0], gpcDistanceShape[1]),\
                    chaospy.Normal(0,1),chaospy.Normal(0,1))
orthoExpansion = chaospy.generate_expansion(Order, distGPC)

if GPCMethod == 'quadrature':
  gpcSamples, weights = chaospy.generate_quadrature(Order, distGPC, rule='gaussian')
else:
  NSamplesGPC = GPCMethod[1]*len(orthoExpansion)
  gpcSamples = distGPC.sample(NSamplesGPC, rule='sobol')
  goodSamples = inbounds(gpcSamples)
  gpcSamples = gpcSamples[:,goodSamples]
gpcEvals = model(gpcSamples, Radius, controller)

if GPCMethod == 'quadrature':
  approxModel = chaospy.fit_quadrature(orthoExpansion, gpcSamples, weights, gpcEvals)
else:
  approxModel = chaospy.fit_regression(orthoExpansion, gpcSamples, gpcEvals)

approxModelCoeffs = [approxModel[0].coefficients, approxModel[1].coefficients]

runInfo['totalGPCTime'] += time.time() - startTime

# samples for GPC and MC
print('Creating initial samples')
distErrors = chaospy.Iid(chaospy.Normal(0,1),2)
distCombined = chaospy.J(chaospy.TruncNormal(-MaxHeadingDev, MaxHeadingDev, initHeadingShape[0], initHeadingShape[1]), chaospy.TruncNormal(-MaxDistanceDev, MaxDistanceDev, initDistanceShape[0], initDistanceShape[1]),\
                         chaospy.Normal(0,1),chaospy.Normal(0,1))
startTime = time.time()
gpcSamples = distCombined.sample(NSamplesEvalGPC, rule='sobol')
gpcGoodSamples = np.ones(NSamplesEvalGPC, dtype=bool)
sampleTime = time.time() - startTime
runInfo['totalGPCTime'] += sampleTime
if mcTracesPickleFile:
  startTime = time.time()
  mcSamplesRaw = pickle.load(open(mcTracesPickleFile,'rb'))
  mcSamples = np.empty((NSamplesEvalMC, TimeSteps+1, StateDimensions))
  for i in range(NSamplesEvalMC):
    for timestep in range(TimeSteps+1):
      mcSamples[i,timestep,0] = mcSamplesRaw[i][0][timestep][0]
      mcSamples[i,timestep,1] = mcSamplesRaw[i][0][timestep][1]
  runInfo['mcSamples'] = mcSamples
  mcGoodSamples = np.ones(NSamplesEvalMC, dtype=bool)
  runInfo['totalMCTime'] += time.time() - startTime
  if NSamplesEvalGPC % NSamplesEvalMC == 0:
    repetitions = NSamplesEvalGPC // NSamplesEvalMC
    gpcSamples[:2,:] = np.tile(mcSamples[:,0,:].T, repetitions) # use same samples for GPC
else:
  if NSamplesEvalGPC == NSamplesEvalMC:
    mcSamples = gpcSamples.copy()
  else:
    mcSamples = distCombined.sample(NSamplesEvalMC, rule='sobol')
  runInfo['totalMCTime'] += sampleTime

print('Initial setup times: GPC:', runInfo['totalGPCTime'], 'MC:', runInfo['totalMCTime'])

# main loop
for timestep in range(TimeSteps):
  print('Timestep',timestep)

  startTime = time.time()

  # eliminate out of bound samples for GPC and store rate of elimination
  goodSamples = inbounds(gpcSamples) & gpcGoodSamples
  gpcGoodSamples = goodSamples
  runInfo['gpcInbounds'].append(np.count_nonzero(goodSamples)/NSamplesEvalGPC)
  print('Fraction of GPC robot states still inbounds:', runInfo['gpcInbounds'][-1])

  # calculate next timestep states using GPC
  print('Evaluating GPC model on samples')
  gpcEvals = eval44Poly(approxModelCoeffs, gpcSamples[:,goodSamples]).T

  runInfo['gpcSamples'].append(gpcEvals.copy())
  runInfo['totalGPCTime'] += time.time() - startTime

  print('Doing Monte Carlo for comparison')
  startTime = time.time()
  mcEvals = None
  if mcTracesPickleFile:
    goodSamples = inbounds(mcSamples[:,timestep,:].T) & mcGoodSamples
    mcGoodSamples = goodSamples
    runInfo['mcInbounds'].append(np.count_nonzero(goodSamples)/NSamplesEvalMC)
    mcEvals = mcSamples[goodSamples,timestep+1,:]
  else:
    goodSamples = inbounds(mcSamples)
    mcSamples = mcSamples[:,goodSamples]
    runInfo['mcInbounds'].append(mcSamples.shape[1]/NSamplesEvalMC)
    mcEvals = model(mcSamples, Radius, controller)
    runInfo['mcSamples'].append(mcEvals.copy())
  print('Fraction of MC robot states still inbounds:', runInfo['mcInbounds'][-1])
  runInfo['totalMCTime'] += time.time() - startTime

  if timestep != TimeSteps-1:
    print('Generating samples for next timestep')
    startTime = time.time()
    gpcErrSamples = distErrors.sample(gpcEvals.shape[0], rule='latin_hypercube', seed=seed)
    seed += 1
    gpcSamples[:,gpcGoodSamples] = np.vstack((gpcEvals.T, gpcErrSamples))
    runInfo['totalGPCTime'] += time.time() - startTime
    if not mcTracesPickleFile:
      startTime = time.time()
      mcErrSamples = distErrors.sample(mcEvals.shape[0], rule='latin_hypercube', seed=seed)
      seed += 1
      mcSamples = np.vstack((mcEvals.T, mcErrSamples))
      runInfo['totalMCTime'] += time.time() - startTime

  print('Comparing distributions')
  for i in range(StateDimensions):
    ks, _ = stats.ks_2samp(gpcEvals[:,i], mcEvals[:,i])
    runInfo['KS'][timestep,i] = ks
    wass = stats.wasserstein_distance(gpcEvals[:,i], mcEvals[:,i])
    runInfo['Wass'][timestep,i] = wass

  if (timestep+1)%plotFreq==0:
    print('Generating plot')
    for i in range(StateDimensions):
      hist, binEdges = np.histogram(gpcEvals[:,i], bins=100)
      cdfGPCX = (binEdges[:-1] + binEdges[1:])/2
      cdfGPCY = np.cumsum(hist)/gpcEvals.shape[0]
      hist, binEdges = np.histogram(mcEvals[:,i], bins=100)
      cdfMCX = (binEdges[:-1] + binEdges[1:])/2
      cdfMCY = np.cumsum(hist)/mcEvals.shape[0]
      ax = axs[i,timestep//plotFreq]
      ax.plot(cdfGPCX, cdfGPCY, color='red')
      ax.plot(cdfMCX, cdfMCY, color='green')
      ax.set_xlabel('Cart-'+('Straight' if Radius==np.inf else 'Curved')+' '+outputNames[i], fontsize=16)
      ax.set_ylabel('CDF', fontsize=16)

  if timestep == TimeSteps-1:
    print('Mean', np.mean(gpcEvals, axis=0), np.mean(mcEvals, axis=0))
    print('Std', np.std(gpcEvals, axis=0), np.std(mcEvals, axis=0))

print('Final data processing')
runInfo['gpcInbounds'] = np.array(runInfo['gpcInbounds'])
runInfo['mcInbounds'] = np.array(runInfo['mcInbounds'])
runInfo['gpcInbounds'] /= runInfo['gpcInbounds'][0]
runInfo['mcInbounds'] /= runInfo['mcInbounds'][0]

print('Max KS:', np.max(runInfo['KS'], axis=0), 'Max Wass:', np.max(runInfo['Wass'], axis=0), 'Max SPD:', np.max(np.abs(runInfo['gpcInbounds']-runInfo['mcInbounds'])))

print('Total time: GPC:', runInfo['totalGPCTime'], ' MCS:', runInfo['totalMCTime'])

# safe prob. statistics
minT = 1.0
for timestep in range(TimeSteps):
  gpcTemp = np.zeros(NSamplesEvalGPC)
  gpcTemp[:int(runInfo['gpcInbounds'][timestep]*NSamplesEvalGPC)] = 1
  mcTemp = np.zeros(NSamplesEvalMC)
  mcTemp[:int(runInfo['mcInbounds'][timestep]*NSamplesEvalMC)] = 1
  statistic, pvalue = stats.ttest_ind(gpcTemp, mcTemp, equal_var=False)
  minT = min(minT, pvalue)
  if pvalue < 0.05:
    print(timestep, pvalue)
l2Scaled = np.linalg.norm(runInfo['gpcInbounds']-runInfo['mcInbounds'],ord=2)/np.sqrt(TimeSteps)
cov = np.cov(np.column_stack((runInfo['gpcInbounds'],runInfo['mcInbounds'])),rowvar=0)
corr = cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])
print('min t-val:', minT, 'l2Scaled:', l2Scaled, 'corr:', corr)

print('Generating final plots')
for i in range(StateDimensions):
  ax = axs[i,TimeSteps//plotFreq]
  ax.plot(range(1,TimeSteps+1),runInfo['KS'][:,i])
  ax.set_xlabel('Time', fontsize=16)
  ax.set_ylabel(outputNames[i]+' KS', fontsize=16)

ax = axs[0,TimeSteps//plotFreq+1]
xRange = range(0,TimeSteps)
ax.plot(xRange, runInfo['gpcInbounds'], color='red')
ax.fill_between(xRange, stats.binom.ppf(0.025, NSamplesEvalGPC, runInfo['gpcInbounds'])/NSamplesEvalGPC, stats.binom.ppf(0.975, NSamplesEvalGPC, runInfo['gpcInbounds'])/NSamplesEvalGPC, color='red', alpha=0.2)
ax.plot(xRange, runInfo['mcInbounds'], color='green')
ax.fill_between(xRange, stats.binom.ppf(0.025, NSamplesEvalMC, runInfo['mcInbounds'])/NSamplesEvalMC, stats.binom.ppf(0.975, NSamplesEvalMC, runInfo['mcInbounds'])/NSamplesEvalMC, color='green', alpha=0.2)
ax.set_xlabel('Time', fontsize=16)
ax.set_ylabel('Cart-'+('Straight' if Radius==np.inf else 'Curved')+' Safe Probability', fontsize=16)
ax.set_ylim([.9,1.001])

timestr = time.strftime('%Y%m%d-%H%M%S')
fig.suptitle('Experiment: {}  Order: {}  GPC Runtime: {:.2f}  MC Runtime: {:.2f}'.format(timestr, Order, runInfo['totalGPCTime'], runInfo['totalMCTime']), fontsize=16)
plt.savefig('gpc.png')
plt.close()

for i in range(2):
  fig, ax = plt.subplots(figsize=(zoom,zoom), constrained_layout=True)
  hist, binEdges = np.histogram(gpcEvals[:,i], bins=100)
  cdfGPCX = (binEdges[:-1] + binEdges[1:])/2
  cdfGPCY = np.cumsum(hist)/gpcEvals.shape[0]
  hist, binEdges = np.histogram(mcEvals[:,i], bins=100)
  cdfMCX = (binEdges[:-1] + binEdges[1:])/2
  cdfMCY = np.cumsum(hist)/mcEvals.shape[0]
  ax.plot(cdfMCX, cdfMCY, color='red', linestyle=(0,(8,2)))
  ax.plot(cdfGPCX, cdfGPCY, color='blue')
  ax.set_xlabel('Cart-'+('Straight' if Radius==np.inf else 'Curved')+' '+outputNames[i], fontsize=16)
  ax.set_ylabel('CDF', fontsize=16)
  plt.savefig(outputNames[i]+'.png')
  plt.close()
fig, ax = plt.subplots(figsize=(zoom,zoom), constrained_layout=True)
xRange = range(0,TimeSteps)
ax.plot(xRange, runInfo['mcInbounds'], color='red', linestyle=(0,(8,2)))
lower = np.choose(runInfo['mcInbounds'] == 1, (stats.binom.ppf(0.025, NSamplesEvalMC, runInfo['mcInbounds'])/NSamplesEvalMC, 1))
upper = np.choose(runInfo['mcInbounds'] == 1, (stats.binom.ppf(0.975, NSamplesEvalMC, runInfo['mcInbounds'])/NSamplesEvalMC, 1))
ax.fill_between(xRange, lower, upper, color='red', alpha=0.2)
ax.plot(xRange, runInfo['gpcInbounds'], color='blue')
lower = np.choose(runInfo['gpcInbounds'] == 1, (stats.binom.ppf(0.025, NSamplesEvalGPC, runInfo['gpcInbounds'])/NSamplesEvalGPC, 1))
upper = np.choose(runInfo['gpcInbounds'] == 1, (stats.binom.ppf(0.975, NSamplesEvalGPC, runInfo['gpcInbounds'])/NSamplesEvalGPC, 1))
ax.fill_between(xRange, lower, upper, color='blue', alpha=0.2)
ax.set_xlabel('Time', fontsize=16)
ax.set_ylabel('Cart-'+('Straight' if Radius==np.inf else 'Curved')+' Safe Probability', fontsize=16)
ax.set_ylim([.9,1.001])
plt.savefig('safe.png')
plt.close()

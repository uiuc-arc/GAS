#! /usr/bin/env python3

import chaospy
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import scipy.stats as stats
import time

from evalPoly import eval44Poly
from robotModel import RobotModel

# input options
initHeadingShape = [0., np.pi/18]
initRatioShape = [0., .1*.76]
gpcHeadingShape = [0., np.pi/18]
gpcRatioShape = [0., .1*.76]
gpcDistPickleFile=None#'perceptionError/gpcPerceptSampO5.pickle'
gpcDistStatsPickleFile=None#'perceptionError/gpcPerceptDistO5.pickle'
mcTracesPickleFile='mcTraces/mcTraces.pickle'
# calculation options
TimeSteps = 100
Order = 4
CrossTruncation = 1.
# GPCMethod = ('regression', 10)
GPCMethod = 'quadrature'
NSamplesEvalGPC = 10000
NSamplesEvalMC = 1000
seed = 0
# output options
zoom = 5
plotFreq = 10

# check if robot is inbounds
MaxHeadingDev = np.pi/6.
MaxRatioDev = .3*.76
def inbounds(samples):
  return (abs(samples[0,:]) <= MaxHeadingDev) & (abs(samples[1,:]) <= MaxRatioDev)

# initial declerations
runInfo = {
  'initStateParams': (initHeadingShape, initRatioShape),
  'TimeSteps': TimeSteps,
  'Order': Order,
  'CrossTruncation': CrossTruncation,
  'GPCMethod': GPCMethod,
  'NSamplesEval': (NSamplesEvalGPC, NSamplesEvalMC),
  'seed': seed,
  'totalGPCTime': 0,
  'totalMCTime': 0,
  'gpcInbounds': [],
  'mcInbounds': [],
  'gpcSamples': [],
  'mcSamples': [],
  'KS': np.ones((TimeSteps,2)),
  'Wass': np.full((TimeSteps,2),np.inf),
}
fig, axs = plt.subplots(2, TimeSteps//plotFreq+2, figsize=((TimeSteps//plotFreq+2)*zoom,2*zoom), constrained_layout=True)
outputNames = ['Heading', 'Distance']

# first and only GPC
print('Creating GPC model')
startTime = time.time()

distGPC = chaospy.J(chaospy.TruncNormal(-MaxHeadingDev, MaxHeadingDev, gpcHeadingShape[0], gpcHeadingShape[1]), chaospy.TruncNormal(-MaxRatioDev, MaxRatioDev, gpcRatioShape[0], gpcRatioShape[1]),
                    # chaospy.Uniform(-MaxHeadingDev, MaxHeadingDev), chaospy.Uniform(-MaxRatioDev, MaxRatioDev),
                    chaospy.Normal(0, 1), chaospy.Normal(0, 1))
orthoExpansion = chaospy.generate_expansion(Order, distGPC, cross_truncation=CrossTruncation)

if GPCMethod == 'quadrature':
  gpcSamples, weights = chaospy.generate_quadrature(Order, distGPC, rule='gaussian')
else:
  NSamplesGPC = GPCMethod[1]*len(orthoExpansion)
  gpcSamples = distGPC.sample(NSamplesGPC, rule='sobol')
gpcEvals = RobotModel(gpcSamples, gpcDistPickleFile=gpcDistPickleFile, gpcDistStatsPickleFile=gpcDistStatsPickleFile)

if GPCMethod == 'quadrature':
  approxModel = chaospy.fit_quadrature(orthoExpansion, gpcSamples, weights, gpcEvals)
else:
  approxModel = chaospy.fit_regression(orthoExpansion, gpcSamples, gpcEvals)

approxModelCoeffs = [approxModel[0].coefficients, approxModel[1].coefficients]

runInfo['totalGPCTime'] += time.time() - startTime

# samples for GPC and MC
print('Creating initial samples')
distErrors = chaospy.Iid(chaospy.Normal(0, 1), 2)
# distCombined = chaospy.J(chaospy.Normal(initHeadingShape[0], initHeadingShape[1]), chaospy.Normal(initRatioShape[0], initRatioShape[1]),
distCombined = chaospy.J(chaospy.TruncNormal(-MaxHeadingDev, MaxHeadingDev, initHeadingShape[0], initHeadingShape[1]), chaospy.TruncNormal(-MaxRatioDev, MaxRatioDev, initRatioShape[0], initRatioShape[1]),
                         chaospy.Normal(0, 1), chaospy.Normal(0, 1))
startTime = time.time()
gpcSamples = distCombined.sample(NSamplesEvalGPC, rule='sobol')
gpcGoodSamples = np.ones(NSamplesEvalGPC, dtype=bool)
# gpcSamples[:2,:] = np.clip(gpcSamples[:2,:], np.array([-MaxHeadingDev, -MaxRatioDev]).reshape((2,1)), np.array([MaxHeadingDev, MaxRatioDev]).reshape((2,1)))
sampleTime = time.time() - startTime
runInfo['totalGPCTime'] += sampleTime
if mcTracesPickleFile:
  startTime = time.time()
  mcSamplesRaw = pickle.load(open(mcTracesPickleFile,'rb'))
  mcSamples = np.empty((NSamplesEvalMC, TimeSteps+1, 2))
  for i in range(NSamplesEvalMC):
    #mcSamples[i,:,:] = mcSamplesRaw[i][0][:TimeSteps+1]
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
  # gpcEvals = approxModel(*(gpcSamples[:,goodSamples])).T
  # gpcEvals = RobotModel(gpcSamples[:,goodSamples])
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
    mcEvals = RobotModel(mcSamples)
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
  for i in range(2):
    ks, _ = stats.ks_2samp(gpcEvals[:,i], mcEvals[:,i])
    runInfo['KS'][timestep,i] = ks
    wass = stats.wasserstein_distance(gpcEvals[:,i], mcEvals[:,i])
    runInfo['Wass'][timestep,i] = wass

  if (timestep+1)%plotFreq==0:
    print('Generating plot')
    for i in range(2):
      hist, binEdges = np.histogram(gpcEvals[:,i], bins=100)
      cdfGPCX = (binEdges[:-1] + binEdges[1:])/2
      cdfGPCY = np.cumsum(hist)/gpcEvals.shape[0]
      hist, binEdges = np.histogram(mcEvals[:,i], bins=100)
      cdfMCX = (binEdges[:-1] + binEdges[1:])/2
      cdfMCY = np.cumsum(hist)/mcEvals.shape[0]
      ax = axs[i,timestep//plotFreq]
      ax.plot(cdfGPCX, cdfGPCY, color='red', label='GAS')
      ax.plot(cdfMCX, cdfMCY, color='green', label='MCS')
      ax.set_xlabel('Crop-Monitor '+outputNames[i], fontsize=16)
      ax.set_ylabel('CDF', fontsize=16)

  if timestep == TimeSteps-1:
    print('Mean', np.mean(gpcEvals, axis=0), np.mean(mcEvals, axis=0))
    print('Std', np.std(gpcEvals, axis=0), np.std(mcEvals, axis=0))

print('Final data processing')
runInfo['gpcInbounds'] = np.array(runInfo['gpcInbounds'])
runInfo['mcInbounds'] = np.array(runInfo['mcInbounds'])
runInfo['gpcInbounds'] /= runInfo['gpcInbounds'][0]
runInfo['mcInbounds'] /= runInfo['mcInbounds'][0]

print('Max KS:', np.max(runInfo['KS'], axis=0), np.argmax(runInfo['KS'], axis=0), 'Max Wass:', np.max(runInfo['Wass'], axis=0), np.argmax(runInfo['Wass'], axis=0), 'Max SPD:', np.max(np.abs(runInfo['gpcInbounds']-runInfo['mcInbounds'])))

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
for i in range(2):
  windowSize = 5
  smoothKS = np.convolve(runInfo['KS'][:,i], np.ones(windowSize)/windowSize, mode='valid')
  ax = axs[i,TimeSteps//plotFreq]
  ax.plot(range(windowSize,TimeSteps+1),smoothKS)
  ax.set_xlabel('Time', fontsize=16)
  ax.set_ylabel(outputNames[i]+' KS', fontsize=16)

ax = axs[0,TimeSteps//plotFreq+1]
xRange = range(0,TimeSteps)
ax.plot(xRange, runInfo['gpcInbounds'], color='red')
ax.fill_between(xRange, stats.binom.ppf(0.025, NSamplesEvalGPC, runInfo['gpcInbounds'])/NSamplesEvalGPC, stats.binom.ppf(0.975, NSamplesEvalGPC, runInfo['gpcInbounds'])/NSamplesEvalGPC, color='red', alpha=0.2)
ax.plot(xRange, runInfo['mcInbounds'], color='green')
ax.fill_between(xRange, stats.binom.ppf(0.025, NSamplesEvalMC, runInfo['mcInbounds'])/NSamplesEvalMC, stats.binom.ppf(0.975, NSamplesEvalMC, runInfo['mcInbounds'])/NSamplesEvalMC, color='green', alpha=0.2)
ax.set_xlabel('Time', fontsize=16)
ax.set_ylabel('Crop-Monitor Safe Probability', fontsize=16)
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
  ax.set_xlabel('Crop-Monitor '+outputNames[i], fontsize=16)
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
ax.set_ylabel('Crop-Monitor Safe Probability', fontsize=16)
ax.set_ylim([.9,1.001])
plt.savefig('safe.png')
plt.close()

fig, ax = plt.subplots(2, 1, figsize=(zoom,zoom), constrained_layout=True, sharex=True)
xRange = range(0,TimeSteps)
for i in range(2):
  windowSize = 5
  smoothKS = np.convolve(runInfo['Wass'][:,i], np.ones(windowSize)/windowSize, mode='valid')
  ax[i].plot(range(windowSize,TimeSteps+1),smoothKS,label=outputNames[i], color='black')
  ax[i].legend(fontsize=16)#, loc='center right')
  ax[i].set_ylabel('Wass', fontsize=16)
ax[1].set_xlabel('Time', fontsize=16)
plt.savefig('Wass.png')
plt.close()

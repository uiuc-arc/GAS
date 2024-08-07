#! /usr/bin/env python3

import chaospy
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import scipy.stats as stats
import time

from evalPoly import eval43Poly, eval63Poly
from robotModel import RobotModel

# state space dimensions - used in many places
StateDimensions = 5
OutputDimensions = 3

# feet to km converter
distanceScale = np.array([0.0003048, 0.0003048, 1.])

# input options
xShape = [-5000,5000]
yShape = [1500,2500]
psiShape = [np.pi-np.pi/4,np.pi+np.pi/4]
# calculation options
TimeSteps = 100
Order = 4
# GPCMethod = ('regression', 10)
GPCMethod = 'quadrature'
polarGPC = False
useNN = False
NSamplesEvalGPC = 10000
NSamplesEvalMC = 1000
# output options
zoom = 5
plotFreq = 20

# check if robot is inbounds
MinSafeSep = 500 # feet
def inbounds(samples):
  return np.linalg.norm(samples[:2,:],ord=2,axis=0) >= MinSafeSep

# adjust angles to [-pi,pi)
def normalizeAngles(ang):
  twoPi = 2*np.pi
  return ang - np.floor(ang/twoPi+.5)*twoPi

# polar <-> cart
def polar2cart(samples):
  output = np.empty_like(samples)
  output[0,:] = samples[0,:] * -np.sin(samples[1,:])
  output[1,:] = samples[0,:] * np.cos(samples[1,:])
  return output

def cart2polar(samples):
  output = np.empty_like(samples)
  output[0,:] = np.linalg.norm(samples, ord=2, axis=0)
  output[1,:] = np.arctan2(-samples[0,:], samples[1,:])
  return output

# initial declerations
runInfo = {
  'initStateParams': (xShape, yShape, psiShape),
  'TimeSteps': TimeSteps,
  'Order': Order,
  'GPCMethod': GPCMethod,
  'NSamplesEval': (NSamplesEvalGPC, NSamplesEvalMC),
  'totalGPCTime': 0,
  'totalMCTime': 0,
  'gpcInbounds': [],
  'mcInbounds': [],
  'gpcSamples': [],
  'mcSamples': [],
  'KS': np.ones((TimeSteps,OutputDimensions)),
  'Wass': np.full((TimeSteps,OutputDimensions),np.inf),
}

# initialize model
robotModel = RobotModel()
model = robotModel.model

# plotting data
fig, axs = plt.subplots(OutputDimensions, TimeSteps//plotFreq+2, figsize=((TimeSteps//plotFreq+2)*zoom,OutputDimensions*zoom), constrained_layout=True)
outputNames = ['Crossrange Distance', 'Downrange Distance', 'Relative Heading', 'Adv', 'tau']

# first and only GPC
startTime = time.time()

print('Creating GPC models')

approxModel = [None]*5
if polarGPC:
  distGPC = chaospy.J(chaospy.Uniform(0,10000), chaospy.Uniform(-np.pi, np.pi), chaospy.Uniform(-np.pi, np.pi))
else:
  distGPC = chaospy.J(chaospy.Uniform(-6000,6000), chaospy.Uniform(-5000,2500), chaospy.Uniform(-np.pi, np.pi))
orthoExpansion = chaospy.generate_expansion(Order, distGPC)
if GPCMethod == 'quadrature':
  gpcSamplesUnaug, weights = chaospy.generate_quadrature(Order, distGPC, rule='gaussian')
else:
  NSamplesGPC = GPCMethod[1]*len(orthoExpansion)
  gpcSamplesUnaug = distGPC.sample(NSamplesGPC, rule='sobol')
gpcSamples = np.vstack((gpcSamplesUnaug, np.zeros((2,gpcSamplesUnaug.shape[1]))))
cartGPCSamples = gpcSamples.copy()
if polarGPC:
  cartGPCSamples[:2,:] = polar2cart(gpcSamples[:2,:])
for pa in range(5):
  gpcSamples[3,:] = pa
  gpcEvals, _ = model(cartGPCSamples, useNN=useNN)
  if GPCMethod == 'quadrature':
    approxModel[pa] = chaospy.fit_quadrature(orthoExpansion, gpcSamplesUnaug, weights, gpcEvals)
  else:
    approxModel[pa] = chaospy.fit_regression(orthoExpansion, gpcSamplesUnaug, gpcEvals)

approxModelCoeffs = [[model[0].coefficients, model[1].coefficients, model[2].coefficients] for model in approxModel]

runInfo['totalGPCTime'] += time.time() - startTime

print('Creating ancillary model')
startTime = time.time()
robotModel.constructPerceptionModel()
runInfo['totalGPCTime'] += time.time() - startTime
print('Ancillary model construction time:', time.time() - startTime)

# samples for GPC and MC
print('Creating initial samples')
distCombined = chaospy.J(chaospy.Uniform(xShape[0], xShape[1]), chaospy.Uniform(yShape[0], yShape[1]), chaospy.Uniform(psiShape[0], psiShape[1]))
startTime = time.time()
gpcSamples = distCombined.sample(NSamplesEvalGPC, rule='sobol')
gpcGoodSamples = np.ones(NSamplesEvalGPC, dtype=bool)
advisories = np.zeros(NSamplesEvalGPC)
taus = np.zeros(NSamplesEvalGPC)
gpcSamples = np.vstack((gpcSamples, advisories, taus))
runInfo['totalGPCTime'] += time.time() - startTime
startTime = time.time()
mcSamples = distCombined.sample(NSamplesEvalMC, rule='sobol')
mcGoodSamples = np.ones(NSamplesEvalMC, dtype=bool)
advisories = np.zeros(NSamplesEvalMC)
taus = np.zeros(NSamplesEvalMC)
mcSamples = np.vstack((mcSamples, advisories, taus))
runInfo['totalMCTime'] += time.time() - startTime

print('Initial setup times: GPC:', runInfo['totalGPCTime'], 'MC:', runInfo['totalMCTime'])

mcMins = np.min(mcSamples, axis=1)
mcMaxs = np.max(mcSamples, axis=1)
def updateMinMax():
  global mcMins, mcMaxs # , gpcMins, gpcMaxs
  if np.count_nonzero(mcGoodSamples) > 0:
    mcMins = np.minimum(mcMins, np.min(mcSamples[:,mcGoodSamples], axis=1))
    mcMaxs = np.maximum(mcMaxs, np.max(mcSamples[:,mcGoodSamples], axis=1))

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
  gpcEvals = np.empty((gpcSamples.shape[1],3))
  if polarGPC:
    gpcSamples[:2,:] = cart2polar(gpcSamples[:2,:])
  for pa in range(5):
    mask = (gpcSamples[3,:] == pa)
    # gpcEvals[mask] = approxModel[pa](*gpcSamples[:,mask]).T
    gpcEvals[mask] = eval43Poly(approxModelCoeffs[pa], gpcSamples[:,mask]).T
  runInfo['gpcSamples'].append(gpcEvals.copy())
  runInfo['totalGPCTime'] += time.time() - startTime

  print('Doing Monte Carlo for comparison')
  startTime = time.time()
  goodSamples = inbounds(mcSamples) & mcGoodSamples
  mcGoodSamples = goodSamples
  runInfo['mcInbounds'].append(np.count_nonzero(goodSamples)/NSamplesEvalMC)
  mcEvals, mcAdvisories = model(mcSamples, useNN=useNN)
  runInfo['mcSamples'].append(mcEvals.copy())
  print('Fraction of MC robot states still inbounds:', runInfo['mcInbounds'][-1])
  runInfo['totalMCTime'] += time.time() - startTime

  if timestep != TimeSteps-1:
    print('Generating samples for next timestep')
    startTime = time.time()
    gpcAdvisories = robotModel.fastClassify(gpcSamples)
    gpcSamples = np.vstack((gpcEvals.T, gpcAdvisories, np.zeros(gpcEvals.shape[0])))
    gpcSamples[2,:] = normalizeAngles(gpcSamples[2,:])
    runInfo['totalGPCTime'] += time.time() - startTime
    startTime = time.time()
    mcSamples = np.vstack((mcEvals.T, mcAdvisories, np.zeros(mcEvals.shape[0])))
    mcSamples[2,:] = normalizeAngles(mcSamples[2,:])
    runInfo['totalMCTime'] += time.time() - startTime

  print('Comparing distributions')
  for i in range(OutputDimensions):
    ks, _ = stats.ks_2samp(gpcEvals[:,i], mcEvals[:,i])
    runInfo['KS'][timestep,i] = ks
    wass = stats.wasserstein_distance(gpcEvals[:,i], mcEvals[:,i])
    runInfo['Wass'][timestep,i] = wass * distanceScale[i]

  if (timestep+1)%plotFreq==0:
    print('Generating plot')
    for i in range(OutputDimensions):
      hist, binEdges = np.histogram(gpcEvals[:,i], bins=100)
      cdfGPCX = (binEdges[:-1] + binEdges[1:])/2
      cdfGPCY = np.cumsum(hist)/gpcEvals.shape[0]
      hist, binEdges = np.histogram(mcEvals[:,i], bins=100)
      cdfMCX = (binEdges[:-1] + binEdges[1:])/2
      cdfMCY = np.cumsum(hist)/mcEvals.shape[0]
      ax = axs[i,timestep//plotFreq]
      ax.plot(cdfGPCX, cdfGPCY, color='red')
      ax.plot(cdfMCX, cdfMCY, color='green')
      ax.set_xlabel('ACAS-'+('NN' if useNN else 'Table')+' '+outputNames[i], fontsize=16)
      ax.set_ylabel('CDF', fontsize=16)

  updateMinMax()
  if timestep == TimeSteps-1:
    print('Mean', np.mean(gpcEvals, axis=0) * distanceScale, np.mean(mcEvals, axis=0) * distanceScale)
    print('Std', np.std(gpcEvals, axis=0) * distanceScale, np.std(mcEvals, axis=0) * distanceScale)

print('MC state variable min/max range:')
print(mcMins[:3])
print(mcMaxs[:3])

print('MC time breakdown: control', robotModel.controlTime, 'dynamics', robotModel.dynamicsTime)

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
for i in range(OutputDimensions):
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
ax.set_ylabel('ACAS-'+('NN' if useNN else 'Table')+' Safe Probability', fontsize=16)
ax.set_ylim([.9,1.001])

timestr = time.strftime('%Y%m%d-%H%M%S')
fig.suptitle('Experiment: {}  Order: {}  GPC Runtime: {:.2f}  MC Runtime: {:.2f}'.format(timestr, Order, runInfo['totalGPCTime'], runInfo['totalMCTime']), fontsize=16)
plt.savefig('gpc.png')
plt.close()

for i in range(OutputDimensions):
  fig, ax = plt.subplots(figsize=(zoom,zoom), constrained_layout=True)
  hist, binEdges = np.histogram(gpcEvals[:,i], bins=100)
  cdfGPCX = (binEdges[:-1] + binEdges[1:])/2
  cdfGPCY = np.cumsum(hist)/gpcEvals.shape[0]
  hist, binEdges = np.histogram(mcEvals[:,i], bins=100)
  cdfMCX = (binEdges[:-1] + binEdges[1:])/2
  cdfMCY = np.cumsum(hist)/mcEvals.shape[0]
  ax.plot(cdfMCX, cdfMCY, color='red', linestyle=(0,(8,2)))
  ax.plot(cdfGPCX, cdfGPCY, color='blue')
  ax.set_xlabel('ACAS-'+('NN' if useNN else 'Table')+' '+outputNames[i], fontsize=16)
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
ax.set_ylabel('ACAS-'+('NN' if useNN else 'Table')+' Safe Probability', fontsize=16)
ax.set_ylim([.8,1.002])
plt.savefig('safe.png')
plt.close()

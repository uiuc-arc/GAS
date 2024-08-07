import chaospy
import numpy as np
import pickle
from scipy.linalg import sqrtm

from perceptionModel import *

# constants
DesiredHeading = 0
DesiredRatio = 0
AngVelMin = -1
AngVelMax = 1
K = 1
Speed = .2
CycleTime = .1

# model of perception as a function of actual state
def perceptionStateDepModel(heading, ratio):
  def evalPoly(coeffs):
    return coeffs[0] + coeffs[1]*heading + coeffs[2]*ratio + coeffs[3]*heading**2 + coeffs[4]*heading*ratio + coeffs[5]*ratio**2 \
           + coeffs[6]*heading**3 + coeffs[7]*heading**2*ratio + coeffs[8]*heading*ratio**2 + coeffs[9]*ratio**3 \
           + coeffs[10]*heading**4 + coeffs[11]*heading**3*ratio + coeffs[12]*heading**2*ratio**2 + coeffs[13]*heading*ratio**3 + coeffs[14]*ratio**4 \
           # + coeffs[15]*heading**5 + coeffs[16]*heading**4*ratio + coeffs[17]*heading**3*ratio**2 + coeffs[18]*heading**2*ratio**3 + coeffs[19]*heading*ratio**4 + coeffs[20]*ratio**5 \
           # blank line required here
  perMeanHeading = evalPoly(PerMeanHeadCoeff)
  perMeanRatio = evalPoly(PerMeanRatioCoeff)
  perVarHeading = evalPoly(PerVarHeadCoeff)
  perVarRatio = evalPoly(PerVarRatioCoeff)
  perCorr = evalPoly(PerCorrCoeff)
  badPredictions = (perVarHeading < 0) | (perVarRatio < 0) | (perCorr < -.999) | (perCorr > .999)
  badCount = np.count_nonzero(badPredictions)
  if badCount > 0:
    print('WARNING:',badCount,'bad prediction(s)!')
  potBadPredictions = ((abs(heading) > np.pi/6.) | (abs(ratio) > .3*.76)) & (~badPredictions)
  potBadCount = np.count_nonzero(potBadPredictions)
  if potBadCount > 0:
    print('WARNING:',potBadCount,'potentially bad prediction(s)!')
  perMean = np.array([perMeanHeading, perMeanRatio])
  sz = perMeanHeading.size
  # L L^T = cov. mat
  perCoStd = np.array([[perVarHeading,np.zeros(sz)],[perCorr*perVarRatio,perVarRatio*np.sqrt(np.ones(sz)-perCorr**2)]])
  return perMean, perCoStd

'''
params is a 4 x N array
Parameter list:
0: initial heading
1: initial ratio
2: error in perception of heading
3: error in perception of ratio
'''
def RobotModel(params, gpcDistStatsPickleFile=None, gpcDistPickleFile=None):
  perceivedHeading, perceivedRatio = None, None
  if gpcDistPickleFile:
    standardDist = chaospy.Normal(0, 1)
    gpcPerceptionSamples = pickle.load(open(gpcDistPickleFile,'rb'))
    distributionMap = {}
    for truth, samples in gpcPerceptionSamples:
      truth = tuple(truth)
      samples = np.array(samples).T
      # create Gaussian KDE
      distribution = chaospy.GaussianKDE(samples)
      distributionMap[truth] = distribution
      print('Created Gaussian KDE for', truth, 'with', samples.shape[1], 'samples')
    perceivedHeading, perceivedRatio = np.empty(params.shape[1]), np.empty(params.shape[1])
    for idx in range(params.shape[1]):
      state = params[0:2,idx]
      _perceivedHeading, _perceivedRatio = None, None
      for truth, distribution in distributionMap.items():
        if np.allclose(state, np.array(truth)):
          _perceivedHeading, _perceivedRatio = distribution.inv(standardDist.fwd(params[2:4,idx]))
          break
      assert(_perceivedHeading is not None and _perceivedRatio is not None)
      print(_perceivedHeading, _perceivedRatio, idx, params.shape[1])
      perceivedHeading[idx], perceivedRatio[idx] = _perceivedHeading, _perceivedRatio
  elif gpcDistStatsPickleFile:
    perceptionData = pickle.load(open(gpcDistStatsPickleFile,'rb'))
    perceivedHeading, perceivedRatio = np.empty(params.shape[1]), np.empty(params.shape[1])
    for idx in range(params.shape[1]):
      state = params[0:2,idx]
      perceivedMean, perceivedCoStd = None, None
      for datum in perceptionData:
        if np.allclose(state, datum[0]):
          perceivedMean, perceivedCoStd = datum[1], datum[2]
          break
      assert(perceivedMean is not None and perceivedCoStd is not None)
      perceivedHeading[idx], perceivedRatio[idx] = perceivedCoStd @ params[2:4,idx] + perceivedMean
  else:
    # calc perceived state
    perceivedMean, perceivedCoStd = perceptionStateDepModel(params[0,:], params[1,:])
    perceivedHeading, perceivedRatio = np.einsum('ijk,jk->ik', perceivedCoStd, params[2:4,:]) + perceivedMean
  # calc error metric
  headingDiff = DesiredHeading - perceivedHeading
  ratioDiff = DesiredRatio - perceivedRatio
  error = headingDiff + np.arctan2(K*ratioDiff, Speed)
  # calc angular velocity
  angVel = error # / CycleTime
  # hard clip
  #angVel2 = angVel.copy()
  angVel = np.clip(angVel, AngVelMin, AngVelMax)
  # soft clip
  # angVel = AngVelMax * np.tanh(angVel/AngVelMax)
  # move robot
  newHeading = params[0,:] + CycleTime*angVel
  newRatio = params[1,:] + Speed*CycleTime*np.sin(params[0,:])
  # return new state
  return np.column_stack((newHeading, newRatio))

'''
params is a 2 x N array
Parameter list:
0: initial heading
1: initial ratio
2: error in perception of heading
3: error in perception of ratio
'''
def perceptionOnly(params):
  perceivedHeading, perceivedRatio = None, None
  perceivedMean, perceivedCoStd = perceptionStateDepModel(params[0,:], params[1,:])
  perceivedHeading, perceivedRatio = np.einsum('ijk,jk->ik', perceivedCoStd, params[2:4,:]) + perceivedMean
  return np.row_stack((perceivedHeading, perceivedRatio))

'''
params is a 4 x N array
Parameter list:
0: initial heading
1: initial ratio
2: perceived heading
3: perceived ratio
'''
def ctrlDynOnly(params):
  # calc error metric
  headingDiff = DesiredHeading - params[2,:]
  ratioDiff = DesiredRatio - params[3,:]
  error = headingDiff + np.arctan2(K*ratioDiff, Speed)
  # calc angular velocity
  angVel = error
  # hard clip
  angVel = np.clip(angVel, AngVelMin, AngVelMax)
  # move robot
  newHeading = params[0,:] + CycleTime*angVel
  newRatio = params[1,:] + Speed*CycleTime*np.sin(params[0,:])
  # return new state
  return np.column_stack((newHeading, newRatio))

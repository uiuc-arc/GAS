import chaospy
import numpy as np
import numpy.linalg as la
import pickle

from perceptionModel import *

class RobotModel:

  # constants
  look_ahead = 6 # meters
  wheelbase = 1.75 # meters
  dT = 0.1 # seconds
  speed = 2.8 # m/s
  maxSteer = .61 # radians
  kPP = .285
  kSt = .45

  '''
  params is a 4 x N array
  Parameter list:
  0: initial heading (measured w.r.t. positive x direction, positive -> counterclockwise)
  1: initial distance from centerline (positive distance -> towards outside of circle)
  2: error in heading
  3: error in distance
  radius is radius of centerline circle
  '''
  def relPerceptionModel(self, params):
    heading = params[0,:]
    distance = params[1,:]
    def evalPoly(coeffs):
      return coeffs[0] + coeffs[1]*heading + coeffs[2]*distance + coeffs[3]*heading**2 + coeffs[4]*heading*distance + coeffs[5]*distance**2 \
             + coeffs[6]*heading**3 + coeffs[7]*heading**2*distance + coeffs[8]*heading*distance**2 + coeffs[9]*distance**3 \
             + coeffs[10]*heading**4 + coeffs[11]*heading**3*distance + coeffs[12]*heading**2*distance**2 + coeffs[13]*heading*distance**3 + coeffs[14]*distance**4 \
             # + coeffs[15]*heading**5 + coeffs[16]*heading**4*distance + coeffs[17]*heading**3*distance**2 + coeffs[18]*heading**2*distance**3 + coeffs[19]*heading*distance**4 + coeffs[20]*distance**5 \
             # blank line required here
    perMeanHeading = evalPoly(PerMeanHeadCoeff)
    perMeanDistance = evalPoly(PerMeanDistanceCoeff)
    perVarHeading = evalPoly(PerVarHeadCoeff)
    perVarDistance = evalPoly(PerVarDistanceCoeff)
    perCorr = evalPoly(PerCorrCoeff)
    badPredictions = (perVarHeading < 0) | (perVarDistance < 0) | (perCorr < -.999) | (perCorr > .999)
    badCount = np.count_nonzero(badPredictions)
    if badCount > 0:
      print('WARNING:',badCount,'bad prediction(s)!')
      perVarHeading = np.clip(perVarHeading, 0, None)
      perVarDistance = np.clip(perVarDistance, 0, None)
      perCorr = np.clip(perCorr, -.999, .999)
    potBadPredictions = ((abs(heading) > np.pi/12.) | (abs(distance) > 1.2)) & (~badPredictions)
    potBadCount = np.count_nonzero(potBadPredictions)
    if potBadCount > 0:
      print('WARNING:',potBadCount,'potentially bad prediction(s)!')
    perMean = np.array([perMeanHeading, perMeanDistance])
    sz = perMeanHeading.size
    # L L^T = cov. mat
    perCoStd = np.array([[perVarHeading,np.zeros(sz)],[perCorr*perVarDistance,perVarDistance*np.sqrt(np.ones(sz)-perCorr**2)]])
    perceivedHeading, perceivedDistance = np.einsum('ijk,jk->ik', perCoStd, params[2:4,:]) + perMean
    return (heading, distance, perceivedHeading, perceivedDistance)

  def pursuitCtrl(self, heading, distance, radius):
    # calculate steering angle
    if radius == np.inf:
      sinBeta = distance/RobotModel.look_ahead
    else:
      sinBeta = (2*radius*distance - distance**2 - RobotModel.look_ahead**2)/(2*RobotModel.look_ahead*(radius - distance))
    beta = np.where(abs(distance) <= RobotModel.look_ahead, np.arcsin(sinBeta), np.pi/2)
    alpha = - beta - heading
    angle_i = np.arctan((2*RobotModel.kPP*RobotModel.wheelbase*np.sin(alpha))/RobotModel.look_ahead)
    angle = angle_i*2
    return angle

  def stanleyCtrl(self, heading, distance, radius):
    # simplifying assumption: assume infinite radius (straight road)
    assert(radius == np.inf)
    # calculate steering angle
    distance_front = distance + np.sin(heading) * RobotModel.wheelbase
    angle = -heading - np.arctan2(RobotModel.kSt*distance_front,RobotModel.speed)
    return angle

  def relModel(self, params, radius, controller):
    # unpack params
    trueHeading, trueDistance, heading, distance = self.relPerceptionModel(params)
    # use specified controller
    if controller == 'pursuit':
      angle = self.pursuitCtrl(heading, distance, radius)
    elif controller == 'stanley':
      angle = self.stanleyCtrl(heading, distance, radius)
    else:
      raise Exception('Bad controller!')
    # clip steering angle
    angle = np.clip(angle, -RobotModel.maxSteer, RobotModel.maxSteer)
    # move
    newDistance = trueDistance + np.sin(trueHeading) * RobotModel.speed * self.dT
    newHeading = trueHeading + (np.tan(angle)/RobotModel.wheelbase - np.cos(trueHeading)/(radius - trueDistance)) * RobotModel.speed * self.dT
    # return new state
    return np.column_stack((newHeading,newDistance))

  '''
  params is a 2 x N array
  Parameter list:
  0: initial heading
  1: initial distance
  2: error in perception of heading
  3: error in perception of distance
  '''
  def perceptionOnly(self, params):
    trueHeading, trueDistance, heading, distance = self.relPerceptionModel(params)
    return np.row_stack((heading, distance))

  '''
  params is a 4 x N array
  Parameter list:
  0: initial heading
  1: initial distance
  2: perceived heading
  3: perceived distance
  '''
  def ctrlDynOnly(self, params, radius):
    angle = self.pursuitCtrl(params[2,:], params[3,:], radius)
    # clip steering angle
    angle = np.clip(angle, -RobotModel.maxSteer, RobotModel.maxSteer)
    # move
    newDistance = params[1,:] + np.sin(params[0,:]) * RobotModel.speed * self.dT
    newHeading = params[0,:] + (np.tan(angle)/RobotModel.wheelbase - np.cos(params[0,:])/(radius - params[1,:])) * RobotModel.speed * self.dT
    # return new state
    return np.column_stack((newHeading,newDistance))

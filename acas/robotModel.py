import chaospy
import h5py
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier

from readNN import readNNet

class RobotModel:

  # constants
  # taus = np.array([0,5,10,15,20,30,40,60]) # seconds
  taus = np.array([0]) # seconds
  tablefileName = 'ACASXuTables/HCAS_rect_TrainingData_v6_pra{}_tau{:02}.h5'
  nnfileName = 'ACASXuNNs/HCAS_rect_v6_pra{}_tau{:02}_25HU_3000.nnet'
  bothVel = 200 # ft/s
  turnRates = np.array([0, 1.5, -1.5, 3, -3])*np.pi/180 # rad/s
  NInputPos = 32*41
  NInputStates = NInputPos*41
  dT = .1 # seconds

  controlTime = 0.
  dynamicsTime = 0.

  def __init__(self):
    # read input state
    inFile = h5py.File(self.tablefileName.format(0,0),'r')
    self.InputState = np.array(inFile['X'])
    # de-normalize input state
    self.inpMeans = np.array(inFile['means'])[:3]
    self.inpRanges = np.array(inFile['ranges'])[:3]
    self.InputState = self.InputState*self.inpRanges + self.inpMeans
    inFile.close()
    # seperate x,y and psi of input state
    self.InputPos = self.InputState[:self.NInputPos,:2]
    self.InputPsi = self.InputState[::self.NInputPos,2]
    # read output scores
    self.Score = np.empty((len(self.taus),5,self.NInputStates,5))
    for tauIdx, tau in enumerate(self.taus):
      for pra in range(5):
        inFile = h5py.File(self.tablefileName.format(pra,tau),'r')
        self.Score[tauIdx,pra,:,:] = np.array(inFile['y'])
        inFile.close()
    # read NNs
    self.NN = np.empty((len(self.taus),5),dtype=object)
    for tauIdx, tau in enumerate(self.taus):
      for pra in range(5):
        weights, biases, _, _, _, _ = readNNet(self.nnfileName.format(pra,tau))
        self.NN[tauIdx,pra] = (weights, biases)

  # cartesian to polar for perception model
  # warning: overwrites original
  def cartToPolar(self, states):
    r = np.sqrt(states[:,0]**2 + states[:,1]**2)
    theta = np.arctan2(states[:,1], states[:,0])
    states[:,0] = r
    states[:,1] = theta

  # construct classifier perception model
  def constructPerceptionModel(self):
    # prepare training data inputs
    X = np.empty((self.NInputStates*5,4))
    X[:,:3] = np.tile(self.InputState,(5,1))
    X[:,3] = np.repeat(np.arange(5),self.NInputStates)
    # self.cartToPolar(X)
    # prepare training data outputs
    Y = np.argmax(np.reshape(self.Score[0,:,:,:],(self.NInputStates*5,5)), axis=1)
    # fit classifier
    # self.classifier = MLPClassifier(hidden_layer_sizes=(10), max_iter=10, random_state=0x60F154)
    self.classifier = DecisionTreeClassifier(max_depth=10)
    # self.classifier = GaussianNB()
    self.classifier.fit(X, Y)
    score = self.classifier.score(X, Y)
    print('Classifier score:', score)

  def fastClassify(self, params):
    states = params[:4,:].T.copy()
    # self.cartToPolar(states)
    predictions = self.classifier.predict(states)
    return predictions

  '''
  params is a 5 x N array
  Parameter list:
  0: intruder X
  1: intruder Y
  2: intruder relative heading (psi)
  3: previous advisory
  4: tau (time to loss of vertical seperation)
  inertialParams is 6 x N array
  Parameter list:
  0: self X
  1: self Y
  2: self heading
  3: intruder X
  4: intruder Y
  5: intruder heading
  '''
  def model(self, params, inertialParams=None, useNN=False):
    NSamp = params.shape[1]
    outputs = np.empty((NSamp,3))
    if inertialParams is not None:
      inertialOutputs = np.empty((6,NSamp))
      inertialOutputs[3,:] = inertialParams[3,:] - self.bothVel * self.dT * np.sin(inertialParams[5,:])
      inertialOutputs[4,:] = inertialParams[4,:] + self.bothVel * self.dT * np.cos(inertialParams[5,:])
      inertialOutputs[5,:] = inertialParams[5,:]
    advisories = np.empty(NSamp, dtype=int)
    for i in range(NSamp):
      startTime = time.time()
      # find closest tau in database
      tauDist = abs(self.taus - params[4,i])
      minTauDist = np.argmin(tauDist)
      if useNN:
        weights, biases = self.NN[minTauDist,params[3,i].astype(int)]
        temp = params[:3,i]
        for layer in range(6):
          temp = weights[layer] @ temp + biases[layer]
          if layer < 5: # no ReLU on output layer
            temp = np.clip(temp, 0, None)
        advisory = np.argmax(temp)
      else:
        # find closest input position in database
        dist = np.linalg.norm(self.InputPos-params[:2,i], ord=2, axis=1)
        minDistIdx = np.argmin(dist)
        # find closest psi in database
        angDist = abs(self.InputPsi - params[2,i])
        minAngDistIdx = np.argmin(angDist)
        # get scores and best policy
        score = self.Score[minTauDist, params[3,i].astype(int), minDistIdx+minAngDistIdx*self.NInputPos, :]
        advisory = np.argmax(score)
      # get turn rate
      turnRate = self.turnRates[advisory]
      self.controlTime += time.time() - startTime
      startTime = time.time()
      # move and get new state
      # need to do affine transform
      dSelfHeading = self.dT * turnRate
      cosDSH = np.cos(dSelfHeading)
      sinDSH = np.sin(dSelfHeading)
      newX = params[0,i] - self.bothVel * self.dT * np.sin(params[2,i])
      newY = params[1,i] + self.bothVel * self.dT * np.cos(params[2,i])
      if turnRate:
        newX += self.bothVel / turnRate * (1 - cosDSH)
        newY -= self.bothVel / turnRate * sinDSH
      else:
        newY -= self.bothVel * self.dT
      newXRot = newX * cosDSH + newY * sinDSH
      newYRot = newX * -sinDSH + newY * cosDSH
      newPsi = params[2,i] - dSelfHeading
      outputs[i,:] = newXRot, newYRot, newPsi
      advisories[i] = advisory
      if inertialParams is not None:
        if turnRate:
          inertialOutputs[0,i] = inertialParams[0,i] + self.bothVel / turnRate * (np.cos(inertialParams[2,i] + dSelfHeading) - np.cos(inertialParams[2,i]))
          inertialOutputs[1,i] = inertialParams[1,i] + self.bothVel / turnRate * (np.sin(inertialParams[2,i] + dSelfHeading) - np.sin(inertialParams[2,i]))
        else:
          inertialOutputs[0,i] = inertialParams[0,i] - self.bothVel * self.dT * np.sin(inertialParams[2,i])
          inertialOutputs[1,i] = inertialParams[1,i] + self.bothVel * self.dT * np.cos(inertialParams[2,i])
        inertialOutputs[2,i] = inertialParams[2,i] + dSelfHeading
      self.dynamicsTime += time.time() - startTime
    if inertialParams is not None:
      return outputs, advisories, inertialOutputs
    else:
      return outputs, advisories

  def modelWithAncillary(self, params):
    NSamp = params.shape[1]
    outputs = np.empty((NSamp,3))
    advisories = np.empty(NSamp, dtype=int)
    tauDist = np.abs(np.subtract.outer(self.taus, params[4,:]))
    minTauDist = np.argmin(tauDist, axis=0)
    advisory = self.fastClassify(params)
    # get turn rate
    turnRate = self.turnRates[advisory]
    # move and get new state
    # need to do affine transform
    dSelfHeading = self.dT * turnRate
    cosDSH = np.cos(dSelfHeading)
    sinDSH = np.sin(dSelfHeading)
    newX = params[0,:] - self.bothVel * self.dT * np.sin(params[2,:])
    newY = params[1,:] + self.bothVel * self.dT * np.cos(params[2,:])
    isTurning = (turnRate != 0)
    newX[isTurning] += self.bothVel / turnRate[isTurning] * (1 - cosDSH[isTurning])
    newY[isTurning] -= self.bothVel / turnRate[isTurning] * sinDSH[isTurning]
    newY[~isTurning] -= self.bothVel * self.dT
    newXRot = newX * cosDSH + newY * sinDSH
    newYRot = newX * -sinDSH + newY * cosDSH
    newPsi = params[2,:] - dSelfHeading
    outputs[:,0], outputs[:,1], outputs[:,2] = newXRot, newYRot, newPsi
    advisories = advisory
    return outputs, advisory


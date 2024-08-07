#! /usr/bin/env python3

import chaospy
import numpy as np
import time

order = 4
dimensions = 4
crossTruncation = 1

# order >= sum(K**(1/crossTruncation))**crossTruncation

def model(params):
  return np.sum(params, axis=0)

print('Generating GPC approximation with following settings:')
print(f'Order: {order:2d}  Dims: {dimensions:2d}  crossTruncation: {crossTruncation:4.2f}')

timeA = time.time()

print('Creating dist', flush=True)
dist = chaospy.Iid(chaospy.Normal(0, 1), dimensions)

timeB = time.time()

print('Creating expansion', flush=True)
orthoExpansion = chaospy.generate_expansion(order, dist, cross_truncation=crossTruncation)

timeC = time.time()

print('Creating quadrature nodes', flush=True)
samples, weights = chaospy.generate_quadrature(order, dist, rule='gaussian')

timeD = time.time()

print('Evaluating samples', flush=True)
evals = model(samples)

timeE = time.time()

print('Fitting approx', flush=True)
approxModel = chaospy.fit_quadrature(orthoExpansion, samples, weights, evals)

timeF = time.time()

print('Time statistics:')
print('Distribution : {:8.3f}'.format(timeB-timeA))
print('Expansion    : {:8.3f}'.format(timeC-timeB))
print('Nodes+Weights: {:8.3f}'.format(timeD-timeC))
print('Evaluation   : {:8.3f}'.format(timeE-timeD))
print('Approximation: {:8.3f}'.format(timeF-timeE))
print('TOTAL        : {:8.3f}'.format(timeF-timeA))

print('Polynomial statistics:')
print('Expansion/Model size: {:8d}'.format(len(orthoExpansion)))
print('Nodes size          : {:8d}'.format(samples.shape[1]))

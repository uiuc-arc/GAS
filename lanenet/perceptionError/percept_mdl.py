#!/usr/bin/env python3

from itertools import islice
import pickle
from typing import Any

import numpy as np
from matplotlib import transforms as transforms, pyplot as plt
from matplotlib.patches import Ellipse

from sklearn.linear_model import LinearRegression

from scipy.linalg import sqrtm
from scipy.stats import ks_2samp
import chaospy


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    See https://matplotlib.org/gallery/statistics/confidence_ellipse.html

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the square-root of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def confidence_ellipse_gaussian(mean: np.ndarray, cov: np.ndarray, ax, n_std=3.0, facecolor='none', **kwargs):
    mean_x, mean_y = mean
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_samples(truth_samples_seq, regressors, nrows=5, ncols=5):
    fig, axs = plt.subplots(nrows, ncols)
    fig.suptitle("Red dot is ground truth.\n"
                 "Yellow dot and ellipse is observed mean and 3*cov by the output of NN.\n"
                 "Purple dot and ellipse is mean and estimated 3*cov of the fitted Gaussian"
                 " by Polynomial Regression.\n"
                 "x is heading angle (deg). "
                 "y is distance to lane center (meter).")
    llrs=[]
    for idx, (truth, est_list) in enumerate(truth_samples_seq):
        print("Truth: (%.2f, %.2f);" % truth, "#Samples: %d" % len(est_list))
        i, j = divmod(idx, nrows)
        ax = axs[i][j]
        ax.set_xlim(-np.pi/10, np.pi/10)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(truth[0], truth[1], 'ro')

        # Plot samples
        if est_list == []:
          est_list = [(np.nan, np.nan)]
        obs_y_arr = np.array(est_list)
        idxNaN = np.isnan(obs_y_arr[:,0]) | np.isnan(obs_y_arr[:,1])
        obs_y_arr = obs_y_arr[~idxNaN,:]
        if obs_y_arr.shape[0] > 0:
          ax.scatter(obs_y_arr[:, 0], obs_y_arr[:, 1], s=1.0)
          obs_mean = np.mean(obs_y_arr, axis=0)
          ax.plot(obs_mean[0], obs_mean[1], 'yo')
        if obs_y_arr.shape[0] >= 5:
          confidence_ellipse(obs_y_arr[:, 0], obs_y_arr[:, 1], ax, edgecolor='y')

        # Plot regression model
        truth = np.array(truth)
        extTruth = extendTruth(truth)
        x_arr = np.array([extTruth])
        mean = np.array([regressors[0].predict(x_arr)[0], regressors[1].predict(x_arr)[0]])
        sigma0, sigma1 = regressors[2].predict(x_arr)[0], regressors[3].predict(x_arr)[0]
        correlation = regressors[4].predict(x_arr)[0]
        if inbounds(truth) and (sigma0 < 0 or sigma1 < 0 or correlation <= -.999 or correlation >= .999):
            print('WARNING: bad prediction:', sigma0, sigma1, correlation)
            sigma0 = np.clip(sigma0, 0, None)
            sigma1 = np.clip(sigma1, 0, None)
            correlation = np.clip(correlation, -.999, .999)
        cov = np.array([[sigma0**2, sigma0*sigma1*correlation],[sigma0*sigma1*correlation, sigma1**2]])
        ax.plot(mean[0], mean[1], 'mo')
        confidence_ellipse_gaussian(mean, cov, ax, edgecolor='m')

        if inbounds(truth):
            mvDist = chaospy.MvNormal(mean, cov)
            predictSamp = mvDist.sample(1000, rule='sobol')
            predictLL = np.mean(np.log(mvDist.pdf(predictSamp)))
            obsLL = np.mean(np.log(mvDist.pdf(obs_y_arr.T)))
            llr = obsLL-predictLL
            if not (np.isinf(llr)):
                llrs.append(llr)
            print('llr:', llr)

    llrs = np.array(llrs)
    minllr, maxllr = np.min(llrs), np.max(llrs)
    print('minllr:', minllr, 'maxllr:', maxllr)
    medianllr, meanllr, stdllr = np.median(llrs), np.mean(llrs), np.std(llrs)
    print('medianllr:', medianllr, 'meanllr:', meanllr, 'stdllr:', stdllr)
    print(np.array2string(llrs, separator=','))

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('model.png')
    plt.close()

# check if inbounds
def inbounds(truth):
    # below based on line beyond which lanenet output became erratic
    # line passing through (pi/12-3.5pi/60, 1.2) and (pi/12, 1.2-6.5*.24)
    heading, distance = truth
    return abs(heading) <= np.pi/12. and abs(distance) <= 1.2 and abs(distance + 8.51252*heading) <= 1.86857

# extend truth for quadriatic regression
def extendTruth(truth):
    extTruth = np.array([truth[0], truth[1] , \
                         truth[0]**2, truth[0]*truth[1], truth[1]**2, \
                         truth[0]**3, truth[0]**2*truth[1], truth[0]*truth[1]**2, truth[1]**3, \
                         truth[0]**4, truth[0]**3*truth[1], truth[0]**2*truth[1]**2, truth[0]*truth[1]**3, truth[1]**4, \
                         # truth[0]**5, truth[0]**4*truth[1], truth[0]**3*truth[1]**2, truth[0]**2*truth[1]**3, truth[0]*truth[1]**4, truth[1]**5, \
                         ])
    return extTruth
    # return truth

def key_for_subplots(entry):
    (heading, dist), _ = entry
    return -dist, heading

def main(argv: Any) -> None:
    for pickle_file_io in argv.pickle_file:
        truth_samples_seq = pickle.load(pickle_file_io)
        truth_samples_seq.sort(key=key_for_subplots)

        x_list = []
        y_list = [[], [], [], [], []] # mean0, mean1, sigma0, sigma1, correlation
        NaN_x_list, NaN_y_list = [], []

        for idx, (truth, samples) in enumerate(truth_samples_seq):
            # don't train on test dataset
            # if idx%2==1:
            #     continue
            row, col = idx//11, idx%11
            # removed = [1, 4, 6, 9] # 7x7
            # removed = [2, 8] # 9x9
            removed = [] # 11x11
            if row in removed or col in removed:
              continue
            if samples == []:
              samples = [(np.nan, np.nan)]
            truth = np.array(truth)
            extTruth = extendTruth(truth)
            samples = np.array(samples)
            # remove NaNs if any
            idxNaN = np.isnan(samples[:,0]) | np.isnan(samples[:,1])
            NaN_x_list.append(extTruth)
            NaN_y_list.append(np.count_nonzero(idxNaN))
            samples = samples[~idxNaN,:]
            if samples.shape[0] < 30:
              print('Skipping samp',truth,samples.shape[0])
              continue
            if not inbounds(truth):
              print('Skipping OOB',truth)
              continue
            # samples = samples[:100,:]
            sampMean = np.mean(samples, axis=0)
            y_list[0].append(sampMean[0])
            y_list[1].append(sampMean[1])
            sampCov = np.cov(samples, rowvar=0)
            y_list[2].append(np.sqrt(sampCov[0,0]))
            y_list[3].append(np.sqrt(sampCov[1,1]))
            y_list[4].append(sampCov[0,1]/np.sqrt(sampCov[0,0]*sampCov[1,1]))
            x_list.append(extTruth)

        regressors = []
        names = ['PerMeanHeadCoeff', 'PerMeanDistanceCoeff', 'PerVarHeadCoeff', 'PerVarDistanceCoeff', 'PerCorrCoeff', 'PerNaNCoeff']
        for i in range(5):
            regressor = LinearRegression(fit_intercept=True, copy_X=True)
            regressor.fit(x_list, y_list[i])
            regressors.append(regressor)
            coeffs = np.insert(regressor.coef_, 0, regressor.intercept_)
            print(names[i], '= ', end='')
            print(np.array2string(coeffs, separator=',', max_line_width=100))
        regressor = LinearRegression(fit_intercept=True, copy_X=True)
        regressor.fit(NaN_x_list, NaN_y_list)
        regressors.append(regressor)
        coeffs = np.insert(regressor.coef_, 0, regressor.intercept_)
        print(names[5], '= ', end='')
        print(np.array2string(coeffs, separator=',', max_line_width=100))
        for i in range(5):
            print(regressors[i].score(x_list, y_list[i]))
        print(regressors[5].score(NaN_x_list, NaN_y_list))

        # Select some ground truths for pretty plots
        if argv.plot:
            nrows, ncols = 11,11
            plot_samples(truth_samples_seq, regressors, nrows, ncols)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('pickle_file', nargs='+', type=argparse.FileType('rb'))
    parser.add_argument('-p', '--plot', action='store_true', help="plot samples of each ground truth")
    parser.add_argument('-o', '--output', type=argparse.FileType('wb'), help="Save model to output as pickle file")
    main(parser.parse_args())


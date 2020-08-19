import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pytoolsMH as ptMH
import pandas as pd
import seaborn as sns
import os,sys
import scipy.io
import scipy.stats as ss
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.optimize
from argparse import Namespace

from mworksbehavior import mat_io
from mworksbehavior import psychfun
from pytoolsMH import dataio

r_ = np.r_


def piecewise_predict(x, xs):
    """This is the predict function
    x: array, (breakpt,slope2)"""
    (breakptX,slope2) = x
    lowIx = xs<breakptX
    assert np.all(np.diff(xs)>=0), 'must be monotonically increasing - add sorting'
    ys = np.hstack((xs[lowIx]*0, (xs[~lowIx]-breakptX)*slope2))
    
    return ys

def piecewise_fun(x,xs,ys):
    """This is the residual function for fitting"""
    ysHat = piecewise_predict(x,xs)
    resid = (ysHat-ys)
    return resid
     
    
def regress_piecewise(logxv, yv, fixSlopeAt=None, fitType='hitrate'):
    """Function for doing the regression one time.  
    Args:
       - fitType: string 'hitrate' or 'earlies' - needs to know to compute x0 and bounds
    Notes:
        - logxv here should be in log units, fitX is in log units
        - our choice of x0 is pretty specific to the purposes of these data.  
           breakpoint can't be estimated with gradient methods if guess too high, so just brute-force subtract 0.5 log units"""
    if fitType == 'hitrate':
        x0 = r_[np.mean(logxv)-0.5, np.max(yv)*5]  #np.mean(logxv)-3.0
        lb = r_[np.log10(0.001), 100]
        ub = r_[np.log10(10), 3000]
    elif fitType == 'earlies':
        x0 = r_[np.mean(logxv)-0.5, np.max(yv)*25]
        lb = r_[np.log10(0.001), 100]
        ub = r_[np.log10(10), 3000]
    else:
        raise RuntimeError('Unknown fitType %s' % fitType)
        

    x_scale = r_[1, 50]

    if fixSlopeAt is not None:
        x0 = x0[0:1]; lb=lb[0:1]; ub=ub[0:1]; x_scale=x_scale[0:1]
        def fit_fun(x, *args, **kwargs):
            return piecewise_fun([x,fixSlopeAt], *args, **kwargs)
    else:
        fit_fun = piecewise_fun


    try:
        res = scipy.optimize.least_squares(fit_fun, x0, bounds=(lb,ub),
                                           args=(logxv,yv), x_scale=x_scale, # jac='3-point',
                                           xtol=1e-30, verbose=0) # loss='soft_l1')
    except ValueError as e:
        if str(e).find("`x0` is infeasible.") > -1:
            res = Namespace(x=r_[np.nan, np.nan])
        else:
            raise
    
    #print(x0, res.x)
    # global optimizers don't work well, I didn't spend time debugging
    #res = scipy.optimize.basinhopping(piecewise_fun, x0, )
    #res = scipy.optimize.differential_evolution(piecewise_fun, bounds=(lb,ub), args=(logxv,yv))
    fitX = res.x
    if fixSlopeAt is not None:
        fitX = [res.x[0], fixSlopeAt]
    return res, fitX

def recompute_yshighlow_for_fixedslope(fitRow, xsLim=r_[0.02,10]):
    """
    xsLim: x limits, in real (not log) units
    fitRow: Series, output of bootstrap_regress_piecewise()
    """
    nPts = 100

    tR = fitRow.copy()
    tR.xsPred = np.linspace(np.log10(xsLim[0]), np.log10(xsLim[1]), nPts)
    tR.ysPred = piecewise_predict(tR.fullFitX, tR.xsPred)
    tR.ysHigh = piecewise_predict(tR.paramHigh, tR.xsPred)
    tR.ysLow = piecewise_predict(tR.paramLow, tR.xsPred)
    return(tR)



def fig_data_and_fit(xv, yv, fitX, xsPred=None, ysLow=None, ysHigh=None, pctVarThresh=50, fitRow=None, crossingPt = 100):
    """xv is NOT log-scaled (only data in and for plotting do we not use log units).
    fitX: orig/full fit params. IS logscaled.
    xsPred if not None IS log-scaled.
    Kind of confusing but I'm not going to fix it now."""
    (fig,ax) = plt.subplots()
    ax.plot(xv, yv, 'k.', label='data pts')
    yLim = r_[-100,250]
    xLim = r_[0.02, 10]

    if fitRow is None:
        raise RuntimeError('Must pass fit output row into this fn now, normally from b_r_p() or saved df 200213 MH (bootOut)')

    crossX = fitRow.crossX
    crossLims = r_[fitRow.crossLow,fitRow.crossHigh]
    crossLims = np.sort(crossLims) # bug in fitting can swap these

    if xsPred is None:
        xsPred = np.linspace(np.log10(0.01),np.log10(10),100) # log units
    ysPred = piecewise_predict(fitX, xsPred)
    ax.plot(10**xsPred,ysPred, label='data fit', color = '#0f3773')

    if ysLow is not None:
        fH = ax.fill_between(10**xsPred, ysLow, ysHigh, facecolor = "#0f37734D", label=r'95% CI', edgecolor = "#0f377300")
        #fillC = fH.get_facecolor()
    
    # fill all above a value if pctVarThreshold is set 
    if pctVarThresh is not None:
        if fitRow.pctVarExpRelToMeanZero < pctVarThresh:
            xsP = xsPred
            ysP1 = ysHigh
            ysP2 = xsP*0
            ysP2[-1] = yLim[-1]
            ax.fill_between(10**xsP, ysP1, ysP2, label=r'95% CI', facecolor="#0f37734D", edgecolor = "#0f377300")
            crossLims[1] = 1.3 # set high crossing to max on xlim

    # add the 100% crossing bar; color=sns.xkcd_rgb['reddish pink']
    if fitRow is not None:
        plt.plot(10**crossLims, crossingPt*r_[1,1], color= '#ff3079',
                 lw=5, alpha=0.8, solid_capstyle = 'butt')
        plt.plot(10**crossX*r_[1,1], [0,crossingPt], color= '#ff3079', lw=1,
                 ls='--')
    
    ax.set_xscale('log')
    ax.set_xlim(xLim)
    ax.set_ylim(yLim)
    ax.axhspan(-100,0, color='0.5', alpha=0.1)
    ax.set_xlabel(r'intens. (mW/mm$^2$)', fontsize = 8, labelpad = 1)
    ax.set_ylabel('Thresh change (%)', fontsize = 8, labelpad = 1)
    ax.set_yticks((-100, 0, 100, 200))
    ax.set_yticklabels((-100, 0, 100, 200), fontsize = 7)
    ax.set_xticks((0.1, 1.0, 10.0))
    ax.set_xticklabels((0.1, 1.0, 100), fontsize = 7)
    ax.tick_params(pad = 0.05)
    plt.legend()

    # fixup x labels to be non-exponential form
    oldformatter = ax.xaxis.get_major_formatter()
    def locf(loc,pos):
        return '%g'%loc
    ax.xaxis.set_major_formatter(plt.FuncFormatter(locf))

    return fig

def FA_fig_data_and_fit(xv, yv, fitX, xsPred=None, ysLow=None, ysHigh=None, pctVarThresh=50, fitRow=None):
    """xv is NOT log-scaled (only data in and for plotting do we not use log units).
    fitX: orig/full fit params. IS logscaled.
    xsPred if not None IS log-scaled.
    Kind of confusing but I'm not going to fix it now."""
    (fig,ax) = plt.subplots()
    ax.plot(xv, yv, 'k.', label='data pts')
    yLim = r_[-100,250]
    xLim = r_[0.02, 10]

    if fitRow is None:
        raise RuntimeError('Must pass fit output row into this fn now, normally from b_r_p() or saved df 200213 MH (bootOut)')

    #crossX = fitRow.crossX
    #crossLims = r_[fitRow.crossLow,fitRow.crossHigh]
    #crossLims = np.sort(crossLims) # bug in fitting can swap these

    #if xsPred is None:
        #xsPred = np.linspace(np.log10(0.01),np.log10(10),100) # log units
    #ysPred = piecewise_predict(fitX, xsPred)
    #ax.plot(10**xsPred,ysPred, label='data fit', color = '#0f3773')

    #if ysLow is not None:
        #fH = ax.fill_between(10**xsPred, ysLow, ysHigh, facecolor = "#0f37734D", label=r'95% CI', edgecolor = "#0f377300")
        #fillC = fH.get_facecolor()
    
    ax.set_xscale('log')
    ax.set_xlim(xLim)
    ax.set_ylim(yLim)
    ax.axhspan(-100,0, color='0.5', alpha=0.1)
    ax.set_xlabel(r'intens. (mW/mm$^2$)', fontsize = 8, labelpad = 1)
    ax.set_ylabel('Thresh change (%)', fontsize = 8, labelpad = 1)
    ax.set_yticks((-100, 0, 100, 200))
    ax.set_yticklabels((-100, 0, 100, 200), fontsize = 7)
    ax.set_xticks((0.1, 1.0, 10.0))
    ax.set_xticklabels((0.1, 1.0, 100), fontsize = 7)
    ax.tick_params(pad = 0.05)
    plt.legend()

    # fixup x labels to be non-exponential form
    oldformatter = ax.xaxis.get_major_formatter()
    def locf(loc,pos):
        return '%g'%loc
    ax.xaxis.set_major_formatter(plt.FuncFormatter(locf))

    return fig




def bootstrap_regress_piecewise(logxv, yv, nReps=1000, ci=95, xsPred=None, fixSlopeAt=None, 
                                doDebugPlots=False, fitType='hitrate', crossingPt = 100):
    """The function that does all the fit work.
    Args:
        fitType: see regress_piecewise
    Note: first param must be log-scaled.
    Also does regular regression and returns results as a nice Series with named indices
    crossX, crossLow, crossHigh: the point at which the three series cross crossingPoint (usually 100)
    """
    crossingPt = crossingPt
    nPts = len(logxv)
    assert len(yv) == nPts

    if xsPred is None:
        xsPred = np.linspace(np.log10(0.01),np.log10(10),100)

    res, fitX = regress_piecewise(logxv, yv, fixSlopeAt=fixSlopeAt, fitType=fitType)

    outYPredM = np.zeros((nReps, len(xsPred)))*np.nan
    outParamM = np.zeros((nReps,2))*np.nan

    for iR in range(nReps):
        tDataN = np.random.choice(r_[0:nPts], size=np.shape(logxv), replace=True)
        eps = np.random.standard_normal(size=np.shape(logxv))*1e-1
        bLogXv = logxv[tDataN] 
        bYv = yv[tDataN]+eps

        bLogXv = bLogXv + np.random.standard_normal(size=np.shape(logxv))*1e-4
        sortNs = np.argsort(bLogXv)
        bLogXv = bLogXv[sortNs]
        bYv = bYv[sortNs]        

        tRes, tX = regress_piecewise(bLogXv, bYv, fixSlopeAt=fixSlopeAt, fitType=fitType)
        tYHat = piecewise_predict(tX, xsPred)
        outYPredM[iR,:] = tYHat
        outParamM[iR,:] = tX

        if doDebugPlots:
            assert nReps<=10, 'do you really want this many bootstrap plots?'
            fig_data_and_fit(10**bLogXv, bYv, tX)
        
    # now collect
    lowp = (100-ci)/2
    highp = 100-lowp
    nanIx = np.any(np.isnan(outParamM),axis=1)
    fractNan = np.sum(nanIx) / len(nanIx)
    if fractNan > 0.1:
        print('More than 10% nan, skipping CIs')
    elif fractNan != 0:
        print(f'Nans found, {fractNan*100:.3g} pct of total, continuing')
        outYPredM = outYPredM[~nanIx,:]
        outParamM = outParamM[~nanIx,:]        

    ysLow = np.percentile(outYPredM, q=lowp, axis=0)
    ysHigh = np.percentile(outYPredM, q=highp, axis=0)
    paramLow = np.percentile(outParamM, q=lowp, axis=0)
    paramHigh = np.percentile(outParamM, q=highp, axis=0)

    # extract crossing of 100
    ysPred = piecewise_predict(fitX, xsPred)
    ysN = np.argmin(np.abs(ysPred-crossingPt))
    yLN = np.argmin(np.abs(ysLow-crossingPt))
    yHN = np.argmin(np.abs(ysHigh-crossingPt))

    ssAll = np.sum(yv**2)  # line at offset 0 is this model
    residModel = piecewise_fun(fitX, logxv, yv)
    ssModel = np.sum(residModel ** 2)
    pctVar = (1 - (ssModel/ssAll)) * 100

    outObj = pd.Series({'xsPred':xsPred, 'ysLow': ysLow, 'ysHigh': ysHigh, 'paramLow': paramLow, 'paramHigh': paramHigh,
                        'ci': ci, 'fullFitRes': res, 'fullFitX': fitX, 'crossX': xsPred[ysN], 'crossLow': xsPred[yLN], 'crossHigh': xsPred[yHN],
                        'pctVarExpRelToMeanZero': pctVar})

    return outObj


def fig_paper_fixup(fig, pdfOutName=None):
    ax = fig.findobj(lambda x: hasattr(x, 'xaxis'))[0]  # find the axis in the plot
    ax.set_xlim([0.05,30])
    leg = fig.findobj(lambda x: type(x) is mpl.legend.Legend)[0]
    leg.set_visible(False)
    fig.set_size_inches(2.5, 2.0)
    ax.set_xticks(r_[0.1,1,10])
    ax.grid(True, color = '#F1F1F1')
    fig.tight_layout()
    fig.savefig(pdfOutName, dpi = 300)
   
    
def fig_paper_fixupIns(fig, pdfOutName=None):
    ax = fig.findobj(lambda x: hasattr(x, 'xaxis'))[0]  # find the axis in the plot
    ax.set_xlim([0.05,30])
    leg = fig.findobj(lambda x: type(x) is mpl.legend.Legend)[0]
    leg.set_visible(False)
    fig.set_size_inches(2.5, 2.25)
    ax.set_xticks(r_[0.1,1,10])
    ax.grid(True, color = '#F1F1F1')
    fig.tight_layout()
    fig.savefig(pdfOutName, dpi = 300)
    
def FA_fig_paper_fixup(fig, pdfOutName=None):
    ax = fig.findobj(lambda x: hasattr(x, 'xaxis'))[0]  # find the axis in the plot
    ax.set_xlim([0.05,30])
    ax.set_ylim(-25, 50)
    leg = fig.findobj(lambda x: type(x) is mpl.legend.Legend)[0]
    leg.set_visible(False)
    fig.set_size_inches(2.5, 2.0)
    ax.set_xticks(r_[0.1,1,10])
    ax.set_yticks([-25, 0, 25, 50])
    ax.set_yticklabels([-25, 0, 25, 50])
    ax.grid(True, color = '#F1F1F1')
    fig.tight_layout()
    fig.savefig(pdfOutName, dpi = 300)
    

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 06 14:51:04 2017

@author: ciullo
"""
import numpy as np
from scipy.interpolate import interp1d


def dikefailure(sb, inflow, hriver, hbas, hground, status_t1,
                Bmax, Brate, simtime, tbreach, critWL):
    ''' Function establising dike failure as well as flow balance between the
        river and the polder

         inflow = flow coming into the node
         status = if False the dike has not failed yet
         critWL = water level above which we have failure

    '''
    tbr = tbreach
    #    h1 = hriver - hbreach
    #    h2 = (hbas + hground) - hbreach

    # h river is a water level, hbas a water depth
    h1 = hriver - (hground + hbas)

    # if the dike has already failed:
    if status_t1 == True:
        B = Bmax * (1 - np.exp(-Brate * (simtime - tbreach)))

        if h1 > 0:
            breachflow = 1.7 * B * (h1)**1.5

        # h1 <0 ==> no flow:
        else:
            breachflow = 0

        outflow = max(0, inflow - breachflow)
        status_t2 = status_t1

    # if the dike has not failed yet:
    else:
        failure = hriver > critWL
        outflow = inflow
        breachflow = 0
        # if it fails:
        if failure:
            status_t2 = True
            tbr = simtime
        # if it does not:
        else:
            status_t2 = False

    # if effects of hydrodynamic system behaviour have to be ignored:
    if sb == False:
        outflow = inflow

    return outflow, breachflow, status_t2, tbr


def Lookuplin(MyFile, inputcol, searchcol, inputvalue):
    ''' Linear lookup function '''

    col_values = MyFile[:, inputcol]
    minTableValue = np.min(col_values)
    maxTableValue = np.max(col_values)

    inputvalue2 = inputvalue

    if inputvalue >= maxTableValue:
        inputvalue = maxTableValue - 0.01
    elif inputvalue < minTableValue:
        inputvalue = minTableValue + 0.01

    A = np.max(MyFile[col_values <= inputvalue, inputcol])
    B = np.min(MyFile[col_values > inputvalue, inputcol])
    C = np.max(MyFile[col_values == A, searchcol])
    D = np.min(MyFile[col_values == B, searchcol])

    outpuvalue = C - ((D - C) * ((inputvalue - A) / (A - B))) * 1.0

    lin2, lin3 = Lookuplin2(MyFile, inputcol, searchcol, inputvalue2), Lookuplin3(MyFile, inputcol, searchcol, inputvalue2)
    if lin2 != lin3:
        if not np.all(np.diff(MyFile[:, inputcol]) > 0):
            print("Not all values of x are increasing")
        print("### WARNING: LOOKUPS DIFFER ###")
        print(f"Input value: {inputvalue2}, bounds: {minTableValue}, {maxTableValue}")
        print(f"Function returns: 1def: {outpuvalue}, 2scipy: {lin2}, 3numpy: {lin3}")
        breakpoint()

    return outpuvalue

def Lookuplin2(MyFile, inputcol, searchcol, inputvalue):
    ''' Linear lookup function '''
    bounds = (MyFile[:, searchcol].min(), MyFile[:, searchcol].max())
    lookup_function = interp1d(MyFile[:, inputcol], MyFile[:, searchcol], kind='linear', fill_value=bounds, bounds_error=False)
    return lookup_function(inputvalue)

def Lookuplin3(MyFile, inputcol, searchcol, inputvalue):
    ''' Linear lookup function '''
    # Copy input and search colums from NumPy array
    inputa, searcha = MyFile[:, inputcol], MyFile[:, searchcol]
    # Define the lower and upper bound for the return value
    rmin, rmax = searcha.min(), searcha.max()
    # Interpolate the inputvalue and clip to the bounds
    return np.clip(np.interp(inputvalue, np.sort(MyFile[:, inputcol]), MyFile[:, searchcol]), rmin, rmax)

def init_node(value, time):
    init = np.repeat(value, len(time)).tolist()
    return init

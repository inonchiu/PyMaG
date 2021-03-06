#!/usr/bin/env python

##################################
#
# Utils for PyMag
#
##################################

import numpy as np
from math import *

try:
    from scipy.spatial import cKDTree as KDT
except ImportError:
    from scipy.spatial import KDTree as KDT



# ---
# matching unique id
# ---
def IdMatch(id1, id2, return_indices = True):
    """
    Unique ID matching in two arrays. The base (reference) array is id1, and the given array is id2. We want to find what the counterparts of id2 are in id1.
    This utilizes numpp function only.
    """
    # sanitize
    id1         =       np.array(id1, ndmin = 1)
    id2         =       np.array(id2, ndmin = 1)
    # find the common id
    common_ids  =       list(set(id1).intersection(id2))
    # Return the index in id1
    if    return_indices:
        return np.nonzero(np.in1d(id1, common_ids))[0]
    else:
        return np.in1d(id1, common_ids)

# ---
# matching
# ---
def CartMatch(coord1, coord2, tol = None, nnearest=1):
    """
    Cartesian Coordinate mathcing
    """
    # sanitize
    coord1      =       np.array(coord1, ndmin = 1)
    coord2      =       np.array(coord2, ndmin = 1)

    # check the dimensions of the coordinate
    npairs1     =       len( coord1 )
    ndim1       =       1    if   len( np.shape(coord1) )  ==   1  else   \
                        np.shape(coord1)[1]
    npairs2     =       len( coord2 )
    ndim2       =       1    if   len( np.shape(coord2) )  ==   1  else   \
                        np.shape(coord2)[1]

    # check whether the coord1 and coord2 have the same shape
    if  ndim1   !=      ndim2:
        raise RuntimeError("The dims of coord1/2 are not the same.")
    else:
        ndim     =       ndim1

    # make proper arrays if they are 1d arrays
    if      ndim == 1:
        coord1  =       np.array([ coord1, np.zeros(len(coord1)) ]).T
        coord2  =       np.array([ coord2, np.zeros(len(coord2)) ]).T

    # kdtree the coord2
    kdt = KDT(coord2)
    if nnearest == 1:
        idxs2 = kdt.query(coord1)[1]
    elif nnearest > 1:
        idxs2 = kdt.query(coord1, nnearest)[1][:, -1]
    else:
        raise ValueError('invalid nnearest ' + str(nnearest))

    # distance - warning: this could be over float if the precision is not enough, we assume that case is beyond the distance of interest...
    ds  =   np.sqrt( np.sum( (coord1 - coord2[idxs2])**2, axis = 1) )

    # index of coord1 
    idxs1 = np.arange(npairs1)

    # distance filtering
    if tol is not None:
        msk = ds < tol
        idxs1 = idxs1[msk]
        idxs2 = idxs2[msk]
        ds = ds[msk]

    return idxs1, idxs2, ds




##############
#
# Testing
#
##############

if   __name__ == "__main__":
    coord1  =   range(0,1000)
    coord2  =   np.random.random_integers(0,1000,100)

    CartMatch(coord1 = coord1, coord2 = coord2, tol = None, nnearest=1)

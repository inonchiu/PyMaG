#!/usr/bin/env python

################################
#
# Functions
#
################################

import numpy as np
from math import *
import scipy.special as special
import pyfits
import pywcs
import os


# ---
# completeness function - simply an error function
# ---
def Completeness_Function(x, mag50, mag_dispersion):
    """
    Completeness funciton defined by the error function with two parmeters.
        
    Parameters:
        -`x`: 1d array or float. x is the input magnitude array or float.
        -`mag50`: float. mag50 is the magnitude where the completeness is 50 per cent.
        -`mag_dispersion`: float. mag_dispersion decribes the fall off speed of the completeness.
    Return:
        -`completeness`: 1d array or float. The completeness as the function of the input magnitude.
    """
    return -0.5 * special.erf((np.array(x,ndmin=1) - mag50)/mag_dispersion) + 0.5


# ---
# completeness_map profiling
# ---
def completeness_map_profiler(path2img, rac, decc, rmpc_edges, mpc2arcmin):
    """
    This function reads in the image called path2img, center (rac, decc) in degree, rmpc_edges and mpc2arcmin, and it will calculate the profile based on rmpc_edges.
    
    Parameters:
        -`path2img`:
        -`rac`:
        -`decc`:
        -`rmpc_edges`:
        -`mpc2arcmin`:
    
    Return:
        -`area_weight`:
        -`cmplt_map`:
        -`cmplt_per_ann`:
    
    """
    # sanitize
    rac                 =       float(rac)
    decc                =       float(decc)
    rmpc_edges          =       np.array( rmpc_edges, ndmin=1 )
    mpc2arcmin          =       float( mpc2arcmin )
    
    # read img
    if   not   os.path.isfile(path2img):
        raise IOError("path2img does not exsit:", path2img)
    else:
        readinimg       =       pyfits.getdata(path2img, ext = -1)

    # sanitize
    readinimg           =       np.ma.array(readinimg, mask = ~np.isfinite(readinimg))
    
    # read wcs
    hdulist             =       pyfits.open(path2img)
    wcs                 =       pywcs.WCS(hdulist[0].header)
    hdulist.close()

    # get the pixscale in the unit of arcsec/pix
    pix2arcsec          =       sqrt( np.abs( np.linalg.det(wcs.wcs.piximg_matrix) ) ) * 3600.0

    # get xc, yc
    xc, yc              =       wcs.wcs_sky2pix(rac, decc, 0)
    # get radii in pixel
    rmpc_edges_pixel    =       rmpc_edges * mpc2arcmin * 60.0 / pix2arcsec
    # get xyedges
    xedges              =       np.arange(wcs.naxis1 + 1) + 0.5
    yedges              =       np.arange(wcs.naxis2 + 1) + 0.5

    # calculate the area weight - in the dimension of (ny, nx) where ny is reverse
    print "#", "Area weighting...",
    area_weight         =       CellWeightAnnMap(xedges      = xedges,
                                                 yedges      = yedges,
                                                 xc          = xc,
                                                 yc          = yc,
                                                 radii_edges = rmpc_edges_pixel)
    # sanitize and a little tweak
    area_weight         =       np.ma.array(area_weight, mask = ( area_weight < 1E-5 ))
    area_weight         =       area_weight / area_weight
    print "Done!"

    # completeness per annulus
    cmplt_map           =       area_weight * readinimg
    cmplt_per_ann       =       np.array([ np.ma.mean(kmap) for kmap in cmplt_map ])
    # return
    return area_weight, cmplt_map, cmplt_per_ann


# ---
# utils funcs from Henk
# ---
def pixwt(xc, yc, r, x, y):
    """
    ; ---------------------------------------------------------------------------
    ; FUNCTION Pixwt( xc, yc, r, x, y )
    ;
    ; Compute the fraction of a unit pixel that is interior to a circle.
    ; The circle has a radius r and is centered at (xc, yc).  The center of
    ; the unit pixel (length of sides = 1) is at (x, y).
    ; ---------------------------------------------------------------------------
    """
    return intarea(xc, yc, r, x-0.5, x+0.5, y-0.5, y+0.5)
    
# ---
# utils funcs from Henk
# ---
def arc(x,y0,y1,r):
    """
    ; ---------------------------------------------------------------------------
    ; Function Arc( x, y0, y1, r )
    ;
    ; Compute the area within an arc of a circle.  The arc is defined by
    ; the two points (x,y0) and (x,y1) in the following manner:  The circle
    ; is of radius r and is positioned at the origin.  The origin and each
    ; individual point define a line which intersects the circle at some
    ; point.  The angle between these two points on the circle measured
    ; from y0 to y1 defines the sides of a wedge of the circle.  The area
    ; returned is the area of this wedge.  If the area is traversed clockwise
    ; then the area is negative, otherwise it is positive.
    ; ---------------------------------------------------------------------------
    """
    return 0.5 * (r**2) * ( np.arctan((y1)/(x)) - np.arctan((y0)/(x)))
    
# ---
# utils funcs from Henk
# ---
def chord( x, y0, y1):
    """
    ; ---------------------------------------------------------------------------
    ; Function Chord( x, y0, y1 )
    ;
    ; Compute the area of a triangle defined by the origin and two points,
    ; (x,y0) and (x,y1).  This is a signed area.  If y1 > y0 then the area
    ; will be positive, otherwise it will be negative.
    ; ---------------------------------------------------------------------------
    """
    return 0.5 * x * ( y1 - y0 )
    
# ---
# utils funcs from Henk - this is the most important one.
# ---
def intarea( xc, yc, r, x0, x1, y0, y1):
    """
    ; ---------------------------------------------------------------------------
    ; Function Intarea( xc, yc, r, x0, x1, y0, y1 )
    ;
    ; Compute the area of overlap of a circle and a rectangle.
    ;    xc, yc  :  Center of the circle.
    ;    r       :  Radius of the circle.
    ;    x0, y0  :  Corner of the rectangle.
    ;    x1, y1  :  Opposite corner of the rectangle.
    ; ---------------------------------------------------------------------------
    """
    x0 = x0 - xc
    y0 = y0 - yc
    x1 = x1 - xc
    y1 = y1 - yc
    return oneside( x1, y0, y1, r ) + oneside( y1, -x1, -x0, r )+  oneside( -x0, -y1, -y0, r ) + oneside( -y0, x0, x1, r )


def oneside(x, y0, y1, r):
    """
    ; ---------------------------------------------------------------------------
    ; Function Oneside( x, y0, y1, r )
    ;
    ; Compute the area of intersection between a triangle and a circle.
    ; The circle is centered at the origin and has a radius of r.  The
    ; triangle has verticies at the origin and at (x,y0) and (x,y1).
    ; This is a signed area.  The path is traversed from y0 to y1.  If
    ; this path takes you clockwise the area will be negative.
    ; ---------------------------------------------------------------------------
    """
    true = 1
    size_x  = np.size( x )

    if size_x <= 1:
        if x == 0:
            return x
        if np.abs(x) >= r:
            return arc( x, y0, y1, r )
        yh = np.sqrt( r**2 - x**2 )
        if (y0 <=-yh):
            if y1 <= -yh:
                return arc( x, y0, y1, r )
            elif y1 <= yh:
                return arc( x, y0, -yh, r ) + chord( x, -yh, y1 )
            else:
                return arc( x, y0, -yh, r ) + chord( x, -yh, yh ) + arc( x, yh, y1, r )
        elif ( y0 <  yh ):
            if y1 <= -yh:
                return chord( x, y0, -yh ) + arc( x, -yh, y1, r )
            elif y1 <= yh:
                return chord( x, y0, y1 )
            else:
                return chord( x, y0, yh ) + arc( x, yh, y1, r )
        else:
            if y1 <= -yh:
                return arc( x, y0, yh, r ) + chord( x, yh, -yh ) + arc( x, -yh, y1, r )
            elif y1 <= yh:
                return arc( x, y0, yh, r ) + chord( x, yh, y1 )
            else:
                return arc( x, y0, y1, r )
    else:
        ans = x*1.
        t0 = ( x == 0)
        count = np.sum(t0)
        if count == np.size( x ):
            return ans
        ans = x * 0.
        yh = ans*1.
        to = ( np.abs( x ) >= r)
        tocount=np.sum(to)
        to = np.arange(size_x)[( np.abs( x ) >= r)]
        ti = ( np.abs( x ) < r)
        ticount=np.sum(ti)
        ti = np.arange(size_x)[( np.abs( x ) < r)]
        if tocount != 0:
            ans[ to ] = arc( x[to], y0[to], y1[to], r )
        if ticount == 0:
            return ans
        yh[ ti ] = np.sqrt( r**2 - x[ti]**2 )
        t1 = (y0[ti] <= -yh[ti])
        count=np.sum(t1)
        t1 = np.arange(size_x)[(y0[ti] <= -yh[ti])]
        if count != 0:
            i = ti[t1]
            
            t2=(y1[i] <= -yh[i])
            count=np.sum(t2)
            t2= np.arange(size_x)[(y1[i] <= -yh[i])]
            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] =  arc( x[j], y0[j], y1[j], r )
            t2=( y1[i] > -yh[i] ) & ( y1[i] <=  yh[i] )
            count = np.sum(t2)
            t2 =  np.arange(size_x)[( y1[i] > -yh[i] ) & ( y1[i] <=  yh[i] )]
            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] = arc( x[j], y0[j], -yh[j], r )+ chord( x[j], -yh[j], y1[j] )
            t2 = (y1[i] > yh[i])
            count=np.sum(t2)
            t2=np.arange(size_x)[(y1[i] > yh[i])]
            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] = arc( x[j], y0[j], -yh[j], r )+ chord( x[j], -yh[j], yh[j] ) + arc( x[j], yh[j], y1[j], r )
        t1 =  ( y0[ti] > -yh[ti] ) & ( y0[ti] < yh[ti] )
        count=np.sum(t1)
        t1=np.arange(size_x)[t1]
        
        if count != 0:
            i = ti[ t1 ]
            t2 = ( y1[i] <= -yh[i])
            count=np.sum(t2)
            t2=np.arange(size_x)[t2]
            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] = chord( x[j], y0[j], -yh[j] ) + arc( x[j], -yh[j], y1[j], r )
            t2 = ( ( y1[i] > -yh[i] ) & ( y1[i] <=  yh[i] ))
            count=np.sum(t2)
            t2=np.arange(size_x)[t2]
            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] = chord( x[j], y0[j], y1[j] )
         

            t2 = ( y1[i] > yh[i])
            count=np.sum(t2)
            t2=np.arange(size_x)[t2]
            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] = chord( x[j], y0[j], yh[j] ) + arc( x[j], yh[j], y1[j], r )
         
        t1 = ( y0[ti] >= yh[ti])
        count = np.sum(t1)
        t1=np.arange(size_x)[t1]
        if count != 0:
            i = ti[ t1 ]
            t2 = ( y1[i] <= -yh[i])
            count = np.sum(t2)
            t2=np.arange(size_x)[t2]
            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] = arc( x[j], y0[j], yh[j], r ) + chord( x[j], yh[j], -yh[j] ) + arc( x[j], -yh[j], y1[j], r )
         

            t2 = ( ( y1[i] > -yh[i] )& ( y1[i] <=  yh[i] ))
            count=np.sum(t2)
            t2=np.arange(size_x)[t2]
            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] = arc( x[j], y0[j], yh[j], r )+chord( x[j], yh[j], y1[j] )
            t2 = ( y1[i] > yh[i])
            count=np.sum(t2)
            t2=np.arange(size_x)[t2]
            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] = arc( x[j], y0[j], y1[j], r )
        return ans

# ---
# Area weighting of a cell overlapping with a circule
# ---
def CellWeightCirMap(xedges, yedges, xc, yc, radii_edges):
    """
    CellWeightCirMap calculates the map of the area fraction of each cell 
    lying on the circles defined by RADII_EDGES.
    All length unit is in the same system.
    
    Parameters:
        -`xedges` the x_edges of the map. It is numpy array.
        -`yedges` the y_edges of the map. It is numpy array
        -`xc`, `yc`: the center of the circles.
        -`radii_edges`: the radii of each circle, it is a numpy array.
           
    Return:
        -`afrac`: The area fraction weight map with the shape of
                  len(radii_edges), len(xbins), len(ybins)
    """
    # Initial set up.
    xedges          =   np.array(xedges, ndmin=1)
    yedges          =   np.array(yedges, ndmin=1)
    radii_edges     =   np.array(radii_edges, ndmin=1)
    xc              =   float(xc)
    yc              =   float(yc)
    
    # meshgrid - this is in the shape of (len(yedges), len(xedges))
    xmesh, ymesh    =   np.meshgrid(xedges, yedges)

    # number of binnings
    nradii  =   len(radii_edges)
    nxbins  =   len(xedges) -   1
    nybins  =   len(yedges) -   1
    
    # weight map - looping is slowm but running just once so it should be fine.
    WeightedMap     =    np.array([ [ [ \
         float(
         intarea(xc, yc, radii_edges[nr], xmesh[ny][nx], xmesh[ny][nx+1], ymesh[ny][nx], ymesh[ny+1][nx]) / \
         abs( (xmesh[ny][nx+1] - xmesh[ny][nx]) * (ymesh[ny+1][nx] - ymesh[ny][nx]) ) \
         ) for ny in xrange(nybins) ] for nx in xrange(nxbins) ] for nr in xrange(nradii) ])
    
    # return
    return WeightedMap


# ---
# Area weighting of a cell overlapping with a ring
# ---
def CellWeightAnnMap(xedges, yedges, xc, yc, radii_edges):
    """
    Same as CellWeightCirMap, but with annulli.
    """
    return CellWeightCirMap(xedges, yedges, xc, yc, radii_edges[1:]) - CellWeightCirMap(xedges, yedges, xc, yc, radii_edges[:-1])




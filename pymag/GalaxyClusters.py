
######################################################
#
# This is the module of Galaxy Clusters
#
######################################################

import numpy as np
from math import *
import matplotlib.pyplot as pyplt
import os
import pyfits
import numpy.lib.recfunctions as rfn # magic, used to join structured arrays
import scipy.optimize as optimize
import cosmolopy.distance as cosdist
from utils import matching
from utils import funcs
import NFW
import warnings


# set up cosmology - has to be compatible with cosmolopy
cosmo           =       {'omega_M_0' : 0.3, 'omega_lambda_0' : 0.7, 'h' : 0.7, 'omega_k_0' : 0.0 }

######################################################
#
# Class of Galaxy Clusters
#
######################################################

class GCluster(NFW.Halo):
    """
    This is the class module of galaxy clusters--GCluster.

    GCluster inherits the NFW class because it uses the NFW model as the general description of a galaxy cluster profile.

    Parameters:
        -`gcluster_name`: string. The name of this instance.
        -`path2photcat`: string. The abs path to the photometry catalog.
        -`path2zcat`: string. The abs path to the photo-z point catalog.
        -`path2pdzcat`: string. The abs path to the pdz catalog.
        -`names_in_cats`: dict. The column names used in `path2photcat`, `path2zcat` and `path2pdzcat`. The keys are listed in the default dicts.
        -`rac`: float. The center of the ra in deg.
        -`decc`: float. The center of the dec in deg.
        -`zd`: float. The redshift.
        -`mass`: float. The mass in Msun. The mas definition is defined by `overden` and `wrt`.
        -`concen`: float. The concentration. I.e., R/rs, where R is the radius defined by the cluster mass.
        -`overden`: float. The overdensity against the `wrt`. Negative if it is virial overdensity.
        -`wrt`: string. `crit` or `mean`. If `overden` < 0.0, then the `wrt` has no effect.
        -`cosmo`: dict. The cosmolopy cosmology dict.

    Return:
        -`GCluster`: A GCluster instance.
    """
    # ---
    # Initialize
    # ---
    def __init__(self, gcluster_name =       "name"     ,
                       path2photcat  =       "test.cat.fits",
                       path2zcat     =       "test.bpz.fits",
                       path2pdzcat   =       "test.pdz.fits",
                       path2cmplt_map=       None,
                       path2fcommag  =       None,
                       names_in_cats =       {
                                              "objid"   : "objid",     # object id across every catalog
                                              "ra"      : "alpha",     # ra name in catalog
                                              "dec"     : "delta",     # dec name in catalog
                                              "s_band"  : "Bdered",    # s_band name in catalog
                                              "zp"      : "z_b",       # photoz name in zcat
                                              "zp_min"  : "z_b_min",   # photoz_min name in zcat (this is 95% confidence level)
                                              "zp_max"  : "z_b_max",   # photoz_max name in zcat (this is 95% confidence level)
                                            "pdzheader" : "c"      ,   # the pdz header in pdzcat, needs to be pdz0, pdz1, ...
                                              },
                       rac           =       79.7       ,
                       decc          =       -48.71     ,
                       zd            =       0.3        ,
                       mass          =       6E14       ,
                       concen        =       3.0        ,
                       overden       =       500.0      ,
                       wrt           =       "crit"     ,
                       cosmo         =       cosmo      ,
                       ):
        # call NFW __init__
        super(GCluster, self).__init__(
                zd              =   zd,
                mass            =   mass,
                concen          =   concen,
                overden         =   overden,
                wrt             =   wrt,
                cosmo           =   cosmo,
                )
        # set up local attribute
        self.gcluster_name      =   gcluster_name
        self.path2photcat       =   path2photcat
        self.path2zcat          =   path2zcat
        self.path2pdzcat        =   path2pdzcat
        self.path2cmplt_map     =   path2cmplt_map
        self.path2fcommag       =   path2fcommag
        self.names_in_cats      =   names_in_cats
        self.rac                =   rac
        self.decc               =   decc

    # ---
    # introduce myself
    # ---
    def i_am(self):
        """
        Just print the diagnostic information
        """
        print
        print "#" * 20
        print "#", "mass is in the unit of Msun."
        print "#", "length is in the unit of Mpc."
        print "#", "density is in the unit of Msun/Mpc^3."
        print "#"
        for name_of_attribute in self.__dict__:
            if    name_of_attribute     not   in   ["_readinphotcat", "_readinzcat", "_readinpdzcat", "_readin_cmpltmap"]:
                print "#", name_of_attribute, ":", getattr(self, name_of_attribute)
        print "#"
        print "#" * 20

        return

    # ---
    # read in img
    # ---
    def readinimgs(self, readinname = "cmpltmap"):
        """
        Check whether the catalogs exist and read them.

        Parameters:
            -`readinname`: string. It has to be "cmpltmap". "cmpltmap": completeness map, must contain correct and enough wcs info.
        """
        # check which image is going to be read in.
        if readinname   ==    "cmpltmap" and \
           self.path2cmplt_map is not None:

            path2img    =     self.path2cmplt_map
            # read img
            if   not   os.path.isfile(path2img):
                raise IOError("path2img does not exsit:", path2img)
            else:
                self._readin_cmpltmap       =       pyfits.getdata(path2img, ext = -1)
        else:
            raise NameError("readinname:", readinname)

        return

    # ---
    # read in cats
    # ---
    def readincats(self, readinname = "photcat"):
        """
        Check whether the catalogs exist and read them.

        Parameters:
            -`readinname`: string. It has to be "photcat", "zcat" or "pdzcat". "photcat": photometry catalog. "zcat": the photo-z point estimator catalog. "pdzcat": the pdz catalog.
        """
        # read in the photometric catalogs 
        if   readinname == "photcat" and \
             self.path2photcat is not None and \
             os.path.isfile(self.path2photcat):

            # read in
            self._readinphotcat      =       pyfits.getdata(self.path2photcat, ext = -1)
            # derive the radius
            Deg2Rad                  =       pi / 180.0
            Factor                   =       np.sin( self._readinphotcat[ self.names_in_cats["dec"] ] * Deg2Rad) * sin( self.decc * Deg2Rad) + \
                                             np.cos( self._readinphotcat[ self.names_in_cats["dec"] ] * Deg2Rad) * cos( self.decc * Deg2Rad) * \
                                             np.cos((self._readinphotcat[ self.names_in_cats["ra" ] ] - self.rac) * Deg2Rad)
            # radii in arcmin and mpc
            radii_arcmin             =       np.arccos(Factor) / Deg2Rad * 60.0
            radii_mpc                =       radii_arcmin * self.arcmin2mpc
            radii_arcmin             =       np.array(radii_arcmin,
                                                      dtype = np.dtype([ ("radii_arcmin", "f8") ]) )
            radii_mpc                =       np.array(radii_mpc,
                                                      dtype = np.dtype([ ("radii_mpc"   , "f8") ]) )
            # join catalogs
            self._readinphotcat      =       rfn.merge_arrays(
                                             [ self._readinphotcat, radii_arcmin, radii_mpc ],
                                             flatten = True, usemask = False)

        elif    readinname == "zcat" and \
                self.path2zcat is not None and \
                os.path.isfile(self.path2zcat):
            # read in
            self._readinzcat         =       pyfits.getdata(self.path2zcat, ext = -1)

        elif    readinname == "pdzcat" and \
                self.path2pdzcat is not None and \
                os.path.isfile(self.path2pdzcat):
            # read in
            self._readinpdzcat       =       pyfits.getdata(self.path2pdzcat, ext = -1)

            # create pdz_mtrx
            # construct the re of pdzheader
            import re
            re_pdzheader             =       re.compile( "^(" + self.names_in_cats["pdzheader"] + "[0-9]|" + \
                                                                self.names_in_cats["pdzheader"] + "[0-9][0-9]|" + \
                                                                self.names_in_cats["pdzheader"] + "[0-9][0-9][0-9])" )
            extracted_pdz_arrays     =       []
            for header_item in self._readinpdzcat.dtype.names:
                if   re_pdzheader.match(header_item)  is   not   None:
                    extracted_pdz_arrays.append(self._readinpdzcat[ header_item ])
            self._pdz_mtrx           =       np.transpose( extracted_pdz_arrays ).copy()

        else:
            raise NameError("readinname:", readinname, "has to be photcat, zcat, or pdzcat")

        return


    # ---
    # Measure the power law index of logN-logS - I call it sslope
    # ---
    def derive_sslope(self,
            name_band       =       None    ,
            excised_rmpc    =       1.0     ,
            md_mag_edges    =       np.arange(18.0, 28.0, 0.1),
            s_mag           =       24.5    ,
            ds_mag          =       0.50    ,
            #mag_disp        =       None    , # FIXME: Dont use error function to characterize completeness function
            #mag50           =       None    , # FIXME: Dont use error function to characterize completeness function
            nboot           =       1000    ,
            use_core_excised=       False   ,
            use_comp_crrct  =       False   ,
            plotme          =       False   ,
            ):
        """
        This method measures the power law index, which is defined as sslope = dlog10 N(<m) / dm.

        The tricky part is that we measure the culmulative count slope--which has covariance among magnitude bins--instead of
        the differential counts, therefore the best approach is to fit the s locally with derived covariance matrix.

        Another complication is the incompleteness of the magnitude distribution, which plays an important role in the faint end.

        The idea is simple.
        First we create the magnitude distribution, md(m), as a function of magnitude.
        Second, we fold in the pre-modelled completeness function, which is an error function with two parameters--mag_disp and mag50.
        Then we have the incompleteness-corrected md_comp(m).
        Third, we bootstrap the md_comp(m) and derive the culmd_comp(m) for each realization. We then can derive the covariance by
        equation~19 in Chiu+16. I.e.,
        Cov[mag_i, mag_j]  =   < (C(mag_i) - <C(mag_i)>) * (C(mag_j) - <C(mag_j)>) >, where C(mag_i) = log10( N(<mag_i) )
        Note that this assumes the number counts are large enough and log10( N(<mag_i) ) is a Guassian distribution.
        Fourth, we minimize the chi2 by equation~20 in Chiu+16. I.e.,
        chi2    =   sum( D_i * C_{i,j}^-1 * Dj ), where D_i = log10Nm(<mag_i) - log10N(<mag_i) and Nm is the parametrized model.

        Parameters:
            -`name_band`: string or None. The name of the band which the power law index is derived. Using `s_band` in `names_in_cats` if None.
            -`excised_rmpc`: float. The radius in Mpc that is used for core excision. Default is 1 Mpc.
            -`md_mag_edges`: 1d array. The magnitude bin that is used for magnitude binning. Default is np.arange(18.0, 28.0, 0.1),
            -`s_mag`: float. The magnitude threshold in `name_band` where the power law index is derived. Default is 24.5 mag.
            -`ds_mag`: float. Same above but it is the magnitude width at the magnitude threshold. Precisely, the power law index is estimated in the magnitude range |mag - `s_mag`| < `ds_mag`. Default is 0.5 mag.
            -`mag_disp`: float. The dispersion of the error function which describes the incompeteness in `name_band`. Default is None, meaning no incompleteness correction is applied.
            -`mag50`: Same above, but this is the mag50.
            -`nboot`: int. The number of the bootstrapping that is used in deriving the covar. Default is 1000.
            -`use_core_excised`: Bool. Core excision if True. Default is False.
            -`use_comp_crrct`: Bool. Applying incompleteness correction is True. Default is False.
            -`plotme`: Bool. Plot the sslope or not. Default is False.

        Return:
            -`sslope`: float. The power law index.
            -`dsslope`: float. Same above but it is the error of the power law index. It requires numdifftools module. Returning np.nan if not installed.

        """

        # check if the readinphotcat exists.
        if  not hasattr(self, '_readinphotcat'):
            print
            print "#", "_readinphotcat does not exist, starting to read it."
            self.readincats(readinname = "photcat")
            print "#", "Done."
            print

        # check the s_band name
        if    name_band    is    None:
            name_band   =   self.names_in_cats["s_band"]

        # use_core_excised?
        if     use_core_excised:
            print
            print "#", "Using core excised catalogs, excluding radius:", excised_rmpc, "Mpc."
            i_am_after_radii_filter       =       ( self._readinphotcat["radii_mpc"] > excised_rmpc )
            print "#", "Done"
            print
        else:
            print
            print "#", "Not using core excised catalog."
            i_am_after_radii_filter       =       np.ones(len(self._readinphotcat["radii_mpc"]), dtype = np.bool)
            print "#", "Done"
            print

        # diag
        print
        print "#", "derive_sslope for name_band:", name_band, "s_mag:", s_mag
        print


        # derive mag binning
        md_mag_bins =   0.5 * ( md_mag_edges[1:] + md_mag_edges[:-1] )
        md_mag_steps=         ( md_mag_edges[1:] - md_mag_edges[:-1] )

        # derive md and mderr
        md          =   np.histogram(self._readinphotcat[name_band][ i_am_after_radii_filter ], bins = md_mag_edges, range = (md_mag_edges.min(), md_mag_edges.max()))[0]
        mderr       =   np.sqrt(md)

        # use_comp_crrct?
        if          use_comp_crrct:
            print
            print "#", "use_comp_crrct...",
            #incmp_crrct     =       1.0 / funcs.Completeness_Function( x = md_mag_bins, mag50 = mag50, mag_dispersion = mag_disp )
            incmp_crrct     =       1.0 / funcs.Interpolate_Completeness_Function( x = md_mag_bins, path2fcommag = self.path2fcommag )
            md              =       md * incmp_crrct
            mderr           =       mderr * incmp_crrct
            # interpolation
            print "Done."
            print

        else:
            print
            print "#", "Not using use_comp_crrct."
            # derive the completeness correction 
            incmp_crrct     =       np.ones(len(md_mag_bins))
            md              =       md * incmp_crrct
            mderr           =       mderr * incmp_crrct
            print "#", "Done."
            print

        # derive culm MD
        culmd       =   np.cumsum( md )

        # ---
        # Construct the covar / inv_covar
        # ---
        # Bootstrap
        md_boot                 =       np.random.poisson( lam = md, size = (nboot, len(md)) )
        culmd_boot              =       np.cumsum( md_boot, axis = 1 )
        log10_culmd_boot        =       np.log10( culmd_boot )
        mean_log10_culmd_boot   =       np.mean( log10_culmd_boot, axis = 0 )
        # Derive the ndim of the covar matrix -- skip the culmd = 0
        useme_to_fit            =       np.isfinite( mean_log10_culmd_boot ) & ( np.abs(md_mag_bins - s_mag) <= ds_mag )
        use_me_to_derive_inv    =       np.array( useme_to_fit, ndmin = 2 )
        # ndim_mtrx is the number of the magnitude bins going to the fit
        ndim_mtrx               =       np.sum( useme_to_fit )

        # If ndim_mtrx > 1 (2 or above), then we do the fit
        if    ndim_mtrx     >   1:
            # Derive the covariance matrix - in the ndim_mtrx by ndim_mtrx
            diff_arry               =       log10_culmd_boot - mean_log10_culmd_boot
            covar_mtrx              =       np.dot( diff_arry.T, diff_arry)\
                                            [ np.dot( use_me_to_derive_inv.T, use_me_to_derive_inv ) ].reshape( \
                                            (ndim_mtrx, ndim_mtrx) ) / nboot
            # Derive the inverse of covar matrix
            # Since corre_mtrx      =       diag(covar)^-0.5 covar_mtrx diag(covar)^-0.5
            # Or diag(covar)^0.5 corre_mtrx diag(covar)^0.5 = covar_mtrx
            # Hence covar_mtrx^-1   =       diag(covar)^-0.5 corre_mtrx^-1 diag(covar)^-0.5
            # Derive the inverse of covar matrix
            diag_mtrx               =       np.diag( 1.0 / np.diag( covar_mtrx ) )
            corre_mtrx              =       np.dot( np.sqrt(diag_mtrx),
                                                    np.dot( covar_mtrx , np.sqrt(diag_mtrx) )
                                                  )
            # try whether we have invertible matrix
            try:
                inv_covar_mtrx          =       np.dot( np.sqrt(diag_mtrx),
                                                        np.dot( np.linalg.inv( corre_mtrx ), np.sqrt(diag_mtrx) )
                                                      )
                perform_the_fit         =       True
            except np.linalg.LinAlgError:
                inv_covar_mtrx          =       np.nan
                perform_the_fit         =       False
                print
                print "#", "the covar is not invertible, we set inv_covar to np.nan"
                print

        # ndim_mtrx <=1, do not construct covar.
        else:

            covar_mtrx              =       np.nan
            corre_mtrx              =       np.nan
            inv_covar_mtrx          =       np.nan
            perform_the_fit         =       False
            print
            print "#", "ndim", ndim_mtrx ,"which is used to construct the covar is <=1, we do not perform the fit and the inv_covar = np.nan"
            print

        # ---
        # Perform the fit
        # ---
        if   not   perform_the_fit:
            sslope, dsslope         =       np.nan, np.nan
            return sslope, dsslope

        else:
            # define the function
            def running_power_law(mag, a, b, c):
                # define logN(<m) = 0.5 * a * mag**2 + b * mag + c, then s = dlogN(<m)/dm = a * mag + b
                return 0.5 * a * mag**2 + b * mag + c

            # define the func2min
            def func2min(params):
                # calc the model
                log10CulModel   =   running_power_law(md_mag_bins[ useme_to_fit ], *params)
                # calc the diff
                darray          =   ( log10CulModel - np.log10( culmd[ useme_to_fit ] ) )
                # calc the chi2
                chi2            =   np.dot( darray.T, np.dot(inv_covar_mtrx, darray) )
                return chi2

            # fit
            fitresult           =   optimize.minimize(
                                    func2min,
                                    x0 = (0.0, 0.4, np.mean(np.log10( culmd[ useme_to_fit ] )) - 0.4 * s_mag),
                                    method = "Nelder-Mead",
                                    )
            # sslope
            sslope              =   fitresult["x"][0] * s_mag + fitresult["x"][1]
            try:
                import numdifftools
                Hfunction       =   numdifftools.Hessian(func2min)
                inv_hessian     =   np.linalg.inv( Hfunction( fitresult["x"] ) )
                dsslope         =   np.sqrt(
                                    np.dot(
                                        np.array([s_mag, 1.0]).T,
                                        np.dot( inv_hessian[:2,:2], np.array([s_mag, 1.0]) )
                                        ) )
            except (ImportError, np.linalg.LinAlgError):
                dsslope         =   np.nan
                print
                print "#", "No numdifftools module, setting dsslope to be np.nan"
                print

        # plotme?
        if plotme:
            pyplt.figure("sslope", figsize = (6.0,6.0))
            pyplt.plot(md_mag_bins, culmd, color = "k", ls = "--", label = "logN(<m)")
            pyplt.plot(md_mag_bins[ useme_to_fit ], 10.0**running_power_law(md_mag_bins[ useme_to_fit ], *fitresult["x"]), "k-", label = "model")
            pyplt.yscale("log")
            pyplt.xlabel("mag")
            pyplt.ylabel("logN(<m)")
            pyplt.legend(loc = 4, numpoints = 1)
            pyplt.show()


        return sslope, dsslope

    # ---
    # Measure the beta distribution - This generally requires (at least) the photo-z catalog of the selected sources.
    # ---
    def derive_Pbeta(self,
            name_band       =       None    ,
            excised_rmpc    =       None    ,
            included_rmpc   =       None    ,
            mag_lo          =       None    ,
            mag_hi          =       None    ,
            use_zpoint      =       True    ,
            pdz_z_bins      =       np.arange(0.01, 5.0+0.0001, 0.01),
            beta_edges      =       None    ,
            plotme          =       False   ,
            ):
        """
        The goal of this function is to derive the Pbeta for the given sources in catalogs.

        Parameters:
            -`name_band`: string or None. The name of the band which the power law index is derived. Using `s_band` in `names_in_cats` if None.
            -`excised_rmpc`: float. The radius in Mpc that is used for core excision. Not applying if None. Default is None.
            -`included_rmpc`: float. Same as included, but it is for the radius that including the region of interest.
            -`mag_lo`: float. The lower bound of the magnitude cut. Not applying if None, which is default.
            -`mag_hi`: float. Same as `mag_lo` but for the upper bound of the magnitude cut.
            -`use_zpoint`: bool. Using photo-z point estimator if True, otherwise using P(z)--pdzcat. Default is True.
            -`pdz_z_bins`: 1d array. The magnitude bins for the pdz catalog. This has to match the number of the binning in pdz format.
            -`beta_edges`: 1d array. The beta binning. Using np.arange(0.0, 1.01, 0.01) if None, which is default.
            -`plotme`: Bool. Plot the sslope or not. Default is False.

        Return:
            -`beta_bins`: 1d array. The beta binning used to derive the P(beta). It is 0.5 * (beta_edges[1:] + beta_edges[:-1]).
            -`pb`: 1d array. The P(beta) for given `beta_bins`.

        """

       # check if the readinphotcat exists.
        if  not hasattr(self, '_readinphotcat'):
            print
            print "#", "_readinphotcat does not exist, starting to read it."
            self.readincats(readinname = "photcat")
            print "#", "Done."
            print

        # ---
        # radial filtering
        # ---
        if    excised_rmpc   is   not   None   and   \
              included_rmpc  is   not   None:
            i_am_in_radial_bin  =      ( excised_rmpc <= self._readinphotcat["radii_mpc"] ) & \
                                       ( self._readinphotcat["radii_mpc"] < included_rmpc )
        else:
            i_am_in_radial_bin  =      np.ones( len(self._readinphotcat["radii_mpc"]), dtype = bool)

        # ---
        # The magnitude cut
        # ---
        if    mag_lo   is   not   None   and   \
              mag_hi   is   not   None:
            if  name_band   is  None:
                i_am_in_mag_bin     =      ( mag_lo <= self._readinphotcat[ self.names_in_cats["s_band"] ] ) & \
                                           ( self._readinphotcat[ self.names_in_cats["s_band"] ] < mag_hi  )
            else:
                i_am_in_mag_bin     =      ( mag_lo <= self._readinphotcat[ name_band ] ) & \
                                           ( self._readinphotcat[ name_band ] < mag_hi  )
        else:
            i_am_in_mag_bin         =      np.ones( len(self._readinphotcat[ self.names_in_cats["s_band"] ]), dtype = bool)


        # ---
        # Derive beta_bins
        # ---
        if    beta_edges   is   None:
            beta_edges      =       np.arange(0.0, 1.01, 0.02)
            beta_bins       =       0.5 * (beta_edges[1:] + beta_edges[:-1])
        else:
            beta_bins       =       0.5 * (beta_edges[1:] + beta_edges[:-1])


        # ---
        # using point estimator or pdz?
        # ---
        if    use_zpoint:

            # check if the zcat exists.
            if  not hasattr(self, '_readinzcat'):
                print
                print "#", "_readinzcat does not exist, starting to read it."
                self.readincats(readinname = "zcat")
                print "#", "Done."
                print


            # if there are objects after the radial/mag binning
            if   np.sum( i_am_in_radial_bin & i_am_in_mag_bin ) > 0:
                # ---
                # matching id for the object __AFTER__ the radial/mag bin
                # ---
                print "#", "matched id...",
                matched_idx1        =       matching.IdMatch(id1 = self._readinphotcat[ self.names_in_cats["objid"] ][ i_am_in_radial_bin & i_am_in_mag_bin ],
                                                             id2 = self._readinzcat[    self.names_in_cats["objid"] ],
                                                             return_indices = True)
                matched_idx2        =       matching.IdMatch(id1 = self._readinzcat[    self.names_in_cats["objid"] ],
                                                             id2 = self._readinphotcat[ self.names_in_cats["objid"] ][ i_am_in_radial_bin & i_am_in_mag_bin ],
                                                             return_indices = True)
                print "Done!"
                # missing anything in matching?
                missing_idx1        =       set( list(xrange(len(self._readinphotcat[ self.names_in_cats["objid"] ][ i_am_in_radial_bin & i_am_in_mag_bin ]))) ) - \
                                            set( matched_idx1 )
                missing_idx1        =       np.array( list(missing_idx1), dtype = np.int)
                missing_idx2        =       set( list(xrange(len(self._readinzcat[    self.names_in_cats["objid"] ]))) ) - \
                                            set( matched_idx2 )
                missing_idx2        =       np.array( list(missing_idx2), dtype = np.int)

                # Diagnostic
                if   len(missing_idx1) != 0:
                    print RuntimeWarning("object_missing:", len(missing_idx1), "matched:", len(matched_idx1))

                # i_am_used - this is against the catalog __BEFORE__ radial/mag filtering
                i_am_used           =       matching.IdMatch(id1 = self._readinphotcat[ self.names_in_cats["objid"] ],
                                                             id2 = self._readinphotcat[ self.names_in_cats["objid"] ][ i_am_in_radial_bin & i_am_in_mag_bin ][ matched_idx1 ],
                                                             return_indices = False)
                # ---
                # Diagnostic
                # ---
                print
                print "#", "total objs:", len(i_am_used)
                print "#", "After radial filtering between", excised_rmpc, "and", included_rmpc, ":", \
                           np.sum( i_am_in_radial_bin )
                print "#", "After mag filtering between", mag_lo, "and", mag_hi, ":", \
                           np.sum( i_am_in_mag_bin )
                print "#", "After radial/mag filtering :", np.sum(i_am_in_radial_bin & i_am_in_mag_bin)
                print "#", "After radial/mag filtering and with redshift infomation:", np.sum( i_am_used )
                print


            # no objects surviving the radial bins
            else:

                # Diagnostic
                print RuntimeWarning("no objects surviving the radial/mag bins.")
                i_am_used    =       np.zeros( len(self._readinphotcat[ self.names_in_cats["objid"] ]), dtype = bool )
                #return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                return beta_bins, np.nan * beta_bins, np.nan * beta_bins, pdz_z_bins, np.nan * pdz_z_bins, np.nan * pdz_z_bins


        else:

            # check if the pdzcat exists.
            if  not hasattr(self, '_readinpdzcat'):
                print
                print "#", "_readinpdzcat does not exist, starting to read it."
                self.readincats(readinname = "pdzcat")
                print "#", "Done."
                print

            # if there are objects after the radial/mag binning
            if   np.sum( i_am_in_radial_bin & i_am_in_mag_bin ) > 0:
                # ---
                # matching id for the object __AFTER__ the radial bin
                # ---
                print "#", "matched id...",
                matched_idx1        =       matching.IdMatch(id1 = self._readinphotcat[ self.names_in_cats["objid"] ][ i_am_in_radial_bin & i_am_in_mag_bin ],
                                                             id2 = self._readinpdzcat[  self.names_in_cats["objid"] ],
                                                             return_indices = True)
                matched_idx2        =       matching.IdMatch(id1 = self._readinpdzcat[  self.names_in_cats["objid"] ],
                                                             id2 = self._readinphotcat[ self.names_in_cats["objid"] ][ i_am_in_radial_bin & i_am_in_mag_bin ],
                                                             return_indices = True)
                print "Done!"
                # missing anything in matching?
                missing_idx1        =       set( list(xrange(len(self._readinphotcat[ self.names_in_cats["objid"] ][ i_am_in_radial_bin & i_am_in_mag_bin ]))) ) - \
                                            set( matched_idx1 )
                missing_idx1        =       np.array( list(missing_idx1), dtype = np.int)
                missing_idx2        =       set( list(xrange(len(self._readinpdzcat[    self.names_in_cats["objid"] ]))) ) - \
                                            set( matched_idx2 )
                missing_idx2        =       np.array( list(missing_idx2), dtype = np.int)

                # Diagnostic
                if   len(missing_idx1) != 0:
                    print RuntimeWarning("object_missing:", len(missing_idx1), "matched:", len(matched_idx1))

                # i_am_used - this is against the catalog __BEFORE__ radial/mag filtering
                i_am_used           =       matching.IdMatch(id1 = self._readinphotcat[ self.names_in_cats["objid"] ],
                                                             id2 = self._readinphotcat[ self.names_in_cats["objid"] ][ i_am_in_radial_bin & i_am_in_mag_bin ][ matched_idx1 ],
                                                             return_indices = False)

                # ---
                # Diagnostic
                # ---
                print
                print "#", "total objs:", len(i_am_used)
                print "#", "After radial filtering between", excised_rmpc, "and", included_rmpc, ":", \
                           np.sum( i_am_in_radial_bin )
                print "#", "After mag filtering between", mag_lo, "and", mag_hi, ":", \
                           np.sum( i_am_in_mag_bin )
                print "#", "After radial filtering and with redshift infoi:", np.sum( i_am_used )
                print

            # no objects surviving the radial bins
            else:

                # Diagnostic
                print RuntimeWarning("no objects surviving the radial/mag bins.")
                i_am_used    =       np.zeros( len(self._readinphotcat[ self.names_in_cats["objid"] ]), dtype = bool )
                #return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                return beta_bins, np.nan * beta_bins, np.nan * beta_bins, pdz_z_bins, np.nan * pdz_z_bins, np.nan * pdz_z_bins

        # ---
        # Deriving the beta
        # ---

        # beta function
        def beta_calculator(zs, zd, cosmo):
            Ds      =   cosdist.angular_diameter_distance(zs, z0 = 0.0, **cosmo)
            Dls     =   cosdist.angular_diameter_distance(zs, z0 =  zd, **cosmo)
            Beta    =   Dls / Ds
            Beta[ (Beta < 0.0) ]    =   0.0
            return Beta

        # zpoint or pdz
        if    use_zpoint:
            # calculate the P(z)
            zstep                =   (pdz_z_bins[1] - pdz_z_bins[0])
            pdz_z_edges          =   np.append( pdz_z_bins - zstep / 2.0, pdz_z_bins[-1] + zstep / 2.0 )
            if    len(matched_idx2) > 0:
                pz                   =   np.histogram(self._readinzcat[ self.names_in_cats["zp"    ] ][ matched_idx2 ], bins = pdz_z_edges, range = (pdz_z_edges.min(), pdz_z_edges.max()))[0] * 1.0
                pz_norm              =   pz / np.sum( pz )
                # calculate beta of each used object
                betas                =   beta_calculator(self._readinzcat[ self.names_in_cats["zp"    ] ][ matched_idx2 ], zd = self.zd, cosmo = self.cosmo )
                # calculate Pb and normalize
                pb                   =   np.histogram(betas, bins = beta_edges, range = (beta_edges.min(), beta_edges.max()))[0] * 1.0
                pb_norm              =   pb / np.sum(pb)
                #pyplt.plot(beta_bins, pb, "k-")
            else:
                return beta_bins, np.nan * beta_bins, np.nan * beta_bins, pdz_z_bins, np.nan * pdz_z_bins, np.nan * pdz_z_bins
        else:
            if    len(matched_idx2) > 0:
                # calculate the pz
                pz                   =   np.mean( self._pdz_mtrx[ matched_idx2 ], axis = 0 )
                pz_norm              =   pz / np.sum( pz )
                # P(beta) = beta(z) P(z)
                beta_of_z            =   beta_calculator(pdz_z_bins, zd = self.zd, cosmo = self.cosmo)
                pb_mtrx              =   beta_of_z * self._pdz_mtrx[ matched_idx2 ]
                pb_stacked           =   np.mean( pb_mtrx, axis = 0 )
                # interpolate and normalize
                pb                   =   np.interp( x = beta_bins, xp = beta_of_z, fp = pb_stacked)
                pb_norm              =   pb / np.sum(pb)
            else:
                return beta_bins, np.nan * beta_bins, np.nan * beta_bins, pdz_z_bins, np.nan * pdz_z_bins, np.nan * pdz_z_bins
        # plotme?
        if plotme:

            pyplt.figure("Pb", figsize = (6.0, 6.0))
            pyplt.plot(beta_bins, pb, "k--")
            pyplt.show()

        # return
        return beta_bins, pb_norm, pb, pdz_z_bins, pz_norm, pz


    # ---
    # Measure the number density
    # ---
    def derive_ngal(self,
            name_band       =       None    ,
            rmpc_edges      =       None    ,
            mag_lo          =       None    ,
            mag_hi          =       None    ,
            use_comp_crrct  =       False   ,
            cmplt_prfl      =       None    ,
            cmplt_rmpc      =       None    ,
            plotme          =       False   ,
            ):
        """
        The goal of this function is to derive number density given a magnitude bin for the given sources in catalogs.

        Parameters:
            -`name_band`: string or None. The name of the band which the power law index is derived. Using `s_band` in `names_in_cats` if None.
            -`rmpc_edges`: 1d array. The radius binning used to derive the ngal and Ngal. Using np.linspace(0.1, 1.0, 10.0) if None. Default is None.
            -`mag_lo`: float. The lower bound of the magnitude cut. Not applying if None, which is default.
            -`mag_hi`: float. Same as `mag_lo` but for the upper bound of the magnitude cut.
            -`cmplt_prfl`: 1d array. The completeness profile.
            -`cmplt_rmpc`: 1d array. The radial bins of the completeness profile (in the unit of Mpc).
            -`use_comp_crrct`: bool. Estimate the completeness for the given radia binning `rmpc_edges` if True. If `cmplt_prfl` and `cmplt_rmpc` are both given, then the completeness of each radial bin is estimated by interpolating the `cmplt_prfl`, otherwise we use `self.path2cmplt_map` to estimate the completeness. The completenee per radial bin is one if False.
            -`plotme`: bool. Plot if True.

        Return:
            -`rmpc_bins`: 1d array. The radial binning in rmp. It is 0.5 * (rmpc_edges[1:] + rmpc_edges[:-1]).
            -`Ngal`: 1d array. The profile of number of sources.
            -`ngal`: 1d array. Same above but for _density_. It is in the unit of #/mpc**2.

        """

       # check if the readinphotcat exists.
        if  not hasattr(self, '_readinphotcat'):
            print
            print "#", "_readinphotcat does not exist, starting to read it."
            self.readincats(readinname = "photcat")
            print "#", "Done."
            print

        # radial binning
        if    rmpc_edges    is  None:
            rmpc_edges      =      np.logspace(-1.0, log10(2.0), 11)
            rmpc_bins       =      0.5 * (rmpc_edges[1:] + rmpc_edges[:-1])
        else:
            rmpc_edges      =      np.array(rmpc_edges, ndmin = 1)
            rmpc_bins       =      0.5 * (rmpc_edges[1:] + rmpc_edges[:-1])

        # The magnitude cut
        if    mag_lo   is   not   None   and   \
              mag_hi   is   not   None:
            if  name_band   is  None:
                i_am_in_mag_bin     =      ( mag_lo <= self._readinphotcat[ self.names_in_cats["s_band"] ] ) & \
                                           ( self._readinphotcat[ self.names_in_cats["s_band"] ] < mag_hi  )
            else:
                i_am_in_mag_bin     =      ( mag_lo <= self._readinphotcat[ name_band ] ) & \
                                           ( self._readinphotcat[ name_band ] < mag_hi  )
        else:
            i_am_in_mag_bin         =      np.ones( len(self._readinphotcat[ self.names_in_cats["s_band"] ]), dtype = bool)

        # ---
        # Calculate the Ngal and ngal
        # ---
        # hist1d
        Ngal                =       np.histogram(self._readinphotcat["radii_mpc"][ i_am_in_mag_bin ],
                                                 bins = rmpc_edges, range = (rmpc_edges.min(), rmpc_edges.max()))[0]
        rmpc_area           =       (rmpc_edges[1:]**2 - rmpc_edges[:-1]**2) * pi
        ngal                =       Ngal * 1.0 / rmpc_area

        # ---
        # Calculate the completeness_profile
        # ---
        if    use_comp_crrct:
            if   cmplt_prfl  is not None and \
                 cmplt_rmpc  is not None:
                     cmplt_per_ann  =   np.interp(x = rmpc_bins, xp = cmplt_rmpc, fp = cmplt_prfl)
            elif os.path.isfile(self.path2cmplt_map):
                area_weight, cmplt_map, cmplt_per_ann   =   \
                    funcs.completeness_map_profiler(
                        path2img   = self.path2cmplt_map,
                        rac        = self.rac,
                        decc       = self.decc,
                        rmpc_edges = rmpc_edges,
                        mpc2arcmin = 1.0/self.arcmin2mpc )
            else:
                raise IOError("path2cmplt_map is None and we want completeness map, something wrong in initiation.")
        else:
            cmplt_per_ann   =   np.ones(len(rmpc_bins))

        # ---
        # Diagnostic
        # ---
        print
        print "#", "use_comp_crrct:", use_comp_crrct
        print "#", "cmplt_per_ann:", cmplt_per_ann
        print
        print "#", "total objs:", len(i_am_in_mag_bin)
        print "#", "After mag filtering between", mag_lo, "and", mag_hi, ":", \
                   np.sum( i_am_in_mag_bin )
        print

        # plotme?
        if plotme:
            pyplt.figure("ngal", figsize = (6.0, 6.0))
            pyplt.errorbar(rmpc_bins, ngal, yerr = np.sqrt(Ngal) / rmpc_area, fmt = "ko")
            pyplt.show()

        # return
        return rmpc_bins, rmpc_area, Ngal, ngal, cmplt_per_ann


    # ---
    # collect_model_materials
    # ---
    def collect_model_material(self,
            name_band,
            use_core_excised=       False   ,
            use_comp_crrct  =       False   ,
            use_zpoint      =       True    ,
            excised_rmpc    =       1.0     ,
            md_mag_edges    =       np.arange(18.0, 28.0, 0.1),
            ds_mag          =       0.50    ,
            #mag_disp        =       None    ,
            #mag50           =       None    ,
            cmplt_prfl      =       None    ,
            cmplt_rmpc      =       None    ,
            nboot           =       1000    ,
            mag_lo          =       20.0    ,
            mag_hi          =       25.0    ,
            pdz_z_bins      =       np.arange(0.01, 5.0+0.0001, 0.01),
            beta_edges      =       np.arange(0.00,       1.01, 0.01),
            rmpc_edges      =       np.logspace(-1.0, log10(2.0), 11),
            ):
        """
        This function collects the materials that are used to build the magnification model.

        # ---
        # Some notes:
        #
        # 1, every scale in Mpc.
        # 2, it follows the formula of Chiu+16.
        #    nd(rmpc) / fcom(rmpc) = u**(2.5s - 1) n_bkg / fcom_bkg
        #        or
        #    Nd(rmpc)   = u**(2.5s - 1) (n_bkg / fcom_bkg) * A(rmpc) * fcom(rmpc)
        # 3. The observables are Nd(rmpc) where the model is
        #    Nmod(rmpc) = u**(2.5s - 1) (n_bkg / fcom_bkg) * A(rmpc) * fcom(rmpc)
        #    where Nmod = Nmod(rmpc, mass, concentration, beta_bkg, sslope_bkg)
        # ---

        """
        # sanitize
        # radii binning
        rmpc_edges          =       np.array(rmpc_edges, ndmin = 1)
        rmpc_bins           =       0.5 * ( rmpc_edges[1:] + rmpc_edges[:-1] )
        rmpc_area           =       ( rmpc_edges[1:]**2 - rmpc_edges[:-1]**2) * pi
        pdz_z_bins          =       np.array(pdz_z_bins, ndmin = 1)
        beta_edges          =       np.array(beta_edges, ndmin = 1)

        # diagnostic
        print
        print "#", "name_band:", name_band
        print "#", "rmpc_edges:", rmpc_edges
        print "#", "mag_lo:", mag_lo
        print "#", "mag_hi = s_mag:", mag_hi
        print "#", "ds_mag:", ds_mag
        print "#", "md_mag_edges:", md_mag_edges
        print "#", "nboot:", nboot
        print "#"
        print "#", "use_core_excised:", use_core_excised
        print "#", "excised_rmpc:", excised_rmpc
        print "#", "use_comp_crrct:", use_comp_crrct
        #print "#", "mag_disp:", mag_disp
        #print "#", "mag50:", mag50
        print "#", "cmplt_prfl:", cmplt_prfl
        print "#", "cmplt_rmpc:", cmplt_rmpc
        print

        # ---
        # derive sslope
        # ---
        print
        print "#", "deriving sslope ... "
        sslope_bkg, dsslope_bkg = self.derive_sslope(
            name_band       =      name_band    ,
            excised_rmpc    =      excised_rmpc ,
            md_mag_edges    =      md_mag_edges ,
            s_mag           =      mag_hi       ,
            ds_mag          =      ds_mag       ,
            #mag_disp        =      mag_disp     ,
            #mag50           =      mag50        ,
            nboot           =      nboot        ,
            use_core_excised=      use_core_excised,
            use_comp_crrct  =      use_comp_crrct,
            )
        print "#", "Done!"
        print

        # ---
        # derive Pb
        # ---
        print
        print "#", "deriving Pb ... "
        beta_bins, pb_norm, pb, pdz_z_bins, pz_norm, pz =   self.derive_Pbeta(
            name_band       =       name_band    ,
            excised_rmpc    =       excised_rmpc ,
            included_rmpc   =       np.inf       ,
            mag_lo          =       mag_lo       ,
            mag_hi          =       mag_hi       ,
            use_zpoint      =       use_zpoint   ,
            pdz_z_bins      =       pdz_z_bins   ,
            beta_edges      =       beta_edges   ,
            )
        mean_beta           =   np.average(beta_bins, weights = pb)
        print "#", "Done!"
        print

        # ---
        # derive ngal
        # ---
        print
        print "#", "derive ngal ... "
        _, _, Ngal, ngal, cmplt_per_ann     =   self.derive_ngal(
            name_band       =       name_band     ,
            rmpc_edges      =       rmpc_edges    ,
            mag_lo          =       mag_lo        ,
            mag_hi          =       mag_hi        ,
            use_comp_crrct  =       use_comp_crrct,
            cmplt_prfl      =       cmplt_prfl    ,
            cmplt_rmpc      =       cmplt_rmpc    ,
            )
        # nbkg - after correcting the incompleteness.
        if    use_core_excised:
            Nbkg    =   np.sum( (Ngal * 1.0 / cmplt_per_ann)[ (rmpc_bins > excised_rmpc) ] )
            nbkg    =   np.sum( (Ngal * 1.0 / cmplt_per_ann)[ (rmpc_bins > excised_rmpc) ] ) / \
                        np.sum( rmpc_area[ (rmpc_bins > excised_rmpc) ] )
        else:
            Nbkg    =   np.sum( (Ngal * 1.0 / cmplt_per_ann) )
            nbkg    =   np.sum( (Ngal * 1.0 / cmplt_per_ann) ) / \
                        np.sum( rmpc_area )
        print "#", "Done!"
        print


        # ---
        # Contain them in a dict
        # ---
        container   =   {
                "sslope_bkg"        :    np.copy(sslope_bkg     ),
                "dsslope_bkg"       :    np.copy(dsslope_bkg    ),
                "beta_bins"         :    np.copy(beta_bins      ),
                "pb"                :    np.copy(pb             ),
                "pb_norm"           :    np.copy(pb_norm        ),
                "mean_beta"         :    np.copy(mean_beta      ),
                "pdz_z_bins"        :    np.copy(pdz_z_bins     ),
                "pz_norm"           :    np.copy(pz_norm        ),
                "pz"                :    np.copy(pz             ),
                "mean_beta"         :    np.copy(mean_beta      ),
                "Ngal"              :    np.copy(Ngal           ),
                "ngal"              :    np.copy(ngal           ),
                "cmplt_per_ann"     :    np.copy(cmplt_per_ann  ),
                "Nbkg"              :    np.copy(Nbkg           ),
                "nbkg"              :    np.copy(nbkg           ),
                }
        # ---
        # Append the materials
        # ---
        self.material   =   container

        return container


    # ---
    # magnimod
    # ---
    def magnimod(self,
            mass            =       3E14                             ,
            concen          =       3.0                              ,
            sslope          =       1.0                              ,
            nbkg            =       1.0                              ,
            mean_beta       =       None                             ,
            mean_zs         =       2.0                              ,
            rmpc_edges      =       np.logspace(-1.0, log10(2.0), 11),
            cmplt_per_ann   =       None                             ,
            contam_per_ann  =       None                             ,
            Ngal_or_ngal    =       "Ngal"                           ,
            ):
        """
        This function creates the magnification model.

        # ---
        # Some notes:
        #
        # 1, every scale in Mpc.
        # 2, it follows the formula of Chiu+16.
        #    nd(rmpc) / fcom(rmpc) = u**(2.5s - 1) n_bkg / fcom_bkg
        #        or
        #    Nd(rmpc)   = u**(2.5s - 1) (n_bkg / fcom_bkg) * A(rmpc) * fcom(rmpc)
        # 3. The observables are Nd(rmpc) where the model is
        #    Nmod(rmpc) = u**(2.5s - 1) (n_bkg / fcom_bkg) * A(rmpc) * fcom(rmpc)
        #    where Nmod = Nmod(rmpc, mass, concentration, beta_bkg, sslope_bkg)
        # ---

        Parameters:
            -`mass`: float. The halo mass in the unit of Msun.
            -`concen`: float. The halo concentration.
            -`sslope`: float. The power law index of the culmulative counts of the background.
            -`mean_beta`: float. The beta value of the background. If it is None, then calculate the beta from zd and zs.
            -`mean_zs`: float. The redshift of the background.
            -`nbkg`: float. The total observed density of the background after the core excision. IMPORTANT: THIS IS AFTER INCOMPLETENESS CORRECTION.
            -`rmpc_edges`: 1d array. The edges of the radial binnings in the unit of Mpc.
            -`cmplt_per_ann`: 1d array. The profile of the completeness.
            -`contam_per_ann`: 1d array. The profile of the contamination by non-source.
            -`Ngal_or_ngal`: string, "Ngal" or "ngal". Return the number of background galaxies if Ngal. Return the number density of background galaxies if ngal.

        Return:
            -`mod`: the function object calculating number density or number of the background galaxies--including magnitication effect based on the paramters above. Return the number of background galaxies if `Ngal_or_ngal` == "Ngal". Return the number density of background galaxies if `Ngal_or_ngal` == "ngal".

        """
        # sanitize
        mass                =       float(mass  )
        concen              =       float(concen)
        sslope              =       float(sslope)
        nbkg                =       float(nbkg  )
        rmpc_edges          =       np.array(rmpc_edges, ndmin = 1)
        rmpc_bins           =       0.5 * ( rmpc_edges[1:] + rmpc_edges[:-1] )
        rmpc_area           =       ( rmpc_edges[1:]**2 - rmpc_edges[:-1]**2) * pi
        # completeness per annuli
        if     cmplt_per_ann    is   None:
            cmplt_per_ann   =       np.ones(len(rmpc_area))
        else:
            cmplt_per_ann   =       np.array(cmplt_per_ann, ndmin = 1)
        # contamination (by non sources) per annuli
        if     contam_per_ann   is   None:
            contam_per_ann  =       np.zeros(len(rmpc_area))
        else:
            contam_per_ann  =       np.array(contam_per_ann, ndmin = 1)

        # create the NFW object
        halo    =   NFW.Halo(
                    zd      = self.zd,
                    mass    = mass,
                    concen  = concen,
                    overden = self.overden,
                    wrt     = self.wrt,
                    cosmo   = self.cosmo,
                    )
        # calculate mu
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu      =   halo.mu(rmpc_bins, zs = mean_zs, beta = mean_beta)
        # calc the number of background galaxies, Nmod,
        # and the number density of the background galaxies, nmod.
        # Important: this is the model that survives *after* the incompleteness.
        nmod    =   mu**(2.5 * sslope - 1.0) * nbkg * cmplt_per_ann / (1.0 - contam_per_ann)
        Nmod    =   nmod * rmpc_area

        # return
        if     Ngal_or_ngal ==  "Ngal":
            return Nmod
        elif   Ngal_or_ngal ==  "ngal":
            return nmod
        else:
            raise NameError("Ngal_or_ngal has to be either Ngal or ngal:", Ngal_or_ngal)


    # ---
    # calc_cstat
    # ---
    def calc_cstat(self,
            D,
            mass            =       3E14                             ,
            concen          =       3.0                              ,
            sslope          =       1.0                              ,
            nbkg            =       1.0                              ,
            mean_beta       =       None                             ,
            mean_zs         =       2.0                              ,
            rmpc_edges      =       np.logspace(-1.0, log10(2.0), 11),
            cmplt_per_ann   =       None                             ,
            contam_per_ann  =       None                             ,
            ):
        """
        This is the cash statistics estimator, cstat, of Nmod that characterizes the magnification bias profile.
        cstat   =   2 * ( M - D + D * ( ln(D) - ln(M) ) )

        # ---
        # Some notes:
        #
        # 1, every scale in Mpc.
        # 2, it follows the formula of Chiu+16.
        #    nd(rmpc) / fcom(rmpc) = u**(2.5s - 1) n_bkg / fcom_bkg
        #        or
        #    Nd(rmpc)   = u**(2.5s - 1) (n_bkg / fcom_bkg) * A(rmpc) * fcom(rmpc)
        # 3. The observables are Nd(rmpc) where the model is
        #    Nmod(rmpc) = u**(2.5s - 1) (n_bkg / fcom_bkg) * A(rmpc) * fcom(rmpc)
        #    where Nmod = Nmod(rmpc, mass, concentration, beta_bkg, sslope_bkg)
        # ---


        Parameters:
            -`D`: 1d array. The observed number counts.
            -`mass`: float. The halo mass in the unit of Msun.
            -`concen`: float. The halo concentration.
            -`sslope`: float. The power law index of the culmulative counts of the background.
            -`mean_beta`: float. The beta value of the background. If it is None, then calculate the beta from zd and zs.
            -`mean_zs`: float. The redshift of the background.
            -`nbkg`: float. The total observed density of the background after the core excision. IMPORTANT: THIS IS AFTER INCOMPLETENESS CORRECTION.
            -`rmpc_edges`: 1d array. The edges of the radial binnings in the unit of Mpc.
            -`cmplt_per_ann`: 1d array. The profile of the completeness.
            -`contam_per_ann`: 1d array. The profile of the contamination by non-source.
            -`Ngal_or_ngal`: string, "Ngal" or "ngal". Return the number of background galaxies if Ngal. Return the number density of background galaxies if ngal.


        Return:
            -`cstat`: 1d array. The cash statistics estimate of the array.

        """
        # sanitize
        D                   =       np.array(D, ndmin=1)
        mass                =       float(mass  )
        concen              =       float(concen)
        sslope              =       float(sslope)
        nbkg                =       float(nbkg  )
        rmpc_edges          =       np.array(rmpc_edges, ndmin = 1)
        rmpc_bins           =       0.5 * ( rmpc_edges[1:] + rmpc_edges[:-1] )
        rmpc_area           =       ( rmpc_edges[1:]**2 - rmpc_edges[:-1]**2) * pi

        # completeness per annuli
        if     cmplt_per_ann    is   None:
            cmplt_per_ann   =       np.ones(len(rmpc_area))
        else:
            cmplt_per_ann   =       np.array(cmplt_per_ann, ndmin = 1)
        # contamination (by non sources) per annuli
        if     contam_per_ann   is   None:
            contam_per_ann  =       np.zeros(len(rmpc_area))
        else:
            contam_per_ann  =       np.array(contam_per_ann, ndmin = 1)


        # create the model
        M       =   self.magnimod(
                    mass            = mass          ,
                    concen          = concen        ,
                    sslope          = sslope        ,
                    nbkg            = nbkg          ,
                    mean_beta       = mean_beta     ,
                    mean_zs         = mean_zs       ,
                    rmpc_edges      = rmpc_edges    ,
                    cmplt_per_ann   = cmplt_per_ann ,
                    contam_per_ann  = contam_per_ann,
                    Ngal_or_ngal    = "Ngal"        ,
                    )
        # calculate cstat
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cstat                 =   2.0 * ( M - D + D * ( np.log(D) - np.log(M) ) )
            cstat[ (D == 0) ]     =   2.0 * M[ (D == 0) ]

        # return
        return np.sum(cstat)



######################################################
#
# Test
#
######################################################

if      __name__ == "__main__":
    test_cluster    =   GCluster()
    import emcee



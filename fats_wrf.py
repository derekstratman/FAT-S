#!/usr/bin/env python
################################################################################
### fats_wrf.py: A program to compute displacement vectors using FAT-S.      ###
###              This version works with WRF.                                ###
################################################################################

#-------------------------------------------------------------------------------
# System imports

import sys
import os
import datetime as DT
import numpy as np
import math
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab as P
import ctables
import random
from optparse import OptionParser
from time import time as timer
from scipy.optimize import fmin_cg
from scipy.optimize import minimize
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import correlate1d
from scipy import ndimage
from scipy import signal
import netCDF4 as ncdf
import state_vector_wrf as state
import pickle
import skimage
from skimage.measure import label
from skimage.measure import regionprops
from skimage.morphology import convex_hull_image
#from skimage.draw import circle
from fats_functions import *


#-----------------------------------------------------------------------------
# Parameters:
#
# spc - Parameter for which points to compute displacement vectors.
#       e.g. spc = 2 results in using every other gridded obs point
# ds - 2x the spacing
# qs - 4x the spacing
# Si - Displacement magnitude limit
# scls - Gaussian standard deviation used to form smoothed fields
#        for first-guess field of displacement vectors  
# nscl - number of smoothing scales
# Sg1 - Object-merging step's Gaussian smoother standard deviation
# min_dbz - Threshold used on the smoothed forecast and observed fields 
# dbz_thd - Threshold used on original forecast and observed fields
# min_dist - Maximum allowable distance between object centroids (gridpoints)
# min_vect - State variables are updated only if the maximum final displacement
#            vector is greater than this threshold
# max_vect - same as min_dist
# eb - domain edge buffer (gridpoints) 
# sigsq - observational error standard deviation squared
# di - grid spacing of displacement vector grid (gridpoints)
# lams - Smoothness penalty function weight
# lamd - Divergence penalty function weight
# lamm - Magnitude penalty function weight
# lamb - Barrier penalty function weight

spc = 5
ds = spc*2
qs = spc*4
Si = 10.
scls = [10]
nscl = len(scls)
Sg1 = 0.5
min_dbz = 1.0
dbz_thd = 25.
min_dist = 30.
min_vect = 0.05
max_vect = min_dist
eb = spc
sigsq = 2.0**2.
di = float(spc)

lams = 6.0
lamd = 3.0 
lamm = 0.1
lamb = 1.0 

#-----------------------------------------------------------------------------------------------------------------------------------
# Main Program

if __name__ == "__main__":

  if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
 
  stime = timer()
  startTime = DT.datetime.now()

  print("\n----------------------------------------------------------------------")
  print("\n                    BEGIN PROGRAM FAT                                 ")
  print(("\n       WALLCLOCK START TIME:  %s \n" % startTime.strftime("%Y-%m-%d %H:%M:%S")))
  print("\n----------------------------------------------------------------------")

  # Parse input command options
  parser = OptionParser()

  parser.add_option("-w", "--wrffile",    dest="wrffile",   type="string",  help = "WRF forecast file")
  parser.add_option("-o", "--obsfile",    dest="obsfile",   type="string",  help = "gridded observation file")
  parser.add_option("-p", "--plot",       dest="plot",      action="store_true",  help = "plot displacement vectors with displayed field")

  (options, args) = parser.parse_args()

  #-------------------------------------------------------------------------------
  # Get WRF file dataset

  if options.wrffile == None:
    print("\n--> FAT:  No WRF filename specified, exiting....\n")
    parser.print_help()
    sys.exit(-1)
  else:
    fcstdata = ncdf.Dataset(options.wrffile, "r")
    print("\n--> FAT:  Successfully read WRF file.\n")

  #-------------------------------------------------------------------------------
  # Get MRMS file dataset

  if options.obsfile == None:
    print("\n--> FAT:  No MRMS filename specified, exiting....\n")
    parser.print_help()
    sys.exit(-1)
  else:
    obsdata = ncdf.Dataset(options.obsfile, "r")
    print("\n--> FAT:  Successfully read MRMS file.\n")

  #-------------------------------------------------------------------------------
  # Read in observation and forecast fields from data files

  ofld = np.copy(obsdata.variables['refl_consv'])
  ffld = np.amax(fcstdata.variables['REFL_10CM'][0,:,:,:], axis=0)

  nx = len(ffld[0,:])
  ny = len(ffld[:,0])
  nz = len(fcstdata.variables['REFL_10CM'][0,:,0,0])

  #-------------------------------------------------------------------------------
  # Get observartions onto model grid

  #ofld = np.copy(obsdata.variables['maxdbz'])
 
  #-------------------------------------------------------------------------------
  # Get gridded fields for minimization purposes

  eb2 = eb*2
  obs  = np.zeros((ny+qs,nx+qs))
  fcst = np.zeros((ny+qs,nx+qs))
  
  obs[ds:ny+ds,ds:nx+ds] = np.where(ofld > 0.0, ofld, obs[ds:ny+ds,ds:nx+ds])
  fcst[ds:ny+ds,ds:nx+ds] = np.where(ffld > 0.0, ffld, fcst[ds:ny+ds,ds:nx+ds])

  #-----------------------------------------------------------------------------
  # Create i,j grid

  ni = np.arange(float(nx+qs))-ds
  nj = np.arange(float(ny+qs))-ds
  i,j = np.meshgrid(ni,nj)

  a = np.zeros((nscl,ny+qs,nx+qs))
  b = np.zeros((nscl,ny+qs,nx+qs))
  sm_o = np.zeros((nscl,ny+qs,nx+qs))
  sm_f = np.zeros((nscl,ny+qs,nx+qs))
  updf = np.zeros((nscl,ny+qs,nx+qs))
  abmax = np.zeros((nscl))

  #-----------------------------------------------------------------------------
  # Provide initial random guess of displacement vectors

  np.random.seed(47546)
  a0 =  0.05*np.random.random((nscl,ny+qs,nx+qs)) - 0.025
  b0 =  0.05*np.random.random((nscl,ny+qs,nx+qs)) - 0.025

  #-----------------------------------------------------------------------------
  # Loop over multiple appilcations at different scales
  lrmse = []

  for ct in range(0,nscl):

    #---------------------------------------------------------------------------
    # Find the reflectivity center of masses and use them to
    # set estimated displacement vectors for grid points
    # existing with the smoothed observed reflectivity
    
    Sg2 = scls[ct]
    print('\nSg1, Sg2, min_dist, min_vect, max_vect, min_dbz, dbz_thd =',Sg1, Sg2, min_dist, min_vect, max_vect, min_dbz, dbz_thd) 

    if ct > 0:
      updf[ct,:,:] = np.copy(updf[ct-1,:,:])
    else:
      updf[ct,:,:] = np.copy(fcst)

    a0[ct,ds+eb:ny+ds-eb,ds+eb:nx+ds-eb], b0[ct,ds+eb:ny+ds-eb,ds+eb:nx+ds-eb], sm_o[ct,ds+eb:ny+ds-eb,ds+eb:nx+ds-eb], sm_f[ct,ds+eb:ny+ds-eb,ds+eb:nx+ds-eb] = \
                   initialize_vectors(np.copy(obs[ds+eb:ny+ds-eb,ds+eb:nx+ds-eb]), updf[ct,ds+eb:ny+ds-eb,ds+eb:nx+ds-eb], a0[ct,ds+eb:ny+ds-eb,ds+eb:nx+ds-eb], \
                   b0[ct,ds+eb:ny+ds-eb,ds+eb:nx+ds-eb], i[ds+eb:ny+ds-eb,ds+eb:nx+ds-eb], j[ds+eb:ny+ds-eb,ds+eb:nx+ds-eb], Sg1, Sg2, min_dist, min_vect, min_dbz, dbz_thd)
   
    ab0max = math.sqrt(np.amax(a0[ct]**2. + b0[ct]**2.))
    print("\nMaximum initial disp vector: %.2f"%ab0max)

    if ab0max < 0.0:

      a0 = 0.0
      b0 = 0.0

      print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++")
      print(" Either observations or forecast contain no storms  ")
      print(" or initial estimated displacements are < 1*dx,     ") 
      print(" so no displacement vectors are applied to the      ")
      print(" forecast fields.                                   ")
      print("++++++++++++++++++++++++++++++++++++++++++++++++++++")

    else:

      #-------------------------------------------------------------------------
      # Pre-shorten displacement vectors extending beyond the
      # domain edges.

      #a0[ct,eb:ny-eb,eb:nx-eb],b0[ct,eb:ny-eb,eb:nx-eb] = correct_vect(a0[ct,eb:ny-eb,eb:nx-eb],b0[ct,eb:ny-eb,eb:nx-eb])

      #-------------------------------------------------------------------------
      # Flatten and append together the 2D vector arrays

      a0f = a0[ct,::spc,::spc].flatten()
      b0f = b0[ct,::spc,::spc].flatten()
      ab0 = np.append(a0f,b0f)

      #-------------------------------------------------------------------------
      # Gradient check

      #GradCheck(ab0,sm_o[ct,::spc,::spc],sm_f[ct,::spc,::spc],i[::spc,::spc],j[::spc,::spc],sigsq,di,Si,lams,lamd,lamb,lamm)

      #-------------------------------------------------------------------------
      # Perform the minimization

      opts = {'maxiter' : 1000.,
              'disp' : True,
              'gtol' : 1e-5, 
              'norm' : np.inf,
              'eps' : 1.4901161193847656e-08}
      
      print("\n       costr   lams*costs   lamd*costd   lamm*costm   lamb*costb")
      res2 = minimize(f,ab0,jac=gradf,args=(sm_o[ct,::spc,::spc],sm_f[ct,::spc,::spc], \
                   i[::spc,::spc],j[::spc,::spc],sigsq,di,Si,lams,lamd,lamb,lamm), method='CG', \
                   options=opts)
      ab = res2.x 
      #-------------------------------------------------------------------------
      # Unflatten displacement vector arrays

      alpha = 1.0
      ifs = i[::spc,::spc].flatten()
      jfs = j[::spc,::spc].flatten()
      na = len(i[::spc,0])
      nb = len(i[0,::spc])
      ngp = na*nb
      a[ct,ds+eb:ny+ds-eb,ds+eb:nx+ds-eb] = alpha*griddata((ifs,jfs),ab[:ngp],(i[ds+eb:ny+ds-eb,ds+eb:nx+ds-eb],j[ds+eb:ny+ds-eb,ds+eb:nx+ds-eb]),method='linear',fill_value=0.0)
      b[ct,ds+eb:ny+ds-eb,ds+eb:nx+ds-eb] = alpha*griddata((ifs,jfs),ab[ngp:],(i[ds+eb:ny+ds-eb,ds+eb:nx+ds-eb],j[ds+eb:ny+ds-eb,ds+eb:nx+ds-eb]),method='linear',fill_value=0.0)
     
      # Removing original noise in a0,b0
      a[ct,:,:] = np.where((np.absolute(a[ct,:,:]) <= 0.025) & (np.absolute(b[ct,:,:]) <= 0.025), 0.0, a[ct,:,:])
      b[ct,:,:] = np.where((np.absolute(a[ct,:,:]) <= 0.025) & (np.absolute(b[ct,:,:]) <= 0.025), 0.0, b[ct,:,:])

      #print(a[ct,:,50])
      #print(j[:,50])
      #print(a[ct,ds:ny+ds:spc,50])
      #print(j[ds:ny+ds:spc,50])
      #print(a[ct,::spc,50])
      #print(j[::spc,50])

      #-------------------------------------------------------------------------------
      # Post-shorten displacement vectors extending beyond the
      # domain edges.
     
      a[ct,ds:nx+ds,ds:nx+ds],b[ct,ds:nx+ds,ds:nx+ds] = correct_vect(a[ct,ds:nx+ds,ds:nx+ds],b[ct,ds:nx+ds,ds:nx+ds],i[0,ds:nx+ds],j[ds:nx+ds,0])

      #-------------------------------------------------------------------------
      # Regrid plotting variables using displacement vectors

      ia = i + a[ct]
      jb = j + b[ct]
    
      ia[ia < np.amin(i)] = np.amin(i)
      jb[jb < np.amin(j)] = np.amin(j) 
      ia[ia > np.amax(i)] = np.amax(i)
      jb[jb > np.amax(j)] = np.amax(j)
 
      abmax[ct] = math.sqrt(np.amax(a[ct]**2. + b[ct]**2.))

      print("\nMaximum final disp vect: %.2f" %abmax[ct])

      if ct > 0:
        fcstf = updf[ct-1,:,:].flatten()
      else:
        fcstf = fcst.flatten()
      ifl = i.flatten()
      jfl = j.flatten()
     
      if ct > 0:
        interpfunct = RegularGridInterpolator((nj,ni),updf[ct-1,:,:],method='linear')
        updf[ct,:,:] = interpfunct((jb,ia),method='linear')
      else: 
        interpfunct = RegularGridInterpolator((nj,ni),fcst,method='linear')
        updf[ct,:,:] = interpfunct((jb,ia),method='linear')

      rmsk = (obs[:,:]>0.0) | (updf[ct,:,:]>0.0)
      rmse = np.sqrt(np.mean((obs[rmsk]-updf[ct][rmsk])**2))
      lrmse.append(rmse)
      ormsk = (obs[:,:]>0.0) | (fcst[:,:]>0.0)
      ormse = np.sqrt(np.mean((obs[ormsk]-fcst[ormsk])**2))  

      if abmax[ct] < min_vect or abmax[ct] > max_vect:
         print("\nDisplacement vector is too small or too big. No adjustments applied.")
         a[ct] = 0.0
         b[ct] = 0.0
         updf[ct,:,:] = fcst[:,:]
      
      if rmse > ormse:
         print("\nRMSE has increased, so no displacement corrections are applied.")
         print("RMSE before:",ormse)
         print("RMSE after:",rmse)
         a[ct] = 0.0
         b[ct] = 0.0
         updf[ct,:,:] = fcst[:,:]

      snobs = np.count_nonzero(np.where(fcst > dbz_thd, 1.0, 0.0))
      enobs = np.count_nonzero(np.where(updf[ct] > dbz_thd, 1.0, 0.0))

      #-------------------------------------------------------------------------
      # Plotting section

      if options.plot:
         skp = 5 #spc
         fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(11, 3), dpi=150)
         levels = list(range(5,80,5))
         levs2 = list(range(0,80,5))
         levs2[0] = min_dbz
         cmap = ctables.NWSRef
         cmapm = ctables.NWSRef_mod
         ax1.contour(i[ds:ny+ds,ds:nx+ds], j[ds:ny+ds,ds:nx+ds], obs[ds:ny+ds,ds:nx+ds], levels, linewidths=0.25, colors = 'grey', alpha = 1.0)
         ax1.contour(i[ds:ny+ds,ds:nx+ds], j[ds:ny+ds,ds:nx+ds], obs[ds:ny+ds,ds:nx+ds], [dbz_thd], linewidths=1.0, colors = 'black', alpha = 1.0)
         ax1.contourf(i[ds:ny+ds,ds:nx+ds], j[ds:ny+ds,ds:nx+ds], fcst[ds:ny+ds,ds:nx+ds], levels, cmap = cmap, alpha = 1.0)
         ax1.text(54, 4, "RMSE: {:.{prec}f}".format(ormse,prec=3), fontsize=10, ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.5, ec="none"))
         ax1.set_title('Original Reflectivity Fields')
         ax1.set_xlabel('i-index')
         ax1.set_ylabel('j-index')
         ax1.set_aspect(1.0)

         ax2.contour(i[ds:ny+ds,ds:nx+ds], j[ds:ny+ds,ds:nx+ds], sm_o[ct,ds:ny+ds,ds:nx+ds], levs2, linewidths=0.5, cmap = cmap, alpha = 1.0)
         ax2.contourf(i[ds:ny+ds,ds:nx+ds], j[ds:ny+ds,ds:nx+ds], sm_f[ct,ds:ny+ds,ds:nx+ds], levs2, cmap = cmapm, alpha = 0.5)
         ax2.quiver(i[ds:ny+ds:skp,ds:nx+ds:skp], j[ds:ny+ds:skp,ds:nx+ds:skp], -1.*a0[ct,2*spc:ny+2*spc:skp,2*spc:nx+2*spc:skp], -1.*b0[ct,2*spc:ny+2*spc:skp,2*spc:nx+2*spc:skp], units='xy', angles='xy', scale_units='xy', scale=1, pivot='tip', zorder=10)
         ax2.set_title('First Guess Disp Vectors')
         ax2.set_xlabel('i-index')
         ax2.set_aspect(1.0)

         ax3.contour(i[ds:ny+ds,ds:nx+ds], j[ds:ny+ds,ds:nx+ds], obs[ds:ny+ds,ds:nx+ds], [dbz_thd], linewidths=1.0, colors = 'black', alpha = 1.0)
         c3 = ax3.contourf(i[ds:ny+ds,ds:nx+ds], j[ds:ny+ds,ds:nx+ds], updf[ct,ds:ny+ds,ds:nx+ds], levels, cmap = cmap)
         ax3.quiver(i[ds:ny+ds:skp,ds:nx+ds:skp], j[ds:ny+ds:skp,ds:nx+ds:skp], -1.*a[ct,ds:ny+ds:skp,ds:nx+ds:skp], -1.*b[ct,ds:ny+ds:skp,ds:nx+ds:skp], units='xy', angles='xy', scale_units='xy', scale=1, pivot='tip')
         divider = make_axes_locatable(ax3)
         cax3 = divider.append_axes("right", size="5%", pad=0.05)
         fig.colorbar(c3, cax=cax3)
         #ax3.text(4, 4, "%d"%(enobs), fontsize=10, ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.5, ec="none"))
         ax3.text(54, 4, "RMSE: {:.{prec}f}".format(rmse,prec=3), fontsize=10, ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.5, ec="none"))
         ax3.set_title('Updated Reflectivity Field')
         ax3.set_xlabel('i-index')
         ax3.set_aspect(1.0)

         plt.savefig('test_fig_step%s_%s.png'%(ct+1,Sg2), format="png", bbox_inches='tight')

  if options.plot and nscl > 1:
     fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(11, 3))
     ax1.contour(i, j, obs, [dbz_thd], linewidths=1.0, colors = 'black', alpha = 1.0)
     ax1.contourf(i, j, fcst, levels, cmap = cmap, alpha = 1.0)
     ax1.quiver(i[::skp,::skp], j[::skp,::skp], -1.*a[0,::skp,::skp], -1.*b[0,::skp,::skp], units='xy', angles='xy', scale_units='xy', scale=1, pivot='tip')
     ax1.text(54, 4, "RMSE: {:.{prec}f}".format(ormse,prec=3), fontsize=10, ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.5, ec="none"))
     ax1.set_title('Step 1 Disp Vectors')
     ax1.set_xlabel('i-index')
     ax1.set_ylabel('j-index')
     ax1.set_aspect(1.0)     

     ax2.contour(i, j, obs, [dbz_thd], linewidths=1.0, colors = 'black', alpha = 1.0)
     c2 = ax2.contourf(i, j, updf[0], levels, cmap = cmap)
     ax2.quiver(i[::skp,::skp], j[::skp,::skp], -1.*a[1,::skp,::skp], -1.*b[1,::skp,::skp], units='xy', angles='xy', scale_units='xy', scale=1, pivot='tip')
     ax2.text(54, 4, "RMSE: {:.{prec}f}".format(lrmse[0],prec=3), fontsize=10, ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.5, ec="none"))
     ax2.set_title('Step 2 Disp Vectors')
     ax2.set_xlabel('i-index')
     ax2.set_aspect(1.0)

     ax3.contour(i, j, obs, [dbz_thd], linewidths=1.0, colors = 'black', alpha = 1.0)
     c3 = ax3.contourf(i, j, updf[1], levels, cmap = cmap)
     ax3.quiver(i[::skp,::skp], j[::skp,::skp], -1.*np.sum(a[:,::skp,::skp],axis=0), -1.*np.sum(b[:,::skp,::skp],axis=0), units='xy', angles='xy', scale_units='xy', scale=1, pivot='tip')
     divider = make_axes_locatable(ax3)
     cax3 = divider.append_axes("right", size="5%", pad=0.05)
     fig.colorbar(c3, cax=cax3)
     ax3.text(54, 4, "RMSE: {:.{prec}f}".format(lrmse[1],prec=3), fontsize=10, ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.5, ec="none"))
     ax3.set_title('Total Correction')
     ax3.set_xlabel('i-index')
     ax3.set_aspect(1.0)

     plt.savefig('test_fig_allsteps_%s_%s.png'%(scls[0],scls[1]),format="png",bbox_inches='tight')

  #-------------------------------------------------------------------------------
  # Regrid and modify state variables using displacement vectors

  wrffat = options.wrffile.replace("_d01_", "_d01_fat_")
  cmd = "cp %s %s" % (options.wrffile, wrffat)
  print("\n %s"%cmd)
  os.system(cmd)
  
  if np.amax(abmax[:]) > min_vect: 
     print("\nUpdating state variables to",wrffat)
     mod_wrfout(wrffat, state.nssl2mom, np.sum(a[:,ds:ny+ds,ds:nx+ds],axis=0), np.sum(b[:,ds:ny+ds,ds:nx+ds],axis=0), i[ds:ny+ds,ds:nx+ds], j[ds:ny+ds,ds:nx+ds], nx, ny, nz)
  else:
     print("\nNot updating state variables...")
     print("%.2f < %.2f or %.2f > %.2f"%(np.amax(abmax[:]),min_vect,np.amax(abmax[:]),max_vect))

  dispvect = options.wrffile.replace("wrfout_d01", "dispvect")
  write_dispvect(dispvect,a[:,ds:ny+ds,ds:nx+ds],b[:,ds:ny+ds,ds:nx+ds],scls,nx,ny,nscl,spc,Si,Sg1,min_dist,min_vect,max_vect,min_dbz,dbz_thd,eb,sigsq,lams,lamd,lamm,lamb) 

  #-----------------------------------------------------------------------------
  # Print out Wallclock time for FAT

  endTime = DT.datetime.now()
  runTime = endTime - startTime

  print("\n----------------------------------------------------------------------")
  print("\n                        END PROGRAM FAT                               ")
  print(("\n           WALLCLOCK END TIME:  %s " % endTime.strftime("%Y-%m-%d %H:%M:%S")))
  print(("\n              Time for completion: %s " % runTime ))
  print("\n----------------------------------------------------------------------\n")

#-------------------------------------------------------------------------------
# Thank you, come again
#-------------------------------------------------------------------------------

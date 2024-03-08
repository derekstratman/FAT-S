#!/usr/bin/env python
################################################################################
### fats_functions.py: Python functions for FAT-S.                           ###
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

#-------------------------------------------------------------------------------
# Function to compute cost function

def f(ab,y,x,i,j,sigsq,di,Si,lams,lamd,lamb,lamm):
    ny = len(i[:,0])
    nx = len(i[0,:])
    i1d = i[0,:]
    j1d = j[:,0]
    lj = ny-2
    li = nx-2
    ngp = nx*ny
    xf = x.flatten()
    ifl = i.flatten()
    jfl = j.flatten()

    a = np.reshape(ab[:ngp],(ny,nx))
    b = np.reshape(ab[ngp:],(ny,nx))
    ia = i + a
    jb = j + b

    interpfunct = RegularGridInterpolator((j1d,i1d),x,method='linear',bounds_error=False,fill_value=0.0) 
    xt = interpfunct((jb,ia),method='linear')
   
    # residual cost function
    costr = np.sum((1./sigsq)*(y[2:lj,2:li]-xt[2:lj,2:li])**2.)

    # smoothness penalty function
    costs = np.sum(((a[3:lj+1,2:li]-2.*a[2:lj,2:li]+a[1:lj-1,2:li])/di**2. +  \
                    (a[2:lj,3:li+1]-2.*a[2:lj,2:li]+a[2:lj,1:li-1])/di**2.)**2.) \
          + np.sum(((b[3:lj+1,2:li]-2.*b[2:lj,2:li]+b[1:lj-1,2:li])/di**2. +  \
                    (b[2:lj,3:li+1]-2.*b[2:lj,2:li]+b[2:lj,1:li-1])/di**2.)**2.)

    # divergence penalty function
    costd = np.sum(((a[2:lj,3:li+1]-a[2:lj,1:li-1])/(2.*di) + \
                    (b[3:lj+1,2:li]-b[1:lj-1,2:li])/(2.*di))**2.)
           
    # magnitude penalty function
    costm = np.sum((a[2:lj,2:li]/Si)**2.) + np.sum((b[2:lj,2:li]/Si)**2.)
 
    # barrier penalty function
    costb = np.sum((a[2:lj,2:li]/Si)**20.) + np.sum((b[2:lj,2:li]/Si)**20.)

    print("%12.3f %12.3f %12.3f %12.3f %12.3f" %(costr,lams*costs,lamd*costd,lamm*costm,lamb*costb))

    # total cost function
    totcost = costr + lams*costs + lamd*costd + lamm*costm + lamb*costb

    #print("CostF:",totcost)

    return totcost

#-------------------------------------------------------------------------------
# Function to compute gradient of cost function

def gradf(ab,y,x,i,j,sigsq,di,Si,lams,lamd,lamb,lamm):
    ny = len(i[:,0])
    nx = len(i[0,:])
    i1d = i[0,:]
    j1d = j[:,0]
    lj = ny-2
    li = nx-2
    igradr = np.zeros((ny,nx))
    igrads = np.zeros((ny,nx))
    igradd = np.zeros((ny,nx))
    igradm = np.zeros((ny,nx))
    igradb = np.zeros((ny,nx))
    igradt = np.zeros((ny,nx))
    jgradr = np.zeros((ny,nx))
    jgrads = np.zeros((ny,nx))
    jgradd = np.zeros((ny,nx))
    jgradm = np.zeros((ny,nx))
    jgradb = np.zeros((ny,nx))
    jgradt = np.zeros((ny,nx))
    xt = np.zeros((ny,nx))
    ngp = nx*ny
    ifl = i.flatten()
    jfl = j.flatten()
    xf = x.flatten()

    a = np.reshape(ab[:ngp],(ny,nx))
    b = np.reshape(ab[ngp:],(ny,nx))
    ia = i + a
    jb = j + b

    iam1 = ia - 0.01*a
    iap1 = ia + 0.01*a
    jbm1 = jb - 0.01*b
    jbp1 = jb + 0.01*b

    interpfunct = RegularGridInterpolator((j1d,i1d),x,method='linear',bounds_error=False,fill_value=0.0) 
    xt = interpfunct((jb,ia),method='linear')
    xtim1 = interpfunct((jb,iam1),method='linear')
    xtip1 = interpfunct((jb,iap1),method='linear')
    xtjm1 = interpfunct((jbm1,ia),method='linear')
    xtjp1 = interpfunct((jbp1,ia),method='linear')

    # gradient of residual cost function  
    igradr[2:lj,2:li] = 2.*(1./sigsq)*(xt[2:lj,2:li]-y[2:lj,2:li]) * \
                        ((xtip1[2:lj,2:li]-xtim1[2:lj,2:li])/(2.*0.01*a[2:lj,2:li]))
    jgradr[2:lj,2:li] = 2.*(1./sigsq)*(xt[2:lj,2:li]-y[2:lj,2:li]) * \
                        ((xtjp1[2:lj,2:li]-xtjm1[2:lj,2:li])/(2.*0.01*b[2:lj,2:li]))

    # gradient of smoothness penalty function
    igrads[2:lj,2:li] = (-4.) * ((1./di**2.)+(1./di**2.)) *                                \
                        ((a[3:lj+1,2:li]-2.*a[2:lj,2:li]+a[1:lj-1,2:li])/di**2. +          \
                         (a[2:lj,3:li+1]-2.*a[2:lj,2:li]+a[2:lj,1:li-1])/di**2.)           \
                      + (2.) * (1./di**2.) *                                               \
                        ((a[4:lj+2,2:li]-2.*a[3:lj+1,2:li]+a[2:lj,2:li])/di**2. +          \
                         (a[3:lj+1,3:li+1]-2.*a[3:lj+1,2:li]+a[3:lj+1,1:li-1])/di**2.)     \
                      + (2.) * (1./di**2.) *                                               \
                        ((a[2:lj,2:li]-2.*a[1:lj-1,2:li]+a[:lj-2,2:li])/di**2. +           \
                         (a[1:lj-1,3:li+1]-2.*a[1:lj-1,2:li]+a[1:lj-1,1:li-1])/di**2.)     \
                      + (2.) * (1./di**2.) *                                               \
                        ((a[3:lj+1,3:li+1]-2.*a[2:lj,3:li+1]+a[1:lj-1,3:li+1])/di**2. +    \
                         (a[2:lj,4:li+2]-2.*a[2:lj,3:li+1]+a[2:lj,2:li])/di**2.)           \
                      + (2.) * (1./di**2.) *                                               \
                        ((a[3:lj+1,1:li-1]-2.*a[2:lj,1:li-1]+a[1:lj-1,1:li-1])/di**2. +    \
                         (a[2:lj,2:li]-2.*a[2:lj,1:li-1]+a[2:lj,:li-2])/di**2.)
    jgrads[2:lj,2:li] = (-4.) * ((1./di**2.)+(1./di**2.)) *                                \
                        ((b[3:lj+1,2:li]-2.*b[2:lj,2:li]+b[1:lj-1,2:li])/di**2. +          \
                         (b[2:lj,3:li+1]-2.*b[2:lj,2:li]+b[2:lj,1:li-1])/di**2.)           \
                      + (2.) * (1./di**2.) *                                               \
                        ((b[4:lj+2,2:li]-2.*b[3:lj+1,2:li]+b[2:lj,2:li])/di**2. +          \
                         (b[3:lj+1,3:li+1]-2.*b[3:lj+1,2:li]+b[3:lj+1,1:li-1])/di**2.)     \
                      + (2.) * (1./di**2.) *                                               \
                        ((b[2:lj,2:li]-2.*b[1:lj-1,2:li]+b[:lj-2,2:li])/di**2. +           \
                         (b[1:lj-1,3:li+1]-2.*b[1:lj-1,2:li]+b[1:lj-1,1:li-1])/di**2.)     \
                      + (2.) * (1./di**2.) *                                               \
                        ((b[3:lj+1,3:li+1]-2.*b[2:lj,3:li+1]+b[1:lj-1,3:li+1])/di**2. +    \
                         (b[2:lj,4:li+2]-2.*b[2:lj,3:li+1]+b[2:lj,2:li])/di**2.)           \
                      + (2.) * (1./di**2.) *                                               \
                        ((b[3:lj+1,1:li-1]-2.*b[2:lj,1:li-1]+b[1:lj-1,1:li-1])/di**2. +    \
                         (b[2:lj,2:li]-2.*b[2:lj,1:li-1]+b[2:lj,:li-2])/di**2.)

    # gradient of divergence penalty function
    igradd[2:lj,2:li] = 0.0                                        \
                      - (a[2:lj,4:li+2]-a[2:lj,2:li])/(2.*di**2.)  \
                      + (a[2:lj,2:li]-a[2:lj,:li-2])/(2.*di**2.)  
    jgradd[2:lj,2:li] = 0.0                                        \
                      - (b[4:lj+2,2:li]-b[2:lj,2:li])/(2.*di**2.)  \
                      + (b[2:lj,2:li]-b[:lj-2,2:li])/(2.*di**2.)            

    # gradient of magnitude penalty function
    igradm[2:lj,2:li] = 2.*(a[2:lj,2:li]/Si**2.)
    jgradm[2:lj,2:li] = 2.*(b[2:lj,2:li]/Si**2.)

    # gradient of barrier penalty function
    igradb[2:lj,2:li] = 20.*(a[2:lj,2:li]**19./Si**20.)
    jgradb[2:lj,2:li] = 20.*(b[2:lj,2:li]**19./Si**20.)

    # total gradient of cost function
    igradt = igradr + lams*igrads + lamd*igradd + lamm*igradm + lamb*igradb
    jgradt = jgradr + lams*jgrads + lamd*jgradd + lamm*jgradm + lamb*jgradb

    igradtf = igradt.flatten()
    jgradtf = jgradt.flatten()    

    ijgradt = np.append(igradtf,jgradtf)

    return ijgradt

#-------------------------------------------------------------------------------
# Function to correct displacement vectors that extend
# beyond the domain edges. Displacement vectors that do
# extend beyond the domain edges are shortened while
# preserving the angle of the vector.

def correct_vect(a,b,ni,nj):
  ny = len(a[:][0])
  nx = len(a[0][:])
  i,j = np.meshgrid(ni,nj)
  ia = i + a
  jb = j + b
  for ii in range(0, nx):
    for jj in range(0, ny):
      if ia[jj,ii] > np.amax(ni):
        b[jj,ii] = b[jj,ii]*(a[jj,ii] - (ia[jj,ii] - (np.amax(ni))))/a[jj,ii]
        a[jj,ii] = a[jj,ii] - (ia[jj,ii] - (np.amax(ni)))
        ia[jj,ii] = i[jj,ii] + a[jj,ii]
        jb[jj,ii] = j[jj,ii] + b[jj,ii]
      if jb[jj,ii] > np.amax(nj):
        a[jj,ii] = a[jj,ii]*(b[jj,ii] - (jb[jj,ii] - (np.amax(nj))))/b[jj,ii]
        b[jj,ii] = b[jj,ii] - (jb[jj,ii] - (np.amax(nj)))
        ia[jj,ii] = i[jj,ii] + a[jj,ii]
        jb[jj,ii] = j[jj,ii] + b[jj,ii]
      if ia[jj,ii] < np.amin(ni):
        b[jj,ii] = b[jj,ii]*(a[jj,ii] - (ia[jj,ii] - np.amin(ni)))/a[jj,ii]
        a[jj,ii] = a[jj,ii] - (ia[jj,ii] - np.amin(ni))
        ia[jj,ii] = i[jj,ii] + a[jj,ii]
        jb[jj,ii] = j[jj,ii] + b[jj,ii]
      if jb[jj,ii] < np.amin(nj):
        a[jj,ii] = a[jj,ii]*(b[jj,ii] - (jb[jj,ii] - np.amin(nj)))/b[jj,ii]
        b[jj,ii] = b[jj,ii] - (jb[jj,ii] - np.amin(nj))
        ia[jj,ii] = i[jj,ii] + a[jj,ii]
        jb[jj,ii] = j[jj,ii] + b[jj,ii]

  return a, b

#-------------------------------------------------------------------------------
# Function to smooth 2D observed and forecast fields

def gauss_smooth(obs, fcst, sigma, dbz_thd):

    sm_o = obs
    #mx_o = 0.5*np.amax(obs)
    #sm_o[np.where(sm_o < mx_o)] = 0.0
    sm_o[np.where(sm_o < dbz_thd)] = 0.0

    sm_f = fcst
    #mx_f = 0.5*np.amax(fcst)
    #sm_f[np.where(sm_f < mx_f)] = 0.0
    sm_f[np.where(sm_f < dbz_thd)] = 0.0

    if sigma > 0.:
       #mpo = 200.0/np.amax(gaussian_filter(sm_o, sigma=sigma, mode='constant'))
       #mpf = 200.0/np.amax(gaussian_filter(sm_f, sigma=sigma, mode='constant'))
       mpo = np.amax(obs)/np.amax(gaussian_filter(sm_o, sigma=sigma, mode='constant'))
       mpf = np.amax(obs)/np.amax(gaussian_filter(sm_f, sigma=sigma, mode='constant'))
       sm_o = mpo*gaussian_filter(sm_o, sigma=sigma, mode='constant')
       sm_f = mpf*gaussian_filter(sm_f, sigma=sigma, mode='constant')
    
    if sigma == 0.:
       mpo = np.amax(obs)/np.amax(sm_o)
       mpf = np.amax(obs)/np.amax(sm_f)
       sm_o = mpo*sm_o
       sm_f = mpf

    return sm_o, sm_f

#-------------------------------------------------------------------------------
# Function to smooth 2D observed and forecast fields

def super_gauss_smooth(obs, fcst, sigma, dbz_thd):

    sm_o = obs
    #mx_o = 0.9*np.amax(sm_o)
    sm_o[np.where(sm_o < dbz_thd)] = 0.0

    sm_f = fcst
    #mx_f = 0.9*np.amax(sm_f)
    sm_f[np.where(sm_f < dbz_thd)] = 0.0

    pfact = 2.0
    radius = int(4.0 * sigma + 0.5)
    xrad = np.arange(-radius, radius+1)
    wgt = np.exp(-1.0*((xrad**2)/(2.0*(sigma**2)))**pfact)
    wgt = wgt/wgt.sum()

    gsmo = np.copy(sm_o)*0.0
    gsmf = np.copy(sm_f)*0.0

    for jj in range(0, len(sm_o[:][0])):
       gsmo[jj,:] = correlate1d(sm_o[jj,:],wgt[::-1])
       gsmf[jj,:] = correlate1d(sm_f[jj,:],wgt[::-1])

    for ii in range(0, len(sm_o[0][:])):
       gsmo[:,ii] = correlate1d(gsmo[:,ii],wgt[::-1])
       gsmf[:,ii] = correlate1d(gsmf[:,ii],wgt[::-1])

    mpo = np.amax(obs)/np.amax(gsmo)
    mpf = np.amax(obs)/np.amax(gsmf)
    sm_o = mpo*gsmo
    sm_f = mpf*gsmf

    return sm_o, sm_f

#-------------------------------------------------------------------------------
# Function to match observed and forecast objects

def object_match(obs, fcst, min_dist, min_dbz):

    # Find observation objects
    obs_temp = np.where(obs >= min_dbz, obs, 0.)
    obs_binary = np.where(obs >= min_dbz, 1, 0)
    obs_binary = obs_binary.astype(int)
    obs_labels, obs_num = label(obs_binary,return_num=True)
    obs_props = regionprops(obs_labels,obs_temp)

    # Find forecast objects
    fcst_temp = np.where(fcst >= min_dbz, fcst, 0.)
    fcst_binary = np.where(fcst >= min_dbz, 1, 0)
    fcst_binary = fcst_binary.astype(int)
    fcst_labels, fcst_num = label(fcst_binary,return_num=True)
    fcst_props = regionprops(fcst_labels,fcst_temp)

    print("\nNumber of obs objects:  %s"%(obs_num))
    print("Number of fcst objects: %s"%(fcst_num))

    if obs_num > 0 and fcst_num > 0:
  
      mxps = max(obs_num,fcst_num)
      tempof = np.zeros((obs_num,7))
      tempfo = np.zeros((fcst_num,7))
 
      for ii in range(0, obs_num):
        temp_ob_x = obs_props[ii].centroid[1]
        temp_ob_y = obs_props[ii].centroid[0]
        temp_min = min_dist

        # object area hard coded... for now
        if obs_props[ii].area > 24:
          for jj in range(0, fcst_num):
            temp_fc_x = fcst_props[jj].centroid[1]
            temp_fc_y = fcst_props[jj].centroid[0]

            temp_dis = np.sqrt((temp_fc_x - temp_ob_x)**2. + (temp_fc_y - temp_ob_y)**2.)

            if temp_dis < temp_min:
              #print(temp_ob_x,temp_ob_y,temp_fc_x,temp_fc_y)
              temp_min = temp_dis
              tempof[ii,:] = [ii+1, jj+1, temp_min, temp_ob_x, temp_ob_y, temp_fc_x, temp_fc_y]

      for jj in range(0, fcst_num):
        temp_fc_x = fcst_props[jj].centroid[1]
        temp_fc_y = fcst_props[jj].centroid[0]
        temp_min = min_dist

        # object area hard coded... for now
        if fcst_props[jj].area > 24:
          for ii in range(0, obs_num):
            temp_ob_x = obs_props[ii].centroid[1]
            temp_ob_y = obs_props[ii].centroid[0]

            temp_dis = np.sqrt((temp_fc_x - temp_ob_x)**2. + (temp_fc_y - temp_ob_y)**2.)

            if temp_dis < temp_min:
              temp_min = temp_dis
              tempfo[jj,:] = [ii+1, jj+1, temp_min, temp_ob_x, temp_ob_y, temp_fc_x, temp_fc_y]

    
      nummo = min(np.count_nonzero(np.unique(tempof[:,1])),np.count_nonzero(np.unique(tempfo[:,0])))
      matchobj = np.zeros((nummo,7))
 
      cnt = 0
      for ii in range(0, obs_num):
        for jj in range(0, fcst_num):
          if tempof[ii,0] == tempfo[jj,0] and tempof[ii,1] == tempfo[jj,1] and tempof[ii,0] > 0.:
            matchobj[cnt,:] = tempof[ii,:]
            cnt = cnt + 1
      
    else:

      matchobj = np.full((1,7),-1.)
      print("\nobject_match: No matched objects. ")

    return matchobj, obs_labels, fcst_labels

#-------------------------------------------------------------------------------
# Function to initialize 2-D field of displacements vectors
# using object matching

def initialize_vectors(obs, fcst, a0, b0, i, j, Sg1, Sg2, min_dist, min_vect, min_dbz, dbz_thd):

    ny = len(a0[:][0])
    nx = len(b0[0][:])
    print(nx)
  
    tmp_o, tmp_f = gauss_smooth(obs, fcst, Sg1, dbz_thd)
    
    ma_obj1, tmp_o_lab, tmp_f_lab = object_match(tmp_o, tmp_f, min_dist, min_dbz)
    
    num_ma = sum(i > 0 for i in ma_obj1[:,0])

    print("\nNumber of matched useable objects:  %s"%(num_ma))

    mask_o = obs*0.
    mask_f = fcst*0.

    if num_ma > 0:

      tmpa0 = np.zeros((num_ma,ny,nx))
      tmpb0 = np.zeros((num_ma,ny,nx))
      tmpsm_o = np.zeros((num_ma,ny,nx))
      tmpsm_f = np.zeros((num_ma,ny,nx))
      wftmp = np.zeros((num_ma,ny,nx))

      for ii in range(0, num_ma):
        tmpa0[ii,:,:] = 0.0 
        tmpb0[ii,:,:] = 0.0 

        mask_o = 0.
        mask_f = 0.
      
        mask_o = np.where(tmp_o_lab == ma_obj1[ii,0], obs, mask_o)
        mask_f = np.where(tmp_f_lab == ma_obj1[ii,1], fcst, mask_f)
      
        tmpsm_o[ii,:,:], tmpsm_f[ii,:,:] = gauss_smooth(mask_o, mask_f, Sg2, dbz_thd)
        
        ma_obj2, sm_o_lab, sm_f_lab = object_match(tmpsm_o[ii,:,:], tmpsm_f[ii,:,:], 0.5*np.amax([ny,nx]), min_dbz)

        sm_o_binary = np.where(tmpsm_o[ii,:,:] >= min_dbz, 1, 0)
        sm_f_binary = np.where(tmpsm_f[ii,:,:] >= min_dbz, 1, 0)
 
        tmpfld = sm_f_lab*0
        tmpfld = np.where(sm_f_lab == ma_obj2[0,1], sm_f_binary, tmpfld)
        tmpfld = np.where(sm_o_lab == ma_obj2[0,0], sm_o_binary, tmpfld)

        if ii > 0:
          chull = False

        chull = convex_hull_image(tmpfld)
        chtemp = np.where(chull == True, 1.0, 0.0)
  
        ag = ma_obj2[0,5] - ma_obj2[0,3]
        bg = ma_obj2[0,6] - ma_obj2[0,4]

        if ag != 0.:
          tmpa0[ii][chtemp == 1.0] = ag
        if bg != 0.:
          tmpb0[ii][chtemp == 1.0] = bg

        if ag != 0.:
          wftmp[ii] = np.where(tmpa0[ii] == ag, 1.0*np.exp(-1.*(((i-ma_obj2[0,5])**2.)/(2.*((nx-1.)/8.)**2.)+((j-ma_obj2[0,6])**2.)/(2.*((nx-1.)/8.)**2.))), 0.0)
        elif bg != 0.:
          wftmp[ii] = np.where(tmpb0[ii] == bg, 1.0*np.exp(-1.*(((i-ma_obj2[0,5])**2.)/(2.*((nx-1.)/8.)**2.)+((j-ma_obj2[0,6])**2.)/(2.*((nx-1.)/8.)**2.))), 0.0)

        # Remove forecast objects due to larger RMSE
        ttmpa0 = np.copy(tmpa0[ii])
        ttmpb0 = np.copy(tmpb0[ii])
        ttmpa0,ttmpb0 = correct_vect(ttmpa0,ttmpb0,i[0,:],j[:,0])
        tmpia = i + ttmpa0
        tmpjb = j + ttmpb0
        interpfunct = RegularGridInterpolator((j[:,0],i[0,:]),mask_f,method='linear')
        updt = interpfunct((tmpjb,tmpia),method='linear')
        ab0max = math.sqrt(np.amax(ttmpa0**2. + ttmpb0**2.))

        trmsk = (mask_o > 0.0) | (mask_f > 0.0)
        trmse = np.sqrt(np.mean((mask_o[trmsk]-mask_f[trmsk])**2))
        urmsk = (mask_o > 0.0) | (updt > 0.0)
        urmse = np.sqrt(np.mean((mask_o[urmsk]-updt[urmsk])**2))
        print("RMSEs for storm %s: %.2f => %.2f" %(ii,trmse,urmse))
        print("Storm initial disp vector: %.2f"%ab0max)

        if urmse > trmse:
           print("Note, RMSE increased.")

      wgtfct = np.zeros((num_ma,ny,nx))
      cntwf = np.count_nonzero(wftmp,axis=0)
      for ii in range(0, num_ma):
        tmpwf = np.where(cntwf > 1., wftmp[ii], 1.0)
        wgtfct[ii] = np.where(np.sqrt(tmpa0[ii]**2.+tmpb0[ii]**2.) > 0.0001, tmpwf, 0.0)

      a0 = np.where(cntwf==0, a0, np.sum(tmpa0*wgtfct/np.sum(wgtfct,axis=0),axis=0))
      b0 = np.where(cntwf==0, b0, np.sum(tmpb0*wgtfct/np.sum(wgtfct,axis=0),axis=0))

      wftmp = 0.0*tmpa0
      
      wgtfct = np.zeros((ny,nx))
      wgtfct = np.where(np.count_nonzero(tmpa0,axis=0) <= 1, 1.0, 0.0)
       
 
      sm_o = np.amax(tmpsm_o,0)
      sm_f = np.amax(tmpsm_f,0)

    else:

      print("\ninitialize_vectors: No vector adjustments made.")

      sm_o, sm_f = gauss_smooth(mask_o, mask_f, Sg2, dbz_thd)

    return a0, b0, sm_o, sm_f

#-------------------------------------------------------------------------------
# Function to read in and modify wrfout file

def mod_wrfout(file, state_vector, a, b, i, j, nx, ny, nz):

  nxy2d = state_vector['nxy2d']
  nxyz3d = state_vector['nxyz3d']

  try:
    f = ncdf.Dataset(file,"r+")
  except:
    print(("\n==> IO ERROR\n\n ==> mod_wrfout:  Opening dataset %s failed!!!\n    EXITING!!! \n\n --> IO ERROR\n" % file))
    sys.exit(1)

  ni = i[0,:]
  nj = j[:,0]
  
  #a,b = correct_vect(a,b)

  ifl = i.flatten()
  jfl = j.flatten()
  
  ia = i + a
  jb = j + b

  stxyvar = np.zeros((nxy2d,ny,nx))  
  adxyvar = np.zeros((nxy2d,ny,nx))
  stvar = np.zeros((nxyz3d,nz,ny,nx))
  advar = np.zeros((nxyz3d,nz,ny,nx))

  # update 2D variables
  for m, key in enumerate(state_vector['xy2d']):
    hkey = state_vector[key]['name']
    if state_vector[key]['writeback']:
      print("Updating variable: %s" %(key))
      stxyvar[m,:,:] = f[hkey][0,:,:]
      interpfunct = RegularGridInterpolator((nj,ni),stxyvar[m,:,:], method='linear', fill_value=0.0)
      adxyvar[m,:,:] = interpfunct((jb,ia),method='linear')
      f[hkey][0,:,:] = adxyvar[m,:,:]

  # update 3D variables
  for m, key in enumerate(state_vector['xyz3d']):
    hkey = state_vector[key]['name']
    if state_vector[key]['writeback']:
      print("Updating variable: %s" %(key))

      if key == "U":
        u = f[hkey][0,:,:,:]
        stvar[m,:,:,:] = CgridtoAgrid(u, key, nx, ny, nz)
        for kk in range(nz):
          interpfunct = RegularGridInterpolator((nj,ni),stvar[m,kk,:,:], method='linear', fill_value=0.0)
          advar[m,kk,:,:] = interpfunct((jb,ia),method='linear')
        f[hkey][0,:,:,:] = AgridtoCgrid(advar[m], u, key, nx, ny, nz)

      elif key == "V":
        v = f[hkey][0,:,:,:]
        stvar[m,:,:,:] = CgridtoAgrid(v, key, nx, ny, nz)
        for kk in range(nz):
          interpfunct = RegularGridInterpolator((nj,ni),stvar[m,kk,:,:], method='linear', fill_value=0.0)
          advar[m,kk,:,:] = interpfunct((jb,ia),method='linear')
        f[hkey][0,:,:,:] = AgridtoCgrid(advar[m], v, key, nx, ny, nz)

      elif key == "W":
        w = f[hkey][0,:,:,:]
        stvar[m,:,:,:] = CgridtoAgrid(w, key, nx, ny, nz)
        for kk in range(nz):
          interpfunct = RegularGridInterpolator((nj,ni),stvar[m,kk,:,:], method='linear', fill_value=0.0)
          advar[m,kk,:,:] = interpfunct((jb,ia),method='linear')
        f[hkey][0,:,:,:] = AgridtoCgrid(advar[m], w, key, nx, ny, nz)

      elif key == "PH":
        ph = f[hkey][0,:,:,:]
        stvar[m,:,:,:] = CgridtoAgrid(ph, key, nx, ny, nz)
        for kk in range(nz):
          interpfunct = RegularGridInterpolator((nj,ni),stvar[m,kk,:,:], method='linear', fill_value=0.0)
          advar[m,kk,:,:] = interpfunct((jb,ia),method='linear')
        f[hkey][0,:,:,:] = AgridtoCgrid(advar[m], ph, key, nx, ny, nz)

      else:
        stvar[m,:,:,:] = f[hkey][0,:,:,:]
        for kk in range(nz):
          interpfunct = RegularGridInterpolator((nj,ni),stvar[m,kk,:,:], method='linear', fill_value=0.0)
          advar[m,kk,:,:] = interpfunct((jb,ia),method='linear')
        f[hkey][0,:,:,:] = advar[m,:,:,:]

  f.sync()
  f.close()

  return

#-------------------------------------------------------------------------------
# Function to convert from C grid to A grid

def CgridtoAgrid(cvar, key, nx, ny, nz):

  avar = np.zeros((nz,ny,nx))

  if key == "U":
    avar[:,:,0]      = 0.5*(cvar[:,:,0] + cvar[:,:,1])
    avar[:,:,nx-1]     = 0.5*(cvar[:,:,nx-1] + cvar[:,:,nx])
    avar[:,:,1:nx-1] = (-cvar[:,:,0:nx-2]  + 13.0*cvar[:,:,1:nx-1] \
                          -cvar[:,:,3:nx+1]  + 13.0*cvar[:,:,2:nx] ) / 24.0

  if key == "V":
    avar[:,0,:]      = 0.5*(cvar[:,0,:] + cvar[:,1,:])
    avar[:,ny-1,:]   = 0.5*(cvar[:,ny-1,:] + cvar[:,ny,:])
    avar[:,1:ny-1,:] = (-cvar[:,0:ny-2,:] + 13.0*cvar[:,1:ny-1,:] \
                        -cvar[:,3:ny+1,:] + 13.0*cvar[:,2:ny,:] ) / 24.0

  if key == "W" or key == "PH":
    avar[0,:,:]      = 0.5*(cvar[0,:,:] + cvar[1,:,:])
    avar[nz-1,:,:]   = 0.5*(cvar[nz-1,:,:] + cvar[nz,:,:])
    avar[1:nz-1,:,:] = (-cvar[0:nz-2,:,:] + 13.0*cvar[1:nz-1,:,:] \
                        -cvar[3:nz+1,:,:] + 13.0*cvar[2:nz,:,:] ) / 24.0

  return avar

#-------------------------------------------------------------------------------
# Function to convert from A grid to C grid

def AgridtoCgrid(avar, ocvar, key, nx, ny, nz):

  tmp = np.zeros((nz,ny,nx))

  if key == "U":
    cvar = ocvar
    tmp[:,:,0]      = avar[:,:,0]      - 0.5*(ocvar[:,:,0]    + ocvar[:,:,1])
    tmp[:,:,nx-1]   = avar[:,:,nx-1]   - 0.5*(ocvar[:,:,nx-1] + ocvar[:,:,nx])
    tmp[:,:,1:nx-1] = avar[:,:,1:nx-1] - \
                      (-ocvar[:,:,0:nx-2] + 13.0*ocvar[:,:,1:nx-1] \
                       -ocvar[:,:,3:nx+1] + 13.0*ocvar[:,:,2:nx] ) / 24.0
    cvar[:,:,1]      = ocvar[:,:,1]    + 0.5*(tmp[:,:,0]    + tmp[:,:,1])
    cvar[:,:,nx-1]   = ocvar[:,:,nx-1] + 0.5*(tmp[:,:,nx-2] + tmp[:,:,nx-1])
    cvar[:,:,2:nx-1] = ocvar[:,:,2:nx-1] + \
                       (-tmp[:,:,0:nx-3] + 7.0*(tmp[:,:,1:nx-2] + tmp[:,:,2:nx-1]) - tmp[:,:,3:nx] ) / 12.0

  if key == "V":
    cvar = ocvar
    tmp[:,0,:]      = avar[:,0,:]      - 0.5*(ocvar[:,0,:]    + ocvar[:,1,:])
    tmp[:,ny-1,:]   = avar[:,ny-1,:]   - 0.5*(ocvar[:,ny-1,:] + ocvar[:,ny,:])
    tmp[:,1:ny-1,:] = avar[:,1:ny-1,:] - \
                      (-ocvar[:,0:ny-2,:] + 13.0*ocvar[:,1:ny-1,:] \
                       -ocvar[:,3:ny+1,:] + 13.0*ocvar[:,2:ny,:] ) / 24.0
    cvar[:,1,:]      = ocvar[:,1,:]    - 0.5*(tmp[:,0,:]    + tmp[:,1,:])
    cvar[:,ny-1,:]   = ocvar[:,ny-1,:] - 0.5*(tmp[:,ny-2,:] + tmp[:,ny-1,:])
    cvar[:,2:ny-1,:] = ocvar[:,2:ny-1,:] + \
                       (-tmp[:,0:ny-3,:] + 7.0*(tmp[:,1:ny-2,:] + tmp[:,2:ny-1,:]) - tmp[:,3:ny,:] ) / 12.0

  if key == "W" or key == "PH":
    cvar = ocvar
    tmp[0,:,:]      = avar[0,:,:]      - 0.5*(ocvar[0,:,:]    + ocvar[1,:,:])
    tmp[nz-1,:,:]   = avar[nz-1,:,:]   - 0.5*(ocvar[nz-1,:,:] + ocvar[nz,:,:])
    tmp[1:nz-1,:,:] = avar[1:nz-1,:,:] - \
                      (-ocvar[0:nz-2,:,:] + 13.0*ocvar[1:nz-1,:,:] \
                       -ocvar[3:nz+1,:,:] + 13.0*ocvar[2:nz,:,:] ) / 24.0
    cvar[1,:,:]      = ocvar[1,:,:]    + 0.5*(tmp[0,:,:]    + tmp[1,:,:])
    cvar[nz-1,:,:]   = ocvar[nz-1,:,:] + 0.5*(tmp[nz-2,:,:] + tmp[nz-1,:,:])
    cvar[2:nz-1,:,:] = ocvar[2:nz-1,:,:] + \
                        (-tmp[0:nz-3,:,:] + 7.0*(tmp[1:nz-2,:,:] + tmp[2:nz-1,:,:]) - tmp[3:nz,:,:] ) / 12.0

  return cvar

#-------------------------------------------------------------------------------
# Function to write displacement vectors to new file

def write_dispvect(dvfile,a,b,scls,nx,ny,nz,spc,Si,Sg1,min_dist,min_vect,max_vect,min_dbz,dbz_thd,eb,sigsq,lams,lamd,lamm,lamb):
   
   print("\nWriting out displacement vectors to:",dvfile)

   ncfile = ncdf.Dataset(dvfile,'w') 
 
   nscl = ncfile.createDimension('nscl', nz) 
   nlat = ncfile.createDimension('nlat', ny)
   nlon = ncfile.createDimension('nlon', nx)

   sigma = ncfile.createVariable('sigma', np.float32, ('nscl',))
   sigma.long_name = 'sigma for gaussian_filter'  

   ivect = ncfile.createVariable('ivect',np.float32,('nscl','nlat','nlon'))
   ivect.units = 'grid units'
   ivect.standard_name = 'east-west displacement vector'
   jvect = ncfile.createVariable('jvect',np.float32,('nscl','nlat','nlon')) 
   jvect.units = 'grid units'
   jvect.standard_name = 'north-south displacement vector'

   sigma[:] = scls 
   ivect[:,:,:] = a[:,:,:]
   jvect[:,:,:] = b[:,:,:]

   ncfile.lams=lams
   ncfile.lamd=lamd
   ncfile.lamm=lamm
   ncfile.lamb=lamb
   ncfile.Si=Si
   ncfile.Sg1=Sg1
   ncfile.sigsq=sigsq
   ncfile.eb=eb
   ncfile.spc=spc
   ncfile.min_dist=min_dist
   ncfile.min_vect=min_vect
   ncfile.max_vect=max_vect
   ncfile.min_dbz=min_dbz
   ncfile.dbz_thd=dbz_thd

   ncfile.close()

#-------------------------------------------------------------------------------
# Function to put observations onto model grid

def regridobs(odata, fdata):
      # Form Obs Lat-Lon Grid
      nlons  = len(odata.dimensions['Lon'])
      nlats  = len(odata.dimensions['Lat'])
      olats = np.float(odata.Latitude)  - np.float(odata.LatGridSpacing) * np.arange(nlats)
      olons = np.float(odata.Longitude) + np.float(odata.LonGridSpacing) * np.arange(nlons)

      olon2d, olat2d = np.meshgrid(olons, olats)

      # Get Fcst Lat-Lon Grid
      flats = fdata.variables['XLAT'][0]
      flons = fdata.variables['XLONG'][0]

      # Get Obs Field
      ofld = np.copy(odata.variables['MergedReflectivityQCComposite'])

      # Flatten grid and obs field
      oltf = olat2d.flatten()
      olnf = olon2d.flatten()
      ofldf = ofld.flatten()

      return griddata((oltf,olnf), ofldf, (flats,flons), method='linear', fill_value=0.0)

#-------------------------------------------------------------------------------
# Function to check if the gradient jives with the cost function

def GradCheck(ab,y,x,i,j,sigsq,di,Si,lams,lamd,lamb,lamm):
    ny = len(i[:,0])
    nx = len(i[0,:])
    N = 2*nx*ny
    rchek0 = 1.0e10
    work = np.zeros(N)
    grad = np.zeros(N)
    rchek = rchek0
    fx1 = f(ab,y,x,i,j,sigsq,di,Si,lams,lamd,lamb,lamm)
    grad = gradf(ab,y,x,i,j,sigsq,di,Si,lams,lamd,lamb,lamm)
    print(("GradCheck: rchek = %f. fx1 = %f \n" %(rchek, fx1)))

    gxnn = 0.0
    for ii in range(0,N):
      gxnn += grad[ii]*grad[ii]
      #print gxnn
    print(("GradCheck: gxnn = %f \n" %(gxnn)))
   
    for jj in range(0,31):
      rchek = rchek*0.1
      for ii in range(0,N):
        work[ii] = ab[ii] + rchek*grad[ii]
        #if work[ii] != 0.0:
        #  print ab[ii], rchek, grad[ii], work[ii]
      fx2 = f(work,y,x,i,j,sigsq,di,Si,lams,lamd,lamb,lamm)
      ffff = (fx2 - fx1)/(gxnn*rchek)
      print(("GradCheck: jj = %d. fx2 = %.4f. ffff = %.12f \n" %(jj,fx2,ffff)))


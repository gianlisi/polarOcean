#!/usr/bin/env python

#''' 
#Copyright (c) 2021 Gianluca Meneghello
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
#'''

'''
Computations can be performed by

./processDataset.py all

with parallelization managed by xarray and dask. Alternatively MPI can be 
leveraged for one step of the computations by running

./processDataset.py regrid
mpirun -n 8 ./processDataset.py  compute
./processDataset.py figures 


Description
-----------

Two different datasets are regridded and combined to compute daily surface 
stress in the Southern Ocean for 2011. They are

-- wind from the reanalysis [1]
-- geostrophic currents computed from satellite altimetry [2]

All dataset are remotely either through an OPeNDAP server (winds) 
or by downloading a file (altimetry)

Parallelization is achieved using xarray and dask for the regridding part, 
and by MPI for the non-linear computation of the surface stress.  The following 
steps are undertaken:

-- datasets are converted to a common format using a preprocess function called 
   by open_mfdataset, see the functions preprocess*()
-- an interpolation method is provided by adding a new class to the xarray 
   definition, see the class regridAccessor()
-- the interpolation if performed on the four datasets leveraging xarray, 
   dask, and pyresample, see the function regrid()
-- the nonlinear computation of the surface stress is parallelized using MPI, 
   see function computeSurfaceStress()

Bibliography:
------------
[1] NCEP Reanalysis data provided by the NOAA/OAR/ESRL PSL, Boulder, Colorado, USA, from their Web site at https://psl.noaa.gov/
[2] Armitage, T. W. K., Kwok, R., Thompson, A. F., & Cunningham, G. (2018). Dynamic topography and sea level anomalies of the Southern Ocean: 
    Variability and teleconnections. Journal of Geophysical Research: Oceans, 123, 613– 630. https://doi.org/10.1002/2017JC013534
'''

import numpy as np
import xarray as xr
import xarray.ufuncs as xu
import dask
from dask.distributed import Client
from dask.diagnostics import ProgressBar
import pandas as pd
from scipy.optimize import newton_krylov
from pyresample import geometry
from pyresample.bilinear import XArrayBilinearResampler
import warnings
import urllib
import os
from os import path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

warnings.filterwarnings("ignore", category=RuntimeWarning)      # ignore warning from pyresample
warnings.filterwarnings("ignore")                               # ignore all warnings 

def coriolis(latitude):
  '''Definition of the Coriolis parameter.

  Parameters
  ----------
  latitude : scalar or array

  Returns
  -------
  The Coriolis parameter at the given latitude, same type as latitude
  '''
  return 0.000145842*xu.sin(latitude*np.pi/180)

def preprocessNCEP(ds):
  '''Pre-process the NCEP dataset to a common format during the xr.open_mfdataset call.

  Specifically:
  - introduce the x and y coordinates required by pyresample
  - shift longitude from 0:360 to -180:180, and roll the dataset accordingly
  - add the 2D fields longitude and latitude that are used to define the grid in pyresample
  '''
  ds = ds.rename(lon='x', lat='y')
  ds = ds.assign_coords(x=(((ds.x + 180) % 360) - 180) ).roll( x=len(ds.x)//2, roll_coords=True)
  lat2d,lon2d = xr.broadcast(ds.y,ds.x)
  ds = ds.assign_coords(longitude=lon2d, latitude=lat2d)
  return ds

def preprocessDOT(ds):
  '''Pre-process the Dynamic Ocean Topography dataset to a common format during the xr.open_mfdataset call.

  Specifically:
  - rename X and Y coordinates to x and y for compatibilty with pyresample
  - sort the y coordinate in descending order for compatibility with pyresample
  - set latitude and longitude fields are coordinates
  - convert coordinates from km to m
  - overwrite time coordinates with datetim64 data
  - convert fields from cm to m
  '''
  ds = ds.rename(X='x', Y='y', date='time', Latitude='latitude', Longitude='longitude').sortby('y', ascending=False).set_coords(['latitude','longitude'])
  time = pd.date_range('2011-01-01', '2016-12-31', freq='MS') + np.timedelta64(14, 'D')
  ds = ds.assign_coords(x=ds.x*1000, y=ds.y*1000, time=time)
  ds.x.attrs['units']='m'
  ds.y.attrs['units']='m'
  ds['MDT'] /= 100
  ds['DOT'] /= 100
  ds['SLA'] /= 100
  ds.MDT.attrs['units']='m'
  ds.DOT.attrs['units']='m'
  ds.SLA.attrs['units']='m'
  return ds
 
@xr.register_dataset_accessor("regrid")
class regridAccessor:
  '''Add an interpolation method based on pyresample to the xarray class.'''
  def __init__(self, xarray_obj):
    '''Initialize grid geometry'''
    self._obj  = xarray_obj
    self._grid = geometry.SwathDefinition(lons=self._obj.longitude, lats=self._obj.latitude)

  def __call__(self, targetGrid, vector=None):
    '''Perform the regridding and optional vector rotation.

    Parameters
    ----------
    targetGrid : the target grid described by a pyresample geometry.SwathDefinition, 
                 geometry.AreaDefinition or geometry.GridDefinition
    vector :     tuple. The keys of the vector components, e.g. ('u','v')

    Returns
    -------
    xarray.DataSet conatining the regridded (and rotated) fields.
    '''

    resampler = XArrayBilinearResampler(self._grid, targetGrid, 1e6) 
    dest      = xr.merge([resampler.resample(self._obj[field]).rename(field) for field in self._obj.keys()])
    lon,lat = targetGrid.get_lonlats()
    dest['longitude'] = (['y','x'], lon)
    dest['latitude']  = (['y','x'], lat)
    dest = dest.set_coords(['latitude','longitude'])

    if vector:
      warnings.warn("Vector rotation is valid only for mapping from latlon to EASE grids. There is no check implemented.",UserWarning)
      ukey, vkey = vector
      u, v = dest[ukey], dest[vkey]
      u2 =  u * xu.cos(np.pi/180*lon) + v * xu.sin(np.pi/180*lon) 
      v2 = -u * xu.sin(np.pi/180*lon) + v * xu.cos(np.pi/180*lon) 
      dest[ukey] = u2
      dest[vkey] = v2
    
    return dest

def regrid(timestart="2011-01-01", timeend="2011-12-31", filenameOut='regridded_dataset.nc'):
  '''Regrid variables to common grid targetGrid, compute geostrophic 
  currents and save data to "regridded_dataset.nc".
  '''
 
  timeChunks = 16
  timeSlice = slice(timestart, timeend)

  targetGrid = geometry.AreaDefinition('ease_sh', 'target grid', 'ease_sh', 'EPSG:3409', 321, 321, (-4023333.8, -4023333.8, 4023333.8, 4023333.8))

  with ProgressBar(minimum=1):
    
    # regrid wind velocity
    print("\nRegrid wind velocity --- the file used is accessed from NOAA's OPeNDAP server\n")
    filename = ['http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.dailyavgs/surface_gauss/%swnd.10m.gauss.2011.nc'%component for component in ['u','v'] ] 
    #filename = ['wind/%s10m-2011-2016.nc'%s for s in ['u','v']]
    with xr.open_mfdataset(filename, preprocess=preprocessNCEP, parallel=True, chunks={ 'time':timeChunks } ) as ds:
      wnd = ds[['uwnd','vwnd']].sel(time=timeSlice, y=slice(-40,-90)).load().regrid(targetGrid,vector=['uwnd','vwnd'])
    
    # regrid dynamic ocean topography
    print('\nRegrid dynamic ocean topography --- the file is downloaded from the web, unless it is already locally present.\n')
    filename = 'CS2_combined_Southern_Ocean_2011-2016.nc'
    if not path.exists(filename): 
      print("Downloading file")
      urllib.request.urlretrieve("http://svante.mit.edu/~mgl/mk/dot/CS2_combined_Southern_Ocean_2011-2016.nc", "CS2_combined_Southern_Ocean_2011-2016.nc");
    with xr.open_mfdataset(filename, preprocess=preprocessDOT, chunks={ 'date':timeChunks }) as ds:
      dot = ds[ ['DOT'] ].sel(time=timeSlice).regrid(targetGrid)

    # merge dataset and interpolate dynamic ocean topography
    print('Merging datasets')
    ds = xr.merge([wnd,dot])
    ds['DOT'] = ds.DOT.interpolate_na('time').ffill('time').bfill('time')   # correct dynamic ocean topography for missing data at the beginning and the end

    # add geostrophic currents to the dataset
    print('Adding geostrophic currents to dataset')
    g = 9.81665
    ds['ugeo'] = -ds.DOT.differentiate('y') * g/coriolis(ds.latitude)
    ds['vgeo'] =  ds.DOT.differentiate('x') * g/coriolis(ds.latitude)


    # save dataset for future reuse
    filename = 'regridded_dataset.nc'
    print('Saving regridded dataset to %s'%filename)
    with ProgressBar():
      ds.to_netcdf(filename)
    ds.close()

def iterativeTau(ds, **kwargs):
  '''Iteratively solve for the surface stress given wind velocities uwnd, vwnd and surface currents ugeo, vgeo.
  
  The equations solved are

  rel = Uwnd - (Ugeo+Uek)
  tau = 0.00125*1.25*|Urel|*Urel
  Uek = Nabla x tau / (rho f)

  where Uek is the Ekman velocity at the surface (see uek, vek definition below).

  Parameters
  ----------
  ds : the input dataset. Must contain the following fields (at least)
    - uwnd, vwdn : the 10m wind velocity 
    - ugeo, vgeo : the geostrophic surface currents
    - latitude : the grid points' latitude

  **kwargs : arguments passed to the newton_krylov solver

  Returns
  -------
  xarray.Dataset with same dimensions as ds, containing the surface stress components taux and tauy

  Notes
  -----
  Fields that do not change during the iterative process are loaded first in order to limit the 
  computational cost of the residual(tau) function.
  '''

  # define the residual function
  def residual(tau, uwnd, vwnd, ugeo, vgeo, rhofd):
    uek = ( tau[0]*np.cos(-np.pi/4) - tau[1]*np.sin(-np.pi/4) )/rhofd
    vek = ( tau[0]*np.sin(-np.pi/4) + tau[1]*np.cos(-np.pi/4) )/rhofd
    
    urel = uwnd - (ugeo+uek)
    vrel = vwnd - (ugeo+vek)
    uvrel = np.sqrt(urel**2+vrel**2)
    
    return xr.concat([
        - tau[0] + (uvrel*urel*CDa),
        - tau[1] + (uvrel*vrel*CDa)
        ],dim='dir').fillna(0)
  
  # load all fields that remain constant during the iterative computations
  CDa = 0.00125*1.25 

  rhofd = ( coriolis(ds.latitude)*1028*20 ).values

  ugeo = ds.ugeo.load()
  vgeo = ds.vgeo.load()
  uwnd = ds.uwnd.load()
  vwnd = ds.vwnd.load()
  
  # initialize initial guess
  tau = xr.zeros_like( ds[['uwnd','vwnd']].rename(uwnd='taux',vwnd='tauy').to_array('dir',name='tauxy') )
 
  #the actual computation
  tau[...] = newton_krylov(lambda tau: residual(tau,uwnd,vwnd,ugeo,vgeo,rhofd), tau, **kwargs)

  mask = ds.ugeo.isnull() | ds.vgeo.isnull() 

  return tau.to_dataset(dim='dir').where(~mask)
 
def computeSurfaceStress():
  '''Compute surface stress.'''
  
  from mpi4py import MPI

  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()

  dask.config.set(scheduler='single-threaded')
  try:
    ds = xr.open_dataset('regridded_dataset.nc')
  except OSError:
    print("regridded_dataset.nc not found. Run ./processDataset.py regrid first.")


  times = np.array_split(ds.time,size)
  
  tau = []
  for i,tr in enumerate(times[rank]):
    print("rank = %2i/%i;  timestep = %3i/%i;  date = "%(rank+1,size,i+1,len(times[rank])),tr.values,flush=True)
    tmp = ds.sel(time=tr).load()
    res = iterativeTau(tmp, verbose=False, method='lgmres' )
    tau.append(res)

  tau = xr.concat(tau,'time')
  tau.to_netcdf('regridded_tau%04i.nc'%rank)

def figures():
  '''Plots the following figures.

  -- a time series of the mean 10m wind kinetic energy 
  -- the time-averaged wind components
  -- the time-averaged geostrophic current components
  -- a monthly climatology of the 10m wind's u component 
  -- the time-averaged surface stress
  '''

  map_proj  = ccrs.Orthographic(-80, -70)
  transform = ccrs.epsg(3409)

  subplot_kws={"projection":map_proj}
  with xr.open_mfdataset('regridded_dataset.nc') as ds:
    p = (ds.uwnd**2+ds.vwnd**2).mean(['x','y']).plot()
    plt.gca().set_ylabel(r'10m wind kinetic energy [m$^2$/s$^2$]')

    p = ds[['uwnd','vwnd']].mean('time').to_array('var').plot(col='var',robust=True,cbar_kwargs={"label": r"wind speed [m/s]"},subplot_kws=subplot_kws,transform=transform)
    for ax in p.axes.flat:
      ax.coastlines()
      ax.set_global()

    p = ds[['ugeo','vgeo']].mean('time').to_array('var').plot(col='var',robust=True,cbar_kwargs={"label": r"current speed [m/s]"},subplot_kws=subplot_kws,transform=transform)
    for ax in p.axes.flat:
      ax.coastlines()
      ax.set_global()

    p = ds.uwnd.groupby('time.month').mean('time').plot(col='month',robust=True,col_wrap=4,cbar_kwargs={"label": r"u wind component [m/s]"},subplot_kws=subplot_kws,transform=transform)
    for ax in p.axes.flat:
      ax.coastlines()
      ax.set_global()


  with xr.open_mfdataset('regridded_tau*.nc',parallel=True) as ds:
    p = ds[['taux','tauy']].mean('time').to_array('var').plot(col='var',robust=True,cbar_kwargs={"label": r"surface stress [N/m$^2$]"},subplot_kws=subplot_kws,transform=transform)
    for ax in p.axes.flat:
      ax.coastlines()
      ax.set_global()

  
  print('Close all figures to exit')
  plt.show()

if __name__ == "__main__":
  import sys
  stage = sys.argv[1] if len(sys.argv)>1 else None
 
  if stage == 'regrid':
    regrid()
  elif stage == 'compute':
    computeSurfaceStress()
  elif stage == 'figures':
    figures()
  elif stage == 'all':
    regrid()
    computeSurfaceStress()
    figures()
  else:
    print(__doc__)

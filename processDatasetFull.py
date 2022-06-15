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

./arcticEkmanPumping.py all

with parallelization managed by xarray and dask. Alternatively MPI can be 
leveraged for one step of the computations by running

./arcticEkmanPumping.py regrid
mpirun -n 8 ./arcticEkmanPumping.py  computeStress
./arcticEkmanPumping.py computeEkman
./arcticEkmanPumping.py figures 

Results from this software have been publised in the following journal articles:

A.Ramadhan, J.Marshall, G.Meneghello, L.Illari, and K.Speer (2022) Observations of upwelling and downwelling around Antarctica mediated by sea ice Frontiers in Marine Science, Physical Oceanography, 
J.E. Lenetsky, B. Tremblay, C. Brunette, G. Meneghello. (2021) Sub-Seasonal predictability of Arctic Ocean sea ice conditions: Bering Strait and Ekman-driven Ocean Heat Transport. J. Clim., doi:10.1175/JCLI-D-20-0544.1 
G. Meneghello, E. W. Doddridge, J. Marshall, J. Scott, J-M Campin (2020) Exploring the role of the “Ice-Ocean governor” and mesoscale eddies in the equilibration of the Beaufort Gyre: lessons from observations. J. Phys. Oceanogr., 50(1), doi:10.1175/JPO-D-18-0223.1
G. Meneghello, J. Marshall, M.-L. Timmermans and J. Scott (2018) Observations of seasonal upwelling and downwelling in the Beaufort Sea mediated by sea ice. J. Phys. Oceanogr., 48(4), 795–805. doi:10.1175/JPO-D-17-0188.1; dataset doi:10.18739/A2J678X08
G. Meneghello, J. Marshall, S.T. Cole, and M.-L. Timmermans (2017) Observational inferences of lateral eddy diffusivity in the halocline of the Beaufort Gyre. Geophys. Res. Lett., 44. doi:10.1002/2017GL075126

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
from pyresample.kd_tree import resample_gauss
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
  ds = ds.assign_coords( longitude = lon2d , latitude = lat2d)
  return ds

def preprocessIceConcentration(ds):
  '''Preprocess the Ice Concentration dataset to a common format during the xr.open_mfdataset call.

  Specifically:
  - rename the xgrid and ygrid coordinates to x and y for compatiblity with pyresample
  - maks land and invalid entried
  '''
  ds = ds.rename(xgrid='x',ygrid='y',seaice_conc_cdr='concentration')
  mask    = (ds['concentration'][0]>=0) & (ds['concentration'][0]<=1)
  ds      = ds.where(mask)
  return ds

def preprocessIceVelocity(ds):
  '''Preprocess the Ice Velocity dataset to a common format during the xr.open_mfdataset call.

  Specifically:
  - set latitude and longitude fields as coordinates
  - rename u and v field to uice and vice
  - convert ice velocity from cm/s to m/s
  - convert time coordinate to datetime64
  '''
  ds = ds.set_coords(['latitude','longitude']).rename(v='vice',u='uice')
  ds['uice'] /= 100; ds.uice.attrs['units'] = 'm/s'
  ds['vice'] /= 100; ds.vice.attrs['units'] = 'm/s'
  ds = ds.assign_coords( time = ds.indexes['time'].to_datetimeindex() )
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
  ds = ds.rename(lon='x', lat='y', date='time').transpose('time','y','x')
  time = pd.date_range('2003-01-01', '2014-12-31', freq='MS') + np.timedelta64(14, 'D')
  lat2d,lon2d = xr.broadcast(ds.y,ds.x)
  ds = ds.assign_coords( longitude = lon2d , latitude = lat2d, time = time)
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

def regrid(timestart="2003-01-01", timeend="2014-12-31", filenameOut='regridded_dataset.nc'): # up to 2014-12-13
  '''Regrid variables to common grid targetGrid, compute geostrophic 
  currents and save data to "regridded_dataset.nc".
  '''
 
  timeChunks = 16
  timeSlice = slice(timestart, timeend)

  #targetGrid = geometry.AreaDefinition('ease_sh', 'target grid', 'ease_sh', 'EPSG:3409', 321, 321, (-4023333.8, -4023333.8, 4023333.8, 4023333.8))
  projection = '+proj=laea +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 +a=6371228 +b=6371228 +units=m +no_defs'
  targetGrid = geometry.AreaDefinition.from_area_of_interest('ease_nh',projection,(361,361),(0,0),25067.525)

  with ProgressBar(minimum=1):

    print('Regrid wind velocity')
    #filename = ['http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.dailyavgs/surface_gauss/%swnd.10m.gauss.%04i.nc'%(component,year) for year in np.arange(yearstart,yearend+1) for component in ['u','v'] ] 
    filename = ['/Users/gianlu/Documents/MIT/data/NCEPNCAR/%swind.2000-2013.daily.nc'%s for s in ['u','v']]
    with xr.open_mfdataset( filename , preprocess=preprocessNCEP , parallel=True , chunks={ 'time':timeChunks } ) as ds:
      wnd  = ds[ ['uwnd','vwnd']].sel( time=timeSlice , y=slice(90,35) ).regrid(targetGrid,vector=['uwnd','vwnd']).compute()
   
    print('Regrid ice concentration')
    filename = 'iceconcentration/seaice_conc_daily_nh_2003-2014_v03r01.nc'
    with xr.open_mfdataset( filename , preprocess=preprocessIceConcentration , parallel=True , chunks={ 'time':timeChunks } , concat_dim = 'time' ) as ds:
      icec = ds[ ['concentration'] ].sel( time=timeSlice ).regrid(targetGrid).compute()
  
    print('Regrid ice velocity')
    filename = 'ICEMOTIONV4data/icemotion_daily_nh_25km_*_v4.1.nc'
    with xr.open_mfdataset( filename , preprocess=preprocessIceVelocity , parallel=True , chunks={ 'time':timeChunks } , data_vars='minimal' ) as ds:
      icev = ds[ ['uice','vice'] ].sel( time=timeSlice ).regrid(targetGrid,vector=None).compute()  # no need to rotate velocity, they are already on the target grid
    
    print('Regrid dynamic ocean topography')
    filename = '/Users/gianlu/Documents/MIT/data/DOTdata/CPOM_DOT.nc'
    with xr.open_mfdataset( filename , preprocess=preprocessDOT ) as ds:
      dot  = ds[ ['DOT'] ].sel( time=timeSlice ).interpolate_na('time').regrid(targetGrid).compute()
  
    
    print('Merging datasets')
    ds = xr.merge([wnd,icec,dot,icev[['uice','vice']]])
    ds['DOT'] = ds.DOT.interpolate_na('time').ffill('time').bfill('time').compute()   # correct dynamic ocean topography for missing data at the beginning and the end

    print('Adding geostrophic currents to dataset')
    g = 9.81665
    ds['ugeo'] = -ds.DOT.differentiate('y') * g/coriolis(ds.latitude)
    ds['vgeo'] =  ds.DOT.differentiate('x') * g/coriolis(ds.latitude)


    # save dataset for future reuse
    filename = 'regridded_dataset.nc'
    print('Saving regridded dataset to %s'%filename)
    ds.to_netcdf(filename)
    ds.close()

def iterativeTau(ds, **kwargs):
  '''Iteratively compute the surface stress using a Newton-Krylov solver.

  Solve the nonlinear problem of computing surface stress given wind velocities uwnd, vwnd, ice velocities uice, vice, surface currents ugeo, vgeo and ice concentration.

  rel = Uice - (Ugeo+Uek)
  tau = (1-concentration)*0.00125*1.25*|Uwnd|*Uwnd + concentration*0.0055*1028*|Urel|*Urel

  where Uek is the Ekman velocity at the surface (see uek, vek definition below).

  Parameters
  ----------
  ds : the input dataset. Must contain the following fields (at least)
    - uwnd, vwdn : the 10m wind velocity 
    - uice, vice : the ice drift velocity
    - ugeo, vgeo : the geostrophic surface currents
    - concentration : the ice concentration
    - latitude : the grid points' latitude

  **kwargs : arguments passed to the newton_krylov solver

  Returns
  -------
  xarray.Dataset with same dimensions as ds containing the surface stress components taux and tauy

  Notes
  -----
  Fields that do not change during the iterative process are first computed in order to limit the 
  computational cost of the residual(tau) function.
  '''

  uvwnd = np.sqrt(ds.uwnd**2+ds.vwnd**2)
  
  CDa = ((1-ds.concentration) * 0.00125*1.25 ).compute()
  CDi =   ( ds.concentration  * 0.0055 *1028 ).compute()
  rhofd = ( coriolis(ds.latitude)*1028*20 ).values

  tauxa = (uvwnd * ds.uwnd * CDa).compute()
  tauya = (uvwnd * ds.vwnd * CDa).compute()

  uice = ds.uice.fillna(0).compute()
  vice = ds.vice.fillna(0).compute()

  ugeo = ds.ugeo.load()
  vgeo = ds.vgeo.load()

  tau = xr.zeros_like( ds[['uwnd','vwnd']].rename(uwnd='taux',vwnd='tauy').to_array('dir',name='tauxy') )

  def residual(tau):
    uek = ( tau[0]*np.cos(-np.pi/4) - tau[1]*np.sin(-np.pi/4) )/rhofd
    vek = ( tau[0]*np.sin(-np.pi/4) + tau[1]*np.cos(-np.pi/4) )/rhofd
    
    urel =  uice - (  uek + ugeo ) 
    vrel =  vice - (  vek + ugeo )
    uvrel = np.sqrt(urel**2+vrel**2)
    
    return xr.concat([
        - tau[0] + ( tauxa + uvrel * urel * CDi ),
        - tau[1] + ( tauya + uvrel * vrel * CDi )
        ],dim='dir').fillna(0)

  tau[...] = newton_krylov( residual, tau, **kwargs )

  mask = ds.ugeo.isnull() | ds.vgeo.isnull() | ds.concentration.isnull() 

  return tau.to_dataset(dim='dir').where(~mask)
 
def computeSurfaceStress():
  '''Compute surface stress.'''
  
  from mpi4py import MPI

  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()

  dask.config.set(scheduler='single-threaded')
  
  ds = xr.open_dataset('regridded_dataset.nc')

  times = np.array_split(ds.time,size)
  
  tau = []
  for i,tr in enumerate(times[rank]):
    if rank == 0:
      print("rank = %5i %4i/%i"%(rank,i+1,len(times[rank])),tr.values,flush=True)
      verbose = True
    else:
      verbose = False
    tmp = ds.sel(time=tr).load()
    tau.append( iterativeTau( tmp , verbose = verbose , method='lgmres' ) )

  tau = xr.concat(tau,'time')
  tau.to_netcdf('regridded_tau%04i.nc'%rank)

def computeEkman():
  '''Compute Ekman pumping'''
  
  tau = xr.open_mfdataset('regridded_tau*.nc',combine='by_coords',parallel=True,chunks={'time':10})
  with ProgressBar():
    taux = tau.taux #.groupby('time.month').mean('time').sel(month=slice(7,9)).mean('month')
    tauy = tau.tauy #.groupby('time.month').mean('time').sel(month=slice(7,9)).mean('month')

    f = coriolis(tau.latitude)
    wek = 1/1028 * ( (tauy/f).differentiate('x') - (taux/f).differentiate('y') ).compute()
    Uek =  (tauy/(1028*f)).compute()
    Vek = -(taux/(1028*f)).compute()

    xr.Dataset({
      'taux':taux,
      'tauy':tauy,
      'wek':wek,
      'Uek':Uek,
      'Vek':Vek
    }).to_netcdf('regridded_ekman.nc')


def figures():
  '''Plots the following figures.

  -- a time series of the mean 10m wind kinetic energy 
  -- the time-averaged wind components
  -- the time-averaged geostrophic current components
  -- a monthly climatology of the 10m wind's u component 
  -- the time-averaged surface stress
  '''
  subplot_kws=dict(facecolor="gray")
  with xr.open_mfdataset('regridded_dataset.nc') as ds:
    (ds.uwnd**2+ds.vwnd**2).mean(['x','y']).plot(); plt.gca().set_ylabel(r'10m wind kinetic energy [m$^2$/s$^2$]')
    ds[['uwnd','vwnd']].mean('time').to_array('var').plot(col='var',robust=True,cbar_kwargs={"label": r"wind speed [m/s]"},subplot_kws=subplot_kws)
    ds[['ugeo','vgeo']].mean('time').to_array('var').plot(col='var',robust=True,cbar_kwargs={"label": r"current speed [m/s]"},subplot_kws=subplot_kws)
    ds.uwnd.groupby('time.month').mean('time').plot(col='month',robust=True,col_wrap=4,cbar_kwargs={"label": r"u wind component [m/s]"},subplot_kws=subplot_kws)

  with xr.open_mfdataset('regridded_tau*.nc',parallel=True) as ds:
    ds[['taux','tauy']].mean('time').to_array('var').plot(col='var',robust=True,cbar_kwargs={"label": r"surface stress [N/m$^2$]"},subplot_kws=subplot_kws)
  
  print('Close all figures to exit')
  plt.show()

if __name__ == "__main__":
  import sys
  stage = sys.argv[1] if len(sys.argv)>1 else None
 
  if stage == 'regrid':
    regrid()
  elif stage == 'computeStress':
    computeSurfaceStress()
  elif stage == 'computeEkman':
    computeEkman()
  elif stage == 'figures':
    figures()
  elif stage == 'all':
    regrid()
    computeSurfaceStress()
    figures()
  else:
    print(__doc__)

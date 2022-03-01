import sys
from numpy import array
import numpy as np
import scipy.stats as sc
import statsmodels.tsa.stattools as stt
import xarray as xr
import pandas as pd

import SBCK as bc
import SBCK.tools as bct
import SBCK.metrics as bcm
import SBCK.datasets as bcd

import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import date
import os, psutil
from datetime import datetime

from calendar import month_abbr

import argparse

# Lookup table from method name to method description
method_lookup = {
    'QM':   'Quantile Mapping',
    'QDM':  'Quantile Delta Mapping',
    'MRec': 'Matrix Recorrelation',
    'dOTC': 'Dynamical Optimal Transport Bias Corrector',
    'CDFt': 'Quantile Mapping, taking account of an evolution of the distribution',
}

# List of valid experiments, useful for error checking, maybe make a dict like above if need
# to look up meta-data or similar for an experiment
experiments = set(['EC-Earth3-Veg', 'KIOST-ESM', 'NorESM2-MM', 'INM-CM4-8', 'MPI-ESM1-2-HR'])

def readin(var, model, month):
    var_obs = ('../../ERA5/'+var+'/ERA5_'+var+'_1989-2010_lowres.nc')
    var_sim = ('../../../../australia_climate/'+var+'/'+var+'_'+model+
                 '_SSP245_r1i1p1f1_K_1850_2100.nc')

    obs = xr.open_dataset(var_obs)
    sim = xr.open_dataset(var_sim)

    obs = obs.sel(time = slice('1989','2010'))
    sim_HIST = sim.sel(time = slice('1989','2010'))
    sim_COR = sim

    obs = obs.sel(time=obs.time.dt.month.isin([month]))
    sim_HIST = sim_HIST.sel(time=sim_HIST.time.dt.month.isin([month]))
    sim_COR = sim_COR.sel(time=sim_COR.time.dt.month.isin([month]))

    obs = obs[var]
    sim_HIST = sim_HIST[var]
    sim_COR = sim_COR[var]

    lats = obs.lat.values
    lons = obs.lon.values

    bc_HIST = np.zeros([len(sim_HIST.time.values),
                                            len(sim_HIST.lat.values),
                                            len(sim_HIST.lon.values)])
    bc_HIST[:] = np.nan

    bc_COR = np.zeros([len(sim_COR.time.values),
                                           len(sim_COR.lat.values),
                                           len(sim_COR.lon.values)])
    bc_COR[:] = np.nan

    if var == 'prec':
        sim_HIST_values = sim_HIST.values*86400
    else:
        sim_HIST_values = sim_HIST.values

    HIST_dict = {}
    HIST_dict['time'] = sim_HIST.time.values
    HIST_dict['lon'] =  sim_HIST.lon.values
    HIST_dict['lat'] =  sim_HIST.lat.values

    if var == 'prec':
        sim_COR_values = sim_COR.values*86400
    else:
        sim_COR_values = sim_COR.values

    COR_dict = {}
    COR_dict['time'] = sim_COR.time.values
    COR_dict['lon'] =  sim_COR.lon.values
    COR_dict['lat'] =  sim_COR.lat.values

    obs_values = obs.values

    return(obs_values, sim_HIST_values, sim_COR_values, bc_HIST, bc_COR,
           HIST_dict, COR_dict, lats, lons)

def write_netcdf(input, var, model, method, method_long, dic, lats, lons, period,
                 month_name):

    dataset = xr.Dataset({var:(('time', 'lat','lon'),
                               input)},
                         coords={'lat': lats,
                                 'lon': lons,
                                 'time':dic['time']})

    dataset['lat'].attrs={'units':'degrees_north',
                          'long_name':'latitude',
                          'standard_name':'latitude',
                          'axis':'Y'}
    dataset['lon'].attrs={'units':'degrees_east',
                          'long_name':'longitude',
                          'standard_name':'longitude',
                          'axis':'X'}

    if var == 'temp':
        dataset[var].attrs={'long_name':'Temperature at 2m',
                            'standard_name':'air_temperature',
                            'units':'K'}
    elif var == 'prec':
        dataset[var]=dataset[var].where(dataset[var]>0,0)
        dataset[var].attrs={'long_name':'Precipitation',
                            'standard_name':'precipitation_amount',
                            'units':'kg m-2'}

    elif var == 'insol':
        dataset[var]=dataset[var].where(dataset[var]>0,0)
        dataset[var].attrs={'long_name':'Downward solar radiation flux',
                            'standard_name':'surface_downwelling_shortwave_flux',
                            'units':'W m-2'}


    dataset.attrs={'Conventions':'CF-1.6',
                   'Model':model+' CMIP6',
                   'Experiment':'SSP245',
                   'Realisation':'r1i1p1f1',
                   'Correctionmethod': method_long,
                   'Date_Created':str(date_created)}

    dataset.to_netcdf(model+'/'+method+'_'+var+'_'+model+'_'+period+'_'+
                      month_name+'.nc',
                      encoding={'time':{'dtype': 'double'},
                                'lat':{'dtype': 'double'},
                                'lon':{'dtype': 'double'},
                                var:{'dtype': 'float32'}
                                }
                      )

def Bias_Correction(model, method, method_long, month, month_name):
    temp_obs_values, temp_sim_HIST_values, temp_sim_COR_values, temp_bc_HIST, \
    temp_bc_COR, temp_HIST_dict, temp_COR_dict, lats, lons = readin('temp',
                                                                    model,
                                                                    month)
    prec_obs_values, prec_sim_HIST_values, prec_sim_COR_values, prec_bc_HIST, \
    prec_bc_COR, prec_HIST_dict, prec_COR_dict, lats, lons = readin('prec',
                                                                    model,
                                                                    month)
    insol_obs_values, insol_sim_HIST_values, insol_sim_COR_values, insol_bc_HIST, \
    insol_bc_COR, insol_HIST_dict, insol_COR_dict, lats, lons = readin('insol',
                                                                       model,
                                                                       month)

    for i,lat in enumerate(lats):
        for j,lon in enumerate(lons):
            params_dict = {}
            if (np.isnan(temp_obs_values[0,i,j]) or
                np.isnan(temp_sim_HIST_values[0,i,j]) or
                np.isnan(prec_obs_values[0,i,j]) or
                np.isnan(prec_sim_HIST_values[0,i,j]) or
                np.isnan(insol_obs_values[0,i,j]) or
                np.isnan(insol_sim_HIST_values[0,i,j])):

                temp_bc_HIST[:,i,j] = np.nan
                prec_bc_HIST[:,i,j] = np.nan
                insol_bc_HIST[:,i,j] = np.nan

                temp_bc_COR[:,i,j] = np.nan
                prec_bc_COR[:,i,j] = np.nan
                insol_bc_COR[:,i,j] = np.nan

                params_dict['lat'] = lat
                params_dict['lon'] = lon
                params_dict['params'] = np.nan
            else:

                ### Combine variables to matrix with 3 columns and ntime rows
                OBS = array([temp_obs_values[:,i,j],prec_obs_values[:,i,j],
                             insol_obs_values[:,i,j]]).transpose()
                HIST = array([temp_sim_HIST_values[:,i,j],
                              prec_sim_HIST_values[:,i,j],
                              insol_sim_HIST_values[:,i,j]]).transpose()
                COR = array([temp_sim_COR_values[:,i,j],
                             prec_sim_COR_values[:,i,j],
                             insol_sim_COR_values[:,i,j]]).transpose()
                try:
                    if method == 'OTC_biv':
                        otc = bc.OTC()
                        otc.fit(OBS, HIST)
                        HIST_BC = otc.predict(HIST)
                        COR_BC = otc.predict(COR)

                    elif method == 'dOTC':
                        dotc = bc.dOTC()
                        dotc.fit(OBS, HIST, COR)
                        COR_BC, HIST_BC = dotc.predict(COR, HIST)

                    elif method == 'ECBC':
                    	irefs = [0]

                    	ecbc = bc.ECBC()
                    	ecbc.fit(OBS, HIST, COR)
                    	COR_BC, HIST_BC = ecbc.predict(COR, HIST)

                    elif method == 'QMrs':
                        irefs = [0]
                        qmrs = bc.QMrs(irefs = irefs)
                        qmrs.fit(OBS, HIST)
                        HIST_BC = qmrs.predict(HIST)
                        COR_BC = qmrs.predict(COR)

                    elif method == 'R2D2':
                        irefs = [0]
                        r2d2 = bc.R2D2(irefs = irefs)
                        r2d2.fit(OBS, HIST, COR)
                        COR_BC, HIST_BC = r2d2.predict(COR, HIST)

                    elif method == 'QDM':
                        qdm = bc.QDM()
                        qdm.fit(OBS, HIST, COR)
                        COR_BC, HIST_BC = qdm.predict(COR, HIST)

                    elif method == 'MBCn':
                        mbcn = bc.MBCn()
                        mbcn.fit(OBS, HIST, COR)
                        COR_BC, HIST_BC = mbcn.predict(COR, HIST)

                    elif method == 'MRec':
                        mbcn = bc.MRec()
                        mbcn.fit(OBS, HIST, COR)
                        COR_BC, HIST_BC = mbcn.predict(COR, HIST)

                    elif method == 'RBC':
                        rbc = bc.RBC()
                        rbc.fit(OBS, HIST, COR)
                        COR_BC, HIST_BC = rbc.predict(COR, HIST)

                    elif method == 'dTSMBC':
                        dtsmbc = bc.dTSMBC()
                        dtsmbc.fit(OBS, HIST, COR)
                        COR_BC, HIST_BC = dtsmbc.predict(COR, HIST)

                    elif method == 'TSMBC':
                        tsmbc = bc.TSMBC(lag=20)
                        tsmbc.fit(OBS, HIST, COR)
                        COR_BC, HIST_BC = tsmbc.predict(COR, HIST)

                except(np.linalg.LinAlgError):
                    print('fail')
                    temp_bc_HIST[:,i,j] = np.nan
                    prec_bc_HIST[:,i,j] = np.nan
                    insol_bc_HIST[:,i,j] = np.nan

                    temp_bc_COR[:,i,j] = np.nan
                    prec_bc_COR[:,i,j] = np.nan
                    insol_bc_COR[:,i,j] = np.nan

                # ### Write bias corrected values into bias correction matrix
                # ### Historical corrected
                # temp_bc_HIST[:,i,j] = HIST_BC[:,0].flatten()
                # prec_bc_HIST[:,i,j] = HIST_BC[:,1].flatten()
                # insol_bc_HIST[:,i,j] = HIST_BC[:,2].flatten()

                ### Projected corrected
                temp_bc_COR[:,i,j] = COR_BC[:,0].flatten()
                prec_bc_COR[:,i,j] = COR_BC[:,1].flatten()
                insol_bc_COR[:,i,j] = COR_BC[:,2].flatten()

                if i%5==0 and j%5==0:
                    print(lat,lon)

    ### Corrected historical temperature to netcdf
    write_netcdf(temp_bc_COR, 'temp', model, method, method_long,
                 temp_COR_dict, lats, lons, 'COR', month_name)
    write_netcdf(prec_bc_COR, 'prec', model, method, method_long,
                 prec_COR_dict, lats, lons, 'COR', month_name)
    write_netcdf(insol_bc_COR, 'insol', model, method, method_long,
                 insol_COR_dict, lats, lons, 'COR', month_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=experiments, required=True)
    parser.add_argument('--method', choices=method_lookup.keys() required=True)
    parser.add_argument('--month_num', type=int, required=True)

    args = parser.parse_args()

    # Grab months to calculate from options
    months = [args.month_num]
    month_names = [month_abbr[m] for m in months]

    startTime = datetime.now()
    date_created = date.today()

    for m, mn in zip(months, month_names):
        Bias_Correction(args.experiment
                        args.method,
                        method_lookup[args.method],
                        m,
                        mn)

    process = psutil.Process(os.getpid())
    print(process.memory_info().rss/(1024 ** 2))
    print(datetime.now() - startTime)

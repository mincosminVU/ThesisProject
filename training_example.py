#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:46:07 2024

@author: dourya
"""

import sys
import numpy as np
import xarray as xr
from emulator_functions import Predictors,Pred,Target,wrapModel 
# from emulator_functions import launch_gpu
from collections import UserDict
from datetime import datetime
import argparse
import tensorflow as tf
import netCDF4 as nc

# - Argument parsing for dynamic learning rate -
# parser = argparse.ArgumentParser(description="Train an RCM Emulator with specified learning rate")
# parser.add_argument('--learning-rate', type=float, required=True, help='Learning rate for the model training')
# args = parser.parse_args()

# launch_gpu(0)


var0_nosfc1 = [ 'rh0500', 'rh0700', 'rh0850',
                't0700', 't0850', 
                'u0700', 'u0850', 
                'v0700', 'v0850', 
                'z0700']


# The idea is to create 1 object predictor for each simulation to put in the training set

namelist_in_1 = UserDict({ 
    'target_var':'precip',
    'domain':'ST', 
    # 'domain_size':(-1,-1), 
    'filepath_in': '../Inputs/140km-nobnds-nopress-fullgreenland.KNMI-1950-2099.FGRN11.BN_RACMO2.3p2_CESM2_FGRN11.DD.nc',  
    'filepath_ref': '../Inputs/140km-nobnds-nopress-fullgreenland.KNMI-1950-2099.FGRN11.BN_RACMO2.3p2_CESM2_FGRN11.DD.nc',
    'aero_ext':False,'aero_stdz':False,'aero_var':'aero',
    'filepath_aero':'',
    'var_list' : var0_nosfc1, 
    'opt_ghg' : 'ONE', 
    'filepath_forc' : '',
    'filepath_grid' : '../Inputs/precip-detailed-noheight-ordered-sgreenland.KNMI-1950-2099.FGRN11.BN_RACMO2.3p2_CESM2_FGRN11.DD.nc',
    'filepath_model' : '../Outputs/precip_140km_8.55mm.keras',
    'filepath_target': '../Inputs/precip-detailed-noheight-ordered-sgreenland.KNMI-1950-2099.FGRN11.BN_RACMO2.3p2_CESM2_FGRN11.DD.nc',
    'filepath_gamma_param': '../Inputs/gamma_param_greenland_8.55mm.nc'
})



# namelist_in_1 = UserDict({ 
    # 'target_var':'tas',
    # 'domain':'ALP', 
    # 'domain_size':(20,16), 
    # 'filepath_in': 'path to inputs', -> one .nc file during a time period with all the variables 
    # 'filepath_ref': 'path to reference file, should be similar to input file, must be the same for any predictor set',
    # 'aero_ext':True,'aero_stdz':False,'aero_var':'aero',
    # 'filepath_aero':'path to corresponding aerosol file (must be daily)',
    # 'var_list' : var0_nosfc1, 
    # 'opt_ghg' : 'ONE', 
    # 'filepath_forc' : 'path to corresponding .csv including ghg, see example file',
    # 'filepath_grid' : 'path to a .nc with output grid information, can be extracted from target file',
    # 'filepath_model' : 'path to save the model, format can be .h5 or .keras',
    # 'filepath_target': 'path to target file (high resolution, including target domain and target variable)'
# })

for key in namelist_in_1: 
    setattr(namelist_in_1,key,namelist_in_1[key]) 

input_1 = Predictors(namelist_in_1.domain,
                     # namelist_in_1.domain_size,
                    filepath=namelist_in_1.filepath_in,
                    filepath_ref=namelist_in_1.filepath_ref,
                    stand=1,
                    var_list=namelist_in_1.var_list,
                    filepath_aero=namelist_in_1.filepath_aero,
                    filepath_forc=namelist_in_1.filepath_forc,
                    opt_ghg=namelist_in_1.opt_ghg,
                    aero_ext=namelist_in_1.aero_ext,
                     aero_var=namelist_in_1.aero_var,
                     aero_stdz=namelist_in_1.aero_stdz)


# namelist_in_2 = UserDict({ 
# 'same as namelist_in_1 with a different input file ex: other simulation. All file must be updated
# })
# for key in namelist_in_2: 
#     setattr(namelist_in_2,key,namelist_in_2[key]) 

# input_2 = Predictors(namelist_in_2.domain,
#                      namelist_in_2.domain_size,
#                     filepath=namelist_in_2.filepath_in,
#                     filepath_ref=namelist_in_2.filepath_ref,
#                     var_list=namelist_in_2.var_list,
#                     filepath_aero=namelist_in_2.filepath_aero,
#                     filepath_forc=namelist_in_2.filepath_forc,
#                     opt_ghg=namelist_in_2.opt_ghg,
#                     aero_ext=namelist_in_2.aero_ext,
#                       aero_var=namelist_in_2.aero_var,
#                       aero_stdz=namelist_in_2.aero_stdz)

listin = []

listin.append(input_1)
# listin.append(input_2)


targets = []
targets.append(Target(namelist_in_1.target_var, filepath=namelist_in_1.filepath_target,filepath_grid=namelist_in_1.filepath_grid).target.values)  
# targets.append(Target(namelist_in_2.target_var, filepath=namelist_in_2.filepath_target,filepath_grid=namelist_in_2.filepath_grid).target.values-273.16)  

amodel = wrapModel(inputIn=listin,
                  targetIn=targets,
                  target_var=namelist_in_1.target_var,
                  filepath_model = namelist_in_1.filepath_model,
                  filepath_grid=namelist_in_1.filepath_grid,
                  filepath_gamma_param=namelist_in_1.filepath_gamma_param,
                  LR=0.001)

now = datetime.now()
print(f"End of training: {now}")

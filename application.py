#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:28:58 2024

@author: dourya
"""

# import sys
import numpy as np
import xarray as xr
from emulator_functions import Predictors,Pred,Target,wrapModel  
#from emulator_functions import launch_gpu
from collections import UserDict
from datetime import date

# ----------------- Namelist ------------------------
# Please modify the following namelist with the values corresponding to 
# your emulator.  

#launch_gpu(0)


var0_nosfc1 = [ 'rh0500', 'rh0700', 'rh0850',
                't0700', 't0850',
                'u0700', 'u0850',
                'v0700', 'v0850',
                'z0700']

## The idea is to create 1 object predictor for each simulation to put in the training set
filename='result-0.76mm-preciptarget-upvars+input-remapped-140km-nodenorm.nc'
namelist_out = UserDict({ 
    'target_var':'precip',
    'domain':'ST', 
    'filepath_in': '../Inputs/140km-nobnds-nopress-fullgreenland.KNMI-1950-2099.FGRN11.BN_RACMO2.3p2_CESM2_FGRN11.DD.nc', 
    'filepath_ref': '../Inputs/140km-nobnds-nopress-fullgreenland.KNMI-1950-2099.FGRN11.BN_RACMO2.3p2_CESM2_FGRN11.DD.nc',
    'aero_ext':False,'aero_stdz':False,'aero_var':'aero',
    'filepath_aero':'',
    'var_list' : var0_nosfc1, 
    'opt_ghg' : 'ONE', 
    'filepath_forc' : '',
    'filepath_grid' : '../Inputs/precip-detailed-noheight-ordered-sgreenland.KNMI-1950-2099.FGRN11.BN_RACMO2.3p2_CESM2_FGRN11.DD.nc',
    'filepath_model' : '../Outputs/precip_140km_0.76mm.keras',
    'filepath_target': '../Inputs/precip-detailed-noheight-ordered-sgreenland.KNMI-1950-2099.FGRN11.BN_RACMO2.3p2_CESM2_FGRN11.DD.nc',
    'filepath_out': '../Outputs/'+filename,
    'filepath_gamma_param': '../Inputs/gamma_param_greenland_8.55mm.nc'
})


# namelist_out = UserDict({ 
#     'target_var':'tas',
#     'domain':'FRA', 
#     'domain_size':(20,16), 
#     'filepath_in':'path to file containing inputs to downscale', 
#     'filepath_ref':'same ref file as the one used for training', 
#     'var_list' : var0_nosfc1, 
#     'opt_ghg' : 'ONE', 
#     'filepath_forc' : 'path to correspong external forcings',
#     'filepath_grid' : 'path to output domain grid',
#     'filepath_model' : 'path to emulator trained'   ,
#     'filepath_aero':'path to corresponding aerosols file',
#     'aero_ext':True,'aero_stdz':True,'aero_var':'od550aer',
#     })

# example of CMIP lookalike attributes
attributes={
#         "Conventions" : 'CF-1.10',
#         "activity_id": 'emulation',
#         "contact": 'contact.aladin-cordex@meteo.fr',
#         "domain_id": 'ALPX-12',
#         "domain": 'EUR-11 CORDEX domain cropped to a domain centered on Alps.',
#         "driving_experiment":'Historical run with GCM forcing',
#         "driving_experiment_id":'historical',
#         "driving_institution_id":'CNRM-CERFACS',
#         "driving_source_id":'CNRM-CM6',
#         "driving_variant_label":'r15i1p1f',
          "emulator":  'CNRM_UNET11, introduced in Doury et al, 2022, is based fully'\
                         'convolutional neural network shaped from the UNeT base (Ronnenberg et al, 2015).'\
                             'The network is  minimizing the mean squared error (mse) loss function.',
          "emulator_id":'CNRM-UNET11',
          "frequency": 'day',
#         "further_info_url":'',
#         "institution":'Centre National de Recherches Meteorologiques,CNRM, Toulouse, France',
#         "institution_id":"CNRM",
#         'mip_era':"CMIP6",
          "native_resolution" : "0.14°",
          "product":"emulator_output",
#         "project_id":"I4C",
#         "realm":"",
#         "source": "CNRM-UNET11 is trained here for the near surface temperature of CNRM-ALADIN63 RCM ",
#         "source_id":'ALADIN63-emul-CNRM-UNET11',
#         "source_type":'RCM_emulator',
#         "version_realization":'v1-r1',
#         "target_institution_id":'CNRM',
#         "target_source_id":'CNRM-ALADIN63',
#         "target_version_realization":'v1',
#         "tracking_id":'',
          "training":'Trained using predictors from the RACMO2 RCM for the period'\
                     'The emulator is trained in perfect model framework, implying'\
                     'that predictors and predictands come from the same RCM simulation.'\
                     'The predictors were downscaled to 140 km resolution using nearest neighbour interpolation.'\
                     'The predictors include precipitative flux, relative humidity, temperature, '\
                     'geopotential height and eastern and northern wind components'\
                     'at 2 pressure levels (700, 850 hpa) plus at 500 hpa for humidity.',
	  "training_config": 'filters=64, LR=0.001, relu activation, max channels=512, without denormalising pred'\
                             'validation loss=1.49, standardized pred input'\
                             'nb_inputs=2' '150years' '8.55mm threshold',
          "variable_id":'precip',
#         "version_realization_info":'this is the 1st realization of the emulator CNRM-ALADIN63-emul-CNRM-UNET11-TP1 over South Greenland.',
#         "license":'',
#          "reference":"""Doury, A., Somot, S., Gadat, S. et al. Regional climate model emulator based on deep learning: 
#             concept and first evaluation of a novel hybrid downscaling approach. Clim Dyn 60, 1751–1779 (2023).
#             https://doi.org/10.1007/s00382-022-06343-9""",
          "creation_date":date.today().strftime("%d/%m/%Y")
}


for key in namelist_out: 
    setattr(namelist_out,key,namelist_out[key]) 
    

listin_pred = []

aninput_pred = Predictors(namelist_out.domain,
                filepath=namelist_out.filepath_in,
                filepath_ref=namelist_out.filepath_ref,
                stand=3,
                var_list=namelist_out.var_list,
                filepath_forc=namelist_out.filepath_forc,
                filepath_aero=namelist_out.filepath_aero,
                opt_ghg=namelist_out.opt_ghg,
                aero_ext=namelist_out.aero_ext,aero_var=namelist_out.aero_var,aero_stdz=namelist_out.aero_stdz)

listin_pred.append(aninput_pred)

targets = []
targets.append(Target(namelist_out.target_var, filepath=namelist_out.filepath_target,filepath_grid=namelist_out.filepath_grid).target.values)  

apred = Pred(namelist_out.domain,
              inputIn = listin_pred,
              targetIn=targets,
              filepath_grid =namelist_out.filepath_grid,
              filepath_out=namelist_out.filepath_out,
              filepath_model=namelist_out.filepath_model,
              target_var=namelist_out.target_var,
              filepath_gamma_param=namelist_out.filepath_gamma_param,
              attributes=attributes)

print(f"apred is {apred.ds}")     

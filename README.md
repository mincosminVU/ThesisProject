# RCM-Emulator


This repository contains the main functions used to build the RCM-emulator used in:

- Doury et al. 2022, 2024
- Vlachopoulos 2024
- Balmos 2025

It is organised as follow : 

emulator_functions.py is including all function necessary to build,train and apply the emulator

training_example.py illustrates how to train the emulator. The design of the network is hidden and automatically done by the unet_maker function inside the wrapModel class

application.py illustrates how to use a trained emulator to predict.


Doury, A., Somot, S., Gadat, S. et al. Regional climate model emulator based on deep learning: concept and first evaluation of a novel hybrid downscaling approach. Clim Dyn (2022). https://doi.org/10.1007/s00382-022-06343-9

Vlachopoulos M., Analysis of how different spatial resolutions affect the performance of an RCM-emulator, Bachelor’s Thesis for Computer Science Bachelor at VU Amsterdam, 2024.


This is a snapshot of the state of the project was in when the bachelor project was finished in July 2025.

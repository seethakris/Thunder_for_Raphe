# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 13:35:15 2015
@author: seetha

For Light Data Analysis
Take user input in this file and call other routines
"""

## Enter Main Folder containing stimulus folders to create text files

Exp_Folder ='/Users/seetha/Desktop/Ruey_Data/Raphe/Intensity_Gradient/Tiff/Registered/Sorted/'
filename_save_prefix = 'Raphe_T600'

#Rewrite text files. 1- Yes
rewrite_flag = 1

#Experiment parameters
img_size_x = 128 #X and Y resolution - if there are images that dont have this resolution, they will be resized
img_size_y = 256
img_size_crop_y1 = 20 #How many pixels to crop on x and y axis. If none say 0
img_size_crop_y2 = 20
img_size_crop_x1 = 20 #How many pixels to crop on x and y axis. If none say 0
img_size_crop_x2 = 20

# Time period within which to do the analysis
time_start = 0
time_end = 600

#Stimulus on and off time and define onset and offset times of the light stimulus
stimulus_pulse = 2 ##Whether it is a long, medium or short light stimulus

if stimulus_pulse == 1:
    stimulus_on_time = [46,86,127,168]
    stimulus_off_time = [65,105,146,187]
    
if stimulus_pulse == 2:
    stimulus_on_time = [60,240,420]
    stimulus_off_time = [180,360,540]
    

## Median filter - threshold
median_filter_threshold = 3
######################################################################


######################################################################
########################## Run Scripts ###############################

# Go into the main function that takes thunder data and 
from main_file_for_textfiles_for_thunder import initial_function

initial_function(Exp_Folder, filename_save_prefix, img_size_x, img_size_y, img_size_crop_x1, img_size_crop_x2, img_size_crop_y1, img_size_crop_y2, \
 time_start,time_end, median_filter_threshold, rewrite_flag, stimulus_on_time, stimulus_off_time)

import pickle

with open(Exp_Folder+filename_save_prefix+'_save_input_variables', 'w') as f:
    pickle.dump([img_size_x,img_size_y,img_size_crop_x1, img_size_crop_x2, img_size_crop_y1, img_size_crop_y2,\
    time_start,time_end,stimulus_pulse, stimulus_on_time, stimulus_off_time], f)



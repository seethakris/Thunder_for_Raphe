
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 17:16:26 2015
@author: seetha

"""

######################################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~ Importing Libraries ~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#Import some python libraries
import os
filesep = os.path.sep
import time
import numpy as np
#Import user defined libraries
from create_textfiles import create_textfile_individual, create_textfile_eachexp, create_textfile_allexps
######################################################################


def  initial_function(Exp_Folder, filename_save_prefix, img_size_x, img_size_y, \
img_size_crop_x1, img_size_crop_x2, img_size_crop_y1, img_size_crop_y2,  time_start,time_end, \
median_filter_threshold, rewrite_flag,stimulus_on_time, stimulus_off_time):
    
    
    ##Create different types of text files and run analysis on them
    ## Get images from each different sections of the folder and create text files. 
    
    
    ##########1. Within each trial of an exp #####################################
    ########## Get Folder information #############################################
    
    Stimulus_Directories = [f for f in os.listdir(Exp_Folder) if os.path.isdir(os.path.join(Exp_Folder, f)) and f.find('Figures')<0]
    
    for ii in xrange(0, np.size(Stimulus_Directories, axis = 0)):
        Trial_Directories = [f for f in os.listdir(os.path.join(Exp_Folder, Stimulus_Directories[ii]))\
        if os.path.isdir(os.path.join(Exp_Folder, Stimulus_Directories[ii], f)) and f.find('Figures')<0] #Get only directories
        
        for jj in xrange(0, np.size(Trial_Directories, axis = 0)):
            Working_Directory = os.path.join(Exp_Folder, Stimulus_Directories[ii], Trial_Directories[jj])+filesep        
           ## Check if textfile exists, else create a new one
            name_for_saving_files = Stimulus_Directories[ii] + '_' + Trial_Directories[jj] + filename_save_prefix+'_individualtrial'
            txt_file = [f for f in os.listdir(Working_Directory) if (f.find(name_for_saving_files+'.txt')==0)]            
           
            if len(txt_file)==0 or rewrite_flag==1:
                start_time = time.time()
                text_file = open(Working_Directory + "log.txt", "w")
                text_file.write("'Saving images to text for all files in %s \n" % Working_Directory)
                print 'Saving images to text for all files in '+ Working_Directory            
                name_for_saving_figures = Stimulus_Directories[ii] + ' ' + Trial_Directories[jj]        
                create_textfile_individual(Working_Directory, name_for_saving_files,  name_for_saving_figures, \
                img_size_x, img_size_y, img_size_crop_x1, img_size_crop_x2, img_size_crop_y1, img_size_crop_y2, \
                time_start,time_end, median_filter_threshold, text_file,stimulus_on_time, stimulus_off_time)
                print 'Saving to text file took '+ str(int(time.time()-start_time)) +' seconds'
                text_file.write("'Saving to text file took %s seconds \n" %  str(int(time.time()-start_time)))
                text_file.close()
                
                                
    #2. Within each exp
                                
    for ii in xrange(0, np.size(Stimulus_Directories, axis = 0)):
        Trial_Directories = [f for f in os.listdir(os.path.join(Exp_Folder, Stimulus_Directories[ii]))\
        if os.path.isdir(os.path.join(Exp_Folder, Stimulus_Directories[ii], f)) and f.find('Figures')<0] #Get only directories
        
        if np.size(Trial_Directories, axis = 0)>1:

            Working_Directory = os.path.join(Exp_Folder, Stimulus_Directories[ii])+filesep        
            
            ## Check if textfile exists, else create a new one
            name_for_saving_files = Stimulus_Directories[ii] + '_'+ filename_save_prefix+'_eachexp'
            txt_file = [f for f in os.listdir(Working_Directory) if (f.find(name_for_saving_files+'.txt')==0)]            
           
            if len(txt_file)==0 or rewrite_flag==1:
                start_time = time.time()
                text_file = open(Working_Directory + "log.txt", "w")
                text_file.write("'Saving images to text for all files in %s \n" % Working_Directory)
                print 'Saving images to text for all files in '+ Working_Directory
                name_for_saving_figures = Stimulus_Directories[ii]       
                create_textfile_eachexp(Working_Directory, name_for_saving_files, name_for_saving_figures, \
                img_size_x, img_size_y, img_size_crop_x1, img_size_crop_x2, img_size_crop_y1, img_size_crop_y2,  \
                time_start,time_end,  median_filter_threshold, text_file,stimulus_on_time, stimulus_off_time)
                print 'Saving to text file took '+ str(int(time.time()-start_time)) +' seconds'
                text_file.write("'Saving to text file took %s seconds \n" %  str(int(time.time()-start_time)))
                text_file.close()
    
    #3. For all exps and all trials
    if np.size(Stimulus_Directories, axis = 0)>1:
        Working_Directory = Exp_Folder
        name_for_saving_files = 'All_exps_'+ filename_save_prefix+'_eachexp'
        txt_file = [f for f in os.listdir(Working_Directory) if (f.find(name_for_saving_files+'.txt')==0)]            

        if len(txt_file)==0 or rewrite_flag==1:
                start_time = time.time()
                text_file = open(Working_Directory + "log.txt", "w")
                text_file.write("'Saving images to text for all files in %s \n" % Working_Directory)
                print 'Saving images to text for all files in '+ Working_Directory
                create_textfile_allexps(Working_Directory, name_for_saving_files, \
                img_size_x, img_size_y, img_size_crop_x1, img_size_crop_x2, img_size_crop_y1, img_size_crop_y2,\
                time_start,time_end, median_filter_threshold, text_file,stimulus_on_time, stimulus_off_time)
                print 'Saving to text file took '+ str(int(time.time()-start_time)) +' seconds'
                text_file.write("'Saving to text file took %s seconds \n" %  str(int(time.time()-start_time)))
                text_file.close()
        
    
    
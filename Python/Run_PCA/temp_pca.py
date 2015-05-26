Exp_Folder ='/Users/seetha/Desktop/Ruey_Habenula/Habenula/Intensity_ramp/Fish963/Tiff/Cropped/Registered/Sorted/Fish963/'
filename_save_prefix = 'Hb_T600'


from thunder import ThunderContext

print 'Starting Thunder Now. Check console for details'
tsc = ThunderContext.start(appName="thunderpca")
import os
filesep = os.path.sep

import matplotlib.pyplot as plt 

import numpy as np
from thunder_pca import run_pca
from thunder_pca import make_pca_maps
from thunder_pca_plots import plot_pca_maps
from pca_thunder_analysis import plot_preprocess_data

from thunder import Colorize
image = Colorize.image

Stimulus_Directories = [f for f in os.listdir(Exp_Folder) if os.path.isdir(os.path.join(Exp_Folder, f)) and f.find('Figures')<0]
#Stimulus_Directories
ii = 0
Trial_Directories = [f for f in os.listdir(os.path.join(Exp_Folder, Stimulus_Directories[ii]))\
if os.path.isdir(os.path.join(Exp_Folder, Stimulus_Directories[ii], f)) and f.find('Figures')<0]
Trial_Directories
jj = 0


flag = 0

name_for_saving_figures = Stimulus_Directories[ii] + ' ' + Trial_Directories[jj]        
Working_Directory = os.path.join(Exp_Folder, Stimulus_Directories[ii], Trial_Directories[jj])+filesep       
name_for_saving_files = Stimulus_Directories[ii] + '_' + Trial_Directories[jj] + filename_save_prefix+'_individualtrial'
#Working_Directory = os.path.join(Exp_Folder, Stimulus_Directories[ii])+filesep     
#name_for_saving_files = Stimulus_Directories[ii] + '_'+ filename_save_prefix+'_eachodor'
#name_for_saving_figures = Stimulus_Directories[ii]       

#Working_Directory = Exp_Folder
#name_for_saving_files = 'All_odors_'+ filename_save_prefix+'_eachodor'
#name_for_saving_figures = Working_Directory

data_filtered = tsc.loadSeries(Working_Directory+name_for_saving_files+'_filtered.txt', inputFormat='text', nkeys=3).toTimeSeries().detrend(method='nonlin', order=5)
data_background = tsc.loadSeries(Working_Directory+name_for_saving_files+'.txt', inputFormat='text', nkeys=3)
data_background.cache()
#plot_preprocess_data(Working_Directory, name_for_saving_files, data_filtered, stim_start, stim_end)
                
data_filtered.center()
data_filtered.zscore(30)
data_filtered.cache()

required_pcs = [1,2]
pca, imgs_pca, new_imgs = run_pca(data_filtered,4,required_pcs)
plt.plot(pca.comps.T)
##
colors_PCA = ['aqua','Fuchsia','Orange','LimeGreen']

img_size_x = np.size(imgs_pca,1)
img_size_y = np.size(imgs_pca,2)
#
maps, pts, pts_nonblack, clrs, clrs_nonblack, recon, unique_clrs, matched_pixels, matched_signals, mean_signal, sem_signal = make_pca_maps(data_background, pca, new_imgs, required_pcs, img_size_x,\
img_size_y, 100, 1000, 0.00001, 'angle', colors_PCA )
#
#plot_pca_maps(Working_Directory, name_for_saving_figures, name_for_saving_files, \
#pca.comps.T, maps, pts,pts_nonblack, clrs, clrs_nonblack, recon, unique_clrs, matched_pixels, matched_signals, stim_start, stim_end, flag,1,required_pcs)
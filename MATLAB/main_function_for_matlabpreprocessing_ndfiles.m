function main_function_for_matlabpreprocessing_ndfiles


%% Few preprocessing techniques in Matlab can be run using this script - for directories with nd files
% 1. Convert nd2 files to tiff using BioFormats reader for Matlab
% 2. Register images
% 3. Sort into stimulus folders for thunderization


Main_Directory_Name = '/Users/seetha/Desktop/Ruey_Data/Raphe/tph2_Raphe_Imaging/';

% 1. Convert nd2 files to tiff using BioFormats reader for Matlab
disp(['Converting nd2 files to tiff....for ', Main_Directory_Name])
save_nd2files_as_tiff(Main_Directory_Name)

%2. Crop images if user requires
disp(['Cropping images....for ', Main_Directory_Name,'..Require User Input'])
Tiff_Folder = [Main_Directory_Name,'Tiff/'];
% crop_pixel_outside_ndfiles(Tiff_Folder)

% 3. Register images
disp(['Registering images....for ', Main_Directory_Name])
Cropped_Folder  = [Main_Directory_Name, 'Tiff/'];
% image_register_ndfiles(Cropped_Folder)

% 4. Sort files into Fish NUmbers and regions and by timepoint for easy
% thunderization
disp(['Moving stimulus....for ', Main_Directory_Name])
Registered_Folder  = [Main_Directory_Name, 'Tiff/Registered/'];
% transfer_to_individual_stim_folders_ndfiles(Registered_Folder) %If Masking Stimulus, flag = 1 

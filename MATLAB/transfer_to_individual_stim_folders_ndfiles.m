function transfer_to_individual_stim_folders_ndfiles(Data_Folder)

%% Sort tiff files by name and transfer to folders based on type of stimulus for easy thunderization

%Create a Registered folder to save all registered images
Result_Folder = [Data_Folder, 'Sorted'];

if ~isdir(Result_Folder)
    mkdir(Result_Folder)
end

%Find files in the folder and remove those that start with . or are folders
files_present = dir([Data_Folder,filesep, '*.tif']);


%Now register all images using base. Save as multitiff
for ff = 1:length(files_present)
    
    %Find parameters of file from file name
    
    File_string = files_present(ff).name;
    
    find_fishnum = strfind(File_string,'Fish');
    find_underscore = strfind(File_string(find_fishnum+5:end),'_');
    Fish_Number = File_string(find_fishnum:find_fishnum+5+find_underscore(1)-2);
    
    find_block =   strfind(File_string, 'Block');
    find_underscore = strfind(File_string(find_block+2:end),'_');
    Block = File_string(find_block:find_block+find_underscore(1));
    
    Fish_Region_Folder = [Result_Folder, filesep, Fish_Number, filesep, Block, filesep];
    
    if ~isdir(Fish_Region_Folder)
        mkdir(Fish_Region_Folder)
    end
    
    disp(['Moving Stimulus...', files_present(ff).name,' To ', Fish_Region_Folder]);
    
    %Find number of time points in image to classify the image as time
    %points
    info = imfinfo([Data_Folder, filesep, files_present(ff).name]);
    num_t = numel(info);
    for tt = 1:num_t
        image = imread([Data_Folder, filesep, files_present(ff).name], tt);
        if tt==1
            imwrite(image,[Fish_Region_Folder, 'T=', int2str(tt), '.tif'],'tif');
        else
            imwrite(image,[Fish_Region_Folder, 'T=', int2str(tt), '.tif'],'tif', 'WriteMode', 'append');
        end
    end
end
end
function save_nd2files_as_tiff(Directory_Name)

%%Get ND2 files using bioformats reader. Read event data and save as text
%%file and save images as tiff. Seperately for each stack, if present

% Add Bioformats matlab folder to the path
addpath(genpath('/Users/seetha/Desktop/Ruey_Data/Habenula/Scripts/MATLAB/bfmatlab/'))

Result_Folder = [Directory_Name, 'Tiff'];

if ~isdir(Result_Folder)
    mkdir(Result_Folder)
end

%Find all nd2 files in the directory.
ND2_files = dir([Directory_Name, '*.nd2']);


for ff = 1:length(ND2_files)
    
    %Get data
    Data = bfopen([Directory_Name, ND2_files(ff).name]);
    
    %Get metadata from data
    omeMeta = Data{1,4};
    stackSizeX = omeMeta.getPixelsSizeX(0).getValue(); % image width, pixels
    stackSizeY = omeMeta.getPixelsSizeY(0).getValue(); % image height, pixels
    stackSizeZ = omeMeta.getPixelsSizeZ(0).getValue(); % number of Z slices
    stackSizeT = omeMeta.getPixelsSizeT(0).getValue(); % number of T slices
    stackSizeC = omeMeta.getPixelsSizeC(0).getValue(); % number of colorplanes
    
    
    %Access data
    for tt = 1:length(Data{1})
        
        %Get Z and T from the key for each data
        temp1 = strfind(Data{1}{tt,2}, 'plane');
        temp2 = strfind(Data{1}{tt,2}(temp1+2:end), '/');
        StringZ = Data{1}{tt,2}(temp1+6:temp1+temp2);
        
        temp1 = strfind(Data{1}{tt,2}, 'T=');
        temp2 = strfind(Data{1}{tt,2}(temp1+2:end), '/');
        StringT = Data{1}{tt,2}(temp1+2:temp1+temp2);
        
        
        
        if isempty(StringZ)
            StringZ = '1';
        end
        
        
        disp(['Saving to tiff ...', ND2_files(ff).name(1:end-4), ' Z=', StringZ, ' T=', StringT])
        
        image1 = Data{1}{tt,1}; %Get image
        
        %Save Data as multitiff according to Z and add description in last
        %frame for stim on and off time
        
        if strcmp('1',StringT)
            imwrite(image1,[Result_Folder, filesep,ND2_files(ff).name(1:end-4),  '.tif'],'tif');
        else
            imwrite(image1,[Result_Folder, filesep,ND2_files(ff).name(1:end-4),  '.tif'],'tif','WriteMode','append');
        end
    end
end





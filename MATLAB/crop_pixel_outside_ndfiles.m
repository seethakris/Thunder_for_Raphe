function crop_pixel_outside_ndfiles(Directory_Name)
%% Display and mark the olfactory bulb and crop anything outside it
addpath(genpath('/Users/seetha/Desktop/Ruey_Data/Habenula/Scripts/MATLAB/export_fig/'))

tiff_files = dir([Directory_Name, '*.tif']);

Result_Folder = [Directory_Name, 'Cropped/'];
if ~isdir(Result_Folder)
    mkdir(Result_Folder)
end

if ~isdir([Result_Folder,'Figures/'])
    mkdir([Result_Folder,'Figures/'])
end
%% Ask user whether to crop all individualy or together

% Find files with the directory name and plot their average for user to
% crop
for ii = 1:length(tiff_files)
    
    File_string = tiff_files(ii).name;
    
    info = imfinfo([Directory_Name, filesep, File_string]); %Get image info
    num_t = numel(info);
    
    %get all images and get their average
    A1 = load_tiff_images([Directory_Name, filesep, File_string]);
    mean_image_uint16 = mean(A1,3);
    
    
    %% Get mask of Region from image and delete the rest - using imrect
    [Region_BW, position_of_Region] = getRegionmask(mean_image_uint16, File_string(1:end-4));
    
    %% Get background pixels for subtracting
    %Create BW image using background pixels for an even background
    waitfor(msgbox('Define background Now'));
    mean_background_pixels = mask_as_backgound(mean_image_uint16, File_string(1:end-4));
    
    %% Ask user if they need to delete anything else using imfreehand
    choice = questdlg('Would you like to delete other ROIs?', 'Any more deletions?', 'Yes','No','No');
    
    if ~isempty(strfind(choice, 'Yes'))
        
        %% draw multiple freehand until the user stops drawing and create mask
        masked_average = immultiply(mean_image_uint16,Region_BW);
        
        %Draw masked image and give it the new figure handle
        others_BW = draw_multiple_freehand(masked_average, File_string(1:end-4));       
        
        save_cropped_images_individually(Result_Folder, File_string, others_BW, mean_background_pixels, A1, position_of_Region,num_t)
        
    else
        save_cropped_images_individually(Result_Folder, File_string, ~Region_BW, mean_background_pixels, A1, position_of_Region, num_t)
                
    end
end

end




function FinalImage = load_tiff_images(FileTif)

InfoImage=imfinfo(FileTif);
mImage=InfoImage(1).Width;
nImage=InfoImage(1).Height;
NumberImages=length(InfoImage);
FinalImage=zeros(nImage,mImage,NumberImages,'uint16');

TifLink = Tiff(FileTif, 'r');
for i=1:NumberImages
    TifLink.setDirectory(i);
    FinalImage(:,:,i)=TifLink.read();
end
TifLink.close();

end

%% Get Olfactory bulb mask and remove everything else
function [BW, position_Region] = getRegionmask(mean_image, File_string)

fs = figure(1);
set(fs, 'color', 'white')
h_im = imshow(mean_image, [0, 1000]);
title(File_string, 'Interpreter', 'none')

h = imrect(gca);
position_Region = wait(h); %Pause for double click
BW = createMask(h,h_im);
close(fs)
end


%% Get background pixel averages to cover the other region masks and mask it
function background_pixels = mask_as_backgound(mean_image, File_string)

fs = figure(1);
set(fs, 'color', 'white')
h_im = imshow(mean_image, [0, 1000]);
title(File_string, 'Interpreter', 'none')

h = imrect(gca);
position = wait(h);

p = getPosition(h);
p = round(p);

temp_mean = mean_image(p(2):p(2)+p(4), p(1):p(1)+p(3));
background_pixels = mean(temp_mean(:));

close(fs)

end

%% Get masks of other regions near Region which couldnt fit into the Region mask
function totMask = draw_multiple_freehand(current_mask, File_string)

fs = figure(1);
set(fs, 'color', 'white')
h_im = imshow(current_mask, [0, 1000]);
title(File_string, 'Interpreter', 'none')

totMask = false( size(current_mask) ); % accumulate all single Regionject masks to this one
h = imfreehand( gca ); setColor(h,'red');
position = wait(h);
BW = createMask(h, h_im);
while sum(BW(:)) > 10 % less than 10 pixels is considered empty mask
    totMask = totMask | BW; % add mask to glRegional mask
    % ask user for another mask
    h = imfreehand(gca); setColor(h,'red');
    position = wait(h);
    BW = createMask(h,h_im);
end
close(fs)
end


%% Use the mask to crop other stimuli and save, and for running individually
function save_cropped_images_individually(Result_Folder, File_string, BW,mean_background_pixels, Image, position_Region,  num_t)

final_images = immultiply(Image,repmat(~BW, [1,1,num_t]));   %Multiply image with mask and save

if mean_background_pixels ~= 0
    ind = find(final_images<=mean_background_pixels);
    final_images_adjust_cutroi = final_images;
    final_images_adjust_cutroi(ind) = 0;
else
    final_images_adjust_cutroi = final_images;
end

mean_final_cropped = imcrop(mean(final_images_adjust_cutroi,3),  position_Region);
plot_mean_for_verification(Result_Folder, mean_final_cropped, File_string)

% Save imaes. If t =1, create tiff file, else append
for tt = 1:num_t
    final_images_crop = imcrop(final_images_adjust_cutroi(:,:,tt), position_Region);
    if tt == 1
        imwrite(final_images_crop,[Result_Folder, filesep,'Cropped_', File_string],'tif');
    else
        imwrite(final_images_crop,[Result_Folder, filesep,'Cropped_', File_string],'tif', 'WriteMode','append');
    end
end

end

function plot_mean_for_verification(Result_Folder, mean_cropped_image, File_string)
fs2 = figure(2);
set(fs2, 'color', 'white')
imshow(mean_cropped_image, [0 1000])
title(File_string, 'Interpreter', 'None')
export_fig(fs2, [Result_Folder, 'Figures/mean_cropped_images.pdf'], '-pdf','-append', '-nocrop')
end
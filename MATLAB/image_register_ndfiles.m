function image_register_ndfiles(Data_Folder)

%% Register images in X-Y

%Create a Registered folder to save all registered images
Result_Folder = [Data_Folder, filesep, 'Registered'];

if ~isdir(Result_Folder)
    mkdir(Result_Folder)
end


%Find files in the folder and remove those that start with . or are folders
files_present = dir([Data_Folder,filesep, '*.tif']);

%Now register all images using base. Save as multitiff
for ff = 1:length(files_present)
     
    File_string = files_present(ff).name;
     
    info = imfinfo([Data_Folder, filesep, files_present(ff).name]); %Get image info
    num_t = numel(info);
    base = (imread([Data_Folder, filesep, files_present(ff).name], 1));
    [yb,xb] = size(base);
    
    %Loop through each time point, compare with base and register
    for t = 1:num_t
        unregistered = (imread([Data_Folder, filesep, files_present(ff).name], t));
        
        
        [yc,xc] = size(unregistered);
        
        %If image is not same size as base, resize
        if yc~=yb || xc~=xb
            unregistered = imresize(unregistered, [yb,xb]);
            [yc,xc] = size(unregistered);
        end
        
        c = normxcorr2(base,unregistered); %Calculate correlation between base and unregistered image
        
        %% Register image by calculating shift
        [y,x] = find(c == max(c(:)),1);
        
        %Find offset
        yoff = y - yc;
        xoff = x - xc;
        
        
        disp(['Filename...', files_present(ff).name, ' Time...', int2str(t), ' X offset...', num2str(xoff), ' Y offset...', num2str(yoff)]);
        
        
        if xoff < 0
            xoffa = abs(xoff)+1;
        else
            xoffa = xoff;
        end
        if yoff < 0
            yoffa = abs(yoff)+1;
        else
            yoffa = yoff;
        end
        
        if abs(xoff)>=20 || abs(yoff)>=20
            xoff=0;
            yoff=0;
        end
        
        register_by_offsets(unregistered, Result_Folder, files_present(ff).name, xc, yc, xoffa, yoffa, xoff, yoff, t)        
        
    end
    
end
end


function register_by_offsets(unregistered, Result_Folder, File_string, xc, yc, xoffa, yoffa, xoff, yoff, t)

% Adjust according to peak correlation
registered = uint16(zeros(yc+abs(yoffa), xc+abs(xoffa)));

if xoff~=0 && yoff==0
    if xoff < 0
        registered(:, xoffa:(xc+xoffa-1)) = unregistered;
        registered(:,end-xoffa+1:end) = [];
    else
        registered(:, 1:xc) = unregistered;
        registered(:,1:xoffa) = [];
    end
elseif xoff==0 && yoff~=0
    if yoff < 0
        registered(yoffa:(yc+yoffa-1), :) = unregistered;
        registered(end-yoffa+1:end,:) = [];
    else
        registered(1:yc, :) = unregistered;
        registered(1:yoffa,:) = [];
    end
elseif xoff~=0 && yoff~=0
    if xoff < 0 && yoff < 0
        registered(yoffa:(yc+yoffa-1), xoffa:(xc+xoffa-1)) = unregistered;
        registered(end-yoffa+1:end,:) = [];
        registered(:,end-xoffa+1:end) = [];
    elseif xoff > 0 && yoff > 0
        registered(1:yc, 1:xc) = unregistered;
        registered(1:yoffa,:) = [];
        registered(:,1:xoffa) = [];
    elseif xoff < 0 && yoff > 0
        registered(1:yc, xoffa:(xc+xoffa-1)) = unregistered;
        registered(1:yoffa,:) = [];
        registered(:,end-xoffa+1:end) = [];
    elseif xoff > 0 && yoff < 0
        registered(yoffa:(yc+yoffa-1), 1:xc) = unregistered;
        registered(end-yoffa+1:end,:) = [];
        registered(:,1:xoffa) = [];
    end
elseif xoff==0 && yoff==0
    registered = (unregistered);
end

%Save images
if t==1
    imwrite(registered,[Result_Folder, filesep,'Registered_',File_string],'tif');
else
    imwrite(registered,[Result_Folder, filesep,'Registered_',File_string],'tif','WriteMode','append');
end

end
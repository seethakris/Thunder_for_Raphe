# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 17:05:40 2015
Run PCA and get colormaps
@author: seetha
"""

#Import python libraries
from numpy import newaxis, squeeze, size, where, array, mean, zeros, round, reshape, float16, delete, hstack, savetxt
from scipy import stats
from numpy import asarray


#Import thunder libraries
from thunder import PCA
from thunder import Colorize


def run_pca(data,pca_components,required_pcs):
    model = PCA(k=pca_components).fit(data)
    imgs = model.scores.pack()
    
    if required_pcs == 0:
        new_imgs = imgs[1:,:,:]
    else:
        if len(imgs.shape)==3:
            new_imgs = imgs[required_pcs,:,:]
        else:
            new_imgs = imgs[required_pcs,:,:,:]

    return model, imgs, new_imgs
    
#Make maps and scatter plots of the pca scores with colormaps for plotting 
def make_pca_maps(data,pca, imgs, required_pcs, img_size_x, img_size_y,  num_pca_colors, num_samples, thresh_pca, color_map):
    
    reference = data.seriesMean().pack()
    if required_pcs == 0:
        maps = Colorize(cmap=color_map, scale=num_pca_colors).transform(imgs,background=reference, mixing=0.3)
    else:
        maps = Colorize(cmap=color_map, scale=num_pca_colors).transform(imgs,background=reference, mixing=0.3)

    pts = pca.scores.subset(num_samples, thresh=thresh_pca, stat='norm')
    
    if required_pcs == 0:
        pca_pts = list()
        for ii in xrange(1,size(pca.comps.T,1)):
            pca_pts.append(pts[:,ii][:,newaxis])        
        clrs = Colorize(cmap=color_map,scale=num_pca_colors).transform(pca_pts).squeeze()
    else:
        pca_pts = list()
        for ii in xrange(0, size(required_pcs)):
            pca_pts.append(pts[:,required_pcs[ii]][:,newaxis])  
        clrs = Colorize(cmap=color_map, scale=num_pca_colors).transform(pca_pts).squeeze()

    #Reconstruct the scores using the pca components
    if required_pcs == 0:
        recon = asarray(map(lambda x: (x[1] * pca.comps[1, :] + x[2] * pca.comps[2, :] + x[3] * pca.comps[3, :]).tolist(), pts))
    else:
        pts_list = pts.tolist()
        recon = zeros((size(pts_list,0),size(pca.comps,1)))
        for ii in range(0,size(pts_list,0)):
            for jj in range(0, size(required_pcs)):
                recon[ii,:] += pts_list[ii][required_pcs[jj]]*pca.comps[required_pcs[jj],:] 
            
    
    #Count number of unique colors in the images
    #Get number of planes based on map dimesnions
    if len(maps.shape)==3:
        num_planes = 1
    else:
        num_planes = size(maps,2)
    num_time = size(pca.comps.T,0)
    
    #Get specific color matches across animals and get mean and standard deviation       
    array1 = [map(int,single_dim) for single_dim in clrs] #Convert the colors to RGB integers
    new_array = [tuple(row) for row in array1] 
    unique_clrs = list(set(new_array))    #Get unique combination of colors   
    unique_clrs.remove((0,0,0))
    matches = [where((array(array1) == match).all(axis=1)) for match in unique_clrs] #Match the colors with the original rows
    
    matches_black = [where((array(array1) == match).all(axis=1)) for match in [0]]  
    pts_nonblack = delete(pts, matches_black, axis=0)
    clrs_nonblack = delete(clrs, matches_black, axis=0)
    
    #From maps get number of pixel matches with color for each plane
    matched_pixels = zeros((size(unique_clrs,0),num_planes))
#    while sum(matched_pixels) == 0 :
    array_maps = round(maps.astype(float16))
    matched_pixels = zeros((size(unique_clrs,0),num_planes))
    if len(maps.shape) == 3:
        array_maps_plane = reshape(array_maps, (size(array_maps,0)*size(array_maps,1),3))
        matched_pixels[:,0] = [size(where((array(array_maps_plane) == match).all(axis=1))) for match in unique_clrs]
    else:     
        for ii in xrange(0,num_planes):
            array_maps_plane = reshape(array_maps[:,:,ii,:], (size(array_maps,0)*size(array_maps,1),3))
            matched_pixels[:,ii] = [size(where((array(array_maps_plane) == match).all(axis=1))) for match in unique_clrs]
             
    
    #Find stats based on the color - but only use the subset of pixels in recon
    matched_signals = [structtype() for i in range(size(matches,0)*num_planes)]
    

    for ii in xrange(0,size(matches,0)):
        temp_ele = array(matches[ii])
        matched_signals[ii].clr_grped_signal = [array(recon[ele]) for ele in temp_ele[0,:]] #Get signals from the reconstruction that match the colors                     
    
    return maps, pts, pts_nonblack, clrs, clrs_nonblack, recon, unique_clrs, matched_pixels, matched_signals

def create_data_in_pca_space(imgs_pca, pca_components, required_pcs,thresh_pca, Working_Directory, name_for_saving_files):
    ## Reconstruct from PCA and save as textfile
    print 'Saving reconstructed PCA data in the text file '   
    if len(imgs_pca.shape) == 3:
        recon_img = zeros((size(imgs_pca,1),size(imgs_pca,2),1,size(pca_components,0)))
        z_plane = 1
    else:
        recon_img = zeros((size(imgs_pca,1),size(imgs_pca,2),size(imgs_pca,3),size(pca_components,0)))
        z_plane = size(imgs_pca,3)
        
    if required_pcs == 0:
        required_pcs = array(xrange(4))
        
    for ii in range(0,size(imgs_pca,1)):
        for jj in range(0,size(imgs_pca,2)):
            
            if z_plane == 1:
                pt_image = imgs_pca[:,ii,jj] 
                if (abs(pt_image)>thresh_pca).any():
                    for pp in range(0, size(required_pcs,0)):
                        recon_img[ii,jj,0,:] += pt_image[required_pcs[pp]]*pca_components[:,required_pcs[pp]]
            
            else:
                for zz in range(0, z_plane):
                    pt_image = imgs_pca[:,ii,jj,zz] 
                    if (abs(pt_image)>thresh_pca).any():
                        for pp in range(0, size(required_pcs,0)):
                            recon_img[ii,jj,zz,:] += pt_image[required_pcs[pp]]*pca_components[:,required_pcs[pp]]
    
                    
    ## Convert data to textfile format and save
    temp_numpy_array_for_thunder1 = zeros([size(recon_img, axis=0)*size(recon_img, axis=1)*size(recon_img,axis=2),3], dtype=int)
    temp_numpy_array_for_thunder2 = zeros([size(recon_img, axis=0)*size(recon_img, axis=1)*size(recon_img,axis=2),size(recon_img,axis=3)], dtype=float)

    count = 0
    for zz in range(0, z_plane):
        for jj in range(0,size(imgs_pca,2)):
            for ii in range(0,size(imgs_pca,1)):
                    
                temp_numpy_array_for_thunder1[count,0] = ii+1;
                temp_numpy_array_for_thunder1[count,1] = jj+1;
                temp_numpy_array_for_thunder1[count,2] = zz+1;
                # Create delta f/f values if necessary
                temp_numpy_array_for_thunder2[count,:] = recon_img[ii,jj,zz,:]                   
                count = count+1 
    
    #Save array
    temp_numpy_array_for_thunder = hstack((temp_numpy_array_for_thunder1, temp_numpy_array_for_thunder2))
    savetxt(Working_Directory+name_for_saving_files+'_pca_recon.txt', temp_numpy_array_for_thunder, fmt='%g')#Save as text file
  
class structtype():
    pass




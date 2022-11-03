#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 18:30:37 2021

@author: Mathew
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import frc
from sklearn.cluster import DBSCAN

# Camera settings
Pixel_size=103.0
    

# Settings
image_width=512
image_height=512
scale=8
photon_adu = 0.0265/0.96
# Thresholds
prec_thresh=25

filename_contains="FitResults.txt"

# Folders to analyse:
root_path=r"path_to_root"

# Cluterinng

to_cluster=1
eps_threshold=1
minimum_locs_threshold=100


pathList=[]




pathList.append(r"path_to_file")

resolution=[]
clus_resolution=[]
mean_precision=[]
mean_signal=[]
mean_SBR=[]
#  Generate SR image (points)
def generate_SR(coords):
    SR_plot_def=np.zeros((image_width*scale,image_height*scale),dtype=float)
    j=0
    for i in coords:
        
        xcoord=coords[j,0]
        ycoord=coords[j,1]
        scale_xcoord=round(xcoord*scale)
        scale_ycoord=round(ycoord*scale)
        # if(scale_xcoord<image_width and scale_ycoord<image_height):
        SR_plot_def[scale_ycoord,scale_xcoord]+=1
        
        j+=1
    return SR_plot_def

def cluster(coords):
     db = DBSCAN(eps=eps_threshold, min_samples=minimum_locs_threshold).fit(coords)
     labels = db.labels_
     n_clusters_ = len(set(labels)) - (1 if-1 in labels else 0)  # This is to calculate the number of clusters.
     print('Estimated number of clusters: %d' % n_clusters_)
     return labels
 
def generate_SR_cluster(coords,clusters):
    SR_plot_def=np.zeros((image_width*scale,image_height*scale),dtype=float)
    j=0
    for i in coords:
        if clusters[j]>-1:
            xcoord=coords[j,0]
            ycoord=coords[j,1]
            scale_xcoord=round(xcoord*scale)
            scale_ycoord=round(ycoord*scale)
            # if(scale_xcoord<image_width and scale_ycoord<image_height):
            SR_plot_def[scale_ycoord,scale_xcoord]+=1
        
        j+=1
    return SR_plot_def

def resolution_per_frame(frames,total):
    res=[]
    frame_range=[]
    length=int(total/frames)
    for i in range(1,length):
        framerem=i*frames
        print(framerem)
        loc_data = pd.read_table(fits_path)

        index_names = loc_data[loc_data['Frame']>framerem].index
        loc_data.drop(index_names, inplace = True)
       
           

        # Extract useful data:
        coords = np.array(list(zip(loc_data['X'],loc_data['Y'])))
    
            
        # Generate points SR (ESMB method):
        img=generate_SR(coords)
        
       
        img = frc.util.square_image(img, add_padding=False)
        img = frc.util.apply_tukey(img)
        # Apply 1FRC technique
        frc_curve = frc.one_frc(img)
        
        img_size = img.shape[0]
        xs_pix = np.arange(len(frc_curve)) / img_size
        
        xs_nm_freq = xs_pix * scale_frc
        frc_res, res_y, thres = frc.frc_res(xs_nm_freq, frc_curve, img_size)
        
        res.append(frc_res)
        frame_range.append(framerem)
    plt.plot(frame_range,res)
    plt.xlabel('Frames',size=20)
    plt.ylabel('Resolution (nm)',size=20)
    plt.show()
    return res,frame_range
    

for path in pathList:
    print(path)
    path=path+"/"

    # Perform the fitting

    # Load the fits:
    for root, dirs, files in os.walk(path):
                for name in files:
                        if filename_contains in name:
                            if ".txt" in name:
                                if ".tif" not in name:
                                    resultsname = name
                                    print(resultsname)
    
                                    fits_path=path+resultsname
                                    # fits_path=path+filename_contains
                                    
                                    
                                    loc_data = pd.read_table(fits_path)
                                    
                                    index_names = loc_data[loc_data['Precision (nm)']>prec_thresh].index
                                    loc_data.drop(index_names, inplace = True)
                                   
                                       
  
                          
                                  
             
                                    # Extract useful data:
                                    coords = np.array(list(zip(loc_data['X'],loc_data['Y'])))
                                    precsx= np.array(loc_data['Precision (nm)'])
                                    precsy= np.array(loc_data['Precision (nm)'])
                                    xcoords=np.array(loc_data['X'])
                                    ycoords=np.array(loc_data['Y'])
                                    signal=np.array(loc_data['Signal'])
                                    background=np.array(loc_data['Background'])
                                    
                                    
                                    precs_nm=precsx
                                    
                                    signal_above_background=(signal-background)*photon_adu
                                    ave_signal= signal_above_background.mean()
                                    signal_bg_ratio=signal/background
                                    ave_sbr=signal_bg_ratio.mean()
                                    
                                    average_precision=precs_nm.mean()
                                    
                                    plt.hist(precs_nm, bins = 50,range=[0,100], rwidth=0.9,color='#ff0000')
                                    plt.xlabel('Precision (nm)',size=20)
                                    plt.ylabel('Number of Features',size=20)
                                    plt.title('Localisation precision',size=20)
                                    plt.savefig(path+"Precision.pdf")
                                    plt.show()
                                    
                                    # Generate points SR (ESMB method):
                                    img=generate_SR(coords)
                                    
                                    scale_frc = scale/Pixel_size

                                    img = frc.util.square_image(img, add_padding=False)
                                    img = frc.util.apply_tukey(img)
                                    # Apply 1FRC technique
                                    frc_curve = frc.one_frc(img)
                                    
                                    img_size = img.shape[0]
                                    xs_pix = np.arange(len(frc_curve)) / img_size
                                    
                                    xs_nm_freq = xs_pix * scale_frc
                                    frc_res, res_y, thres = frc.frc_res(xs_nm_freq, frc_curve, img_size)
                                    
                                    text='Resolution = '+str(round(frc_res,2))+' nm'
                                    plt.plot(xs_nm_freq, thres(xs_nm_freq))
                                    plt.plot(xs_nm_freq, frc_curve)
                                    plt.xlabel('Spatial resolution (nm$^{-1}$)',size=20)
                                    plt.ylabel('FRC',size=20)
                                    
                                    plt.title(text,size=12)
                                    plt.savefig(path+"Resolution.pdf")
                                    plt.show()
                                    
                                    print(frc_res)
                                    
                                    if to_cluster==1:
                                        clusters=cluster(coords)
                                    
                                        # Check how many localisations per cluster
                                     
                                        cluster_list=clusters.tolist()    # Need to convert the dataframe into a list- so that we can use the count() function. 
                                        maximum=max(cluster_list)+1  
                                        
                                        
                                        cluster_contents=[]         # Make a list to store the number of clusters in
                                        
                                        for i in range(0,maximum):
                                            n=cluster_list.count(i)     # Count the number of times that the cluster number i is observed
                                           
                                            cluster_contents.append(n)  # Add to the list. 
                                        
                                        if len(cluster_contents)>0:
                                            average_locs=sum(cluster_contents)/len(cluster_contents)
                                     
                                            
                                            cluster_arr=np.array(cluster_contents)
                                        
                                            median_locs=np.median(cluster_arr)
                                            mean_locs=cluster_arr.mean()
                                            std_locs=cluster_arr.std()
                                            
                                        
                                            # Generate the SR image.
                                            SR_img=generate_SR_cluster(coords,clusters)
                                            
                                          
                                            img = frc.util.square_image(SR_img, add_padding=False)
                                            img = frc.util.apply_tukey(SR_img)
                                            # Apply 1FRC technique
                                            frc_curve = frc.one_frc(img)
                                            
                                            img_size = img.shape[0]
                                            xs_pix = np.arange(len(frc_curve)) / img_size
                                            
                                            xs_nm_freq = xs_pix * scale_frc
                                            clu_frc_res, clu_res_y, clu_thres = frc.frc_res(xs_nm_freq, frc_curve, img_size)
                                            
                                            textclus='Resolution = '+str(round(clu_frc_res,2))+' nm'
                                            plt.plot(xs_nm_freq, thres(xs_nm_freq))
                                            plt.plot(xs_nm_freq, frc_curve)
                                            plt.xlabel('Spatial resolution (nm$^{-1}$)',size=20)
                                            plt.ylabel('FRC',size=20)
                                            
                                            plt.title(textclus,size=12)
                                            plt.savefig(path+"Cluster_Resolution.pdf")
                                            plt.show()
                                            
                                            
                                            print(clu_frc_res)
                                 
                                    
                                    resolution.append(frc_res)   
                                    clus_resolution.append(clu_frc_res)
                                    mean_precision.append(average_precision)
                                    mean_signal.append(ave_signal)
                                    mean_SBR.append(ave_sbr)
                                    
                                    df = pd.DataFrame(list(zip(resolution,clus_resolution,mean_precision,mean_signal,mean_SBR)),columns =['Resolution', 'Custered Resolution','Precision','Signal','SBR'])
                                    df.to_csv(root_path+ 'Resolution.csv', sep = '\t')
                                        
                                    
                                    
                                    

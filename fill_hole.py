# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 14:39:00 2018

@author: dongming.wei@sjtu.edu.cn
"""
import md
import SimpleITK as sitk
import os

'''
funtion 'fill hole'
input: segmentation image name and iteration index
output: filled segmentation image
'''


def Fillhole(IMG_NAME, iter=30):
    img = sitk.ReadImage(IMG_NAME)
    filled = sitk.VotingBinaryIterativeHoleFilling(img)
    # img_data = sitk.GetArrayFromImage(img)
    for i in range(iter):
        filled = sitk.VotingBinaryIterativeHoleFilling(filled)
        print i

    return filled  # the filled can be saved using sitk.WriteImage(filled,'path')


# Source_root_3d = '/data0/geyunhao/filehole/ZS13221585/Untitled1.nii.gz'
# out_root_3d = '/data0/geyunhao/filehole/ZS13221585/mr_100.nii.gz'
Source_root_3d = '/data0/geyunhao/filehole/'
out_root_3d = '/data0/geyunhao/filehole/'
for roots, dirs, files in os.walk(Source_root_3d):
    for file in files:
        file_path = os.path.join(roots, file)
        if 'Untitled.nii.gz' in file:  # MR

            output = Fillhole(file_path)
            target_filename = roots.split('/')[-1] + '_ct.nii.gz'
            sitk.WriteImage(output, out_root_3d + '/' + target_filename)  #
            print target_filename





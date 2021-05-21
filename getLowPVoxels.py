import numpy as np
import nibabel as nib
import math
import argparse

def get_indices_of_k_smallest(arr, k):
    idx = np.argpartition(arr.ravel(), k)
    return tuple(np.array(np.unravel_index(idx, arr.shape))[:, range(min(k, 0), max(k, 0))])

img = nib.load("normal_corr_pvals_10.nii.gz")
imgarr = np.array(img.dataobj)
print(imgarr.shape)
overallcorr = np.sum(imgarr[:,:,:,:],axis = 3)
print(overallcorr.shape)
maskarr = np.zeros(overallcorr.shape)

super_threshold_indices = overallcorr == 0
overallcorr[super_threshold_indices] = np.inf

overallcorr[0:2,:,:] = np.inf
overallcorr[:,0:2,:] = np.inf
overallcorr[:,:,0:2] = np.inf

r = np.array(get_indices_of_k_smallest(overallcorr, 20000)) #get 20000 most correlated voxels 2%
rvals = np.zeros((r.shape[1],4))
print(r.shape)
print(rvals.shape)
for i in range (0,r.shape[1]):
	rvals[i][3] = overallcorr[r[0][i]][r[1][i]][r[2][i]]
	rvals[i,0:3] = r[:,i]
	maskarr[r[0][i]][r[1][i]][r[2][i]] = 4000

output_img = nib.Nifti1Image(maskarr, img.affine)
nib.save(output_img, "Mask20LowProbs")
sortedvoxels = rvals[rvals[:, 3].argsort()] 
print(sortedvoxels)

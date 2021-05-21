import numpy as np
import nibabel as nib
import math
import argparse

def get_indices_of_k_smallest(arr, k):
    idx = np.argpartition(arr.ravel(), k)
    return tuple(np.array(np.unravel_index(idx, arr.shape))[:, range(min(k, 0), max(k, 0))])

img = nib.load("normal_corr_pvals_10.nii.gz")
imgarr = np.array(img.dataobj)
maskarr = np.zeros(imgarr.shape)
print(imgarr.shape)
for j in range (0,imgarr.shape[3]):
	pvalvol = np.array(imgarr[:,:,:,j])
	print(pvalvol.shape)

	super_threshold_indices = pvalvol == 0 #correct for background
	pvalvol[super_threshold_indices] = 1
	pvalvol[0:2,:,:] = 1
	pvalvol[:,0:2,:] = 1

	r = np.array(get_indices_of_k_smallest(pvalvol, 50000)) #get 20000 most correlated voxels 2%
	rvals = np.zeros((r.shape[1],4))
	for i in range (0,r.shape[1]):
		rvals[i][3] = pvalvol[r[0][i]][r[1][i]][r[2][i]]
		rvals[i,0:3] = r[:,i]
		maskarr[r[0][i]][r[1][i]][r[2][i]][j] = 4000
		#sortedvoxels = rvals[rvals[:, 3].argsort()] 

output_img = nib.Nifti1Image(maskarr, img.affine)
nib.save(output_img, "Mask50LowProbsTime.nii.gz")
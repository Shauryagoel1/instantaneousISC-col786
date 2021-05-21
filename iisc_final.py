#!/usr/bin/python
import sys
import os
import argparse
import logging
import textwrap
from glob import glob
import numpy as np
# import scipy.misc as scm
import scipy.special as scm
import nibabel as nib
from statsmodels.sandbox.stats.multicomp import fdrcorrection0
import math
from scipy.stats import t
# from instantaneos_isc import *

# GLOBAL VARIABLES :
#
xdim, ydim, zdim = 0, 0, 0
epsilon = 0.0000001
# vt_pos_counter, vt_neg_counter = np.zeros((200000, 100))
nsubs = 0
# Setting up logger
logger = logging.getLogger(__name__)
io_error = []
# Set up argument parser


def parse_arguments(args):
    parser = argparse.ArgumentParser(
        description=("Python-based command-line program for computing "
                     "leave-one-out instantaneous intersubject correlations (IISCs)"),
        epilog=(textwrap.dedent("""
      This program provides a simple Python-based command-line interface (CLI)
      for running intstantaneous intersubject correlation (IISC) analysis.
      IISCs are computed using the leave-one-out approach where each subject's
      time series (per voxel) is correlated with the average of the remaining
      subjects' time series.
      The --input should be two or more 4-dimensional NIfTI (.nii or
      .nii.gz) files, one for each subject. Alternatively, a wildcard can be used
      to indicate multiple files (e.g., *.nii.gz).
      The --output path should be given as just the directory in which it gives the outputs
      in the IISC_analysis folder in this directory.
      The --mask is again an optional input if masking is to be done.
      The --delta is how many time steps back, iisc needs to be calculated upon.
      For example, if delta is given 3, then the difference is taken between B_sv(i) - B_sv(i-3)
      Default is taken to be 1.
      """)), formatter_class=argparse.RawTextHelpFormatter)
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required_arguments')
    required.add_argument("-i", "--input", nargs='+', required=True,
                          help=("NIfTI input files on which to compute ISCs"))
    required.add_argument("-o", "--output", type=str, required=True,
                          help=("output directory inside which a folder with name IISC_analysis"
                                "would be created containing the results"))
    optional.add_argument("-f", "--file", action="store_true",
                          dest="file",
                          help="whether input is to be given as a file or not")
    optional.add_argument("-m", "--mask", type=str,
                          help=("NIfTI mask file for masking input data"))
    optional.add_argument("-d", "--delta", type=int,
                          help=("Time delta for taking difference"))
    # optional.add_argument("-s", "--summarize", type=str,
    #                       choices=['mean', 'median', 'stack'],
    #                       help=("summarize results across participants "
    #                             "using either mean, median, or stack"))
    optional.add_argument("-n", "--normal", action="store_true",
                          dest="do_normal",
                          help=("consider a normal model on subjects to calculate IISC probabilities"))
    optional.add_argument("-t", "--tail", type=int,
                          help=("One-tailed or two-tailed t-test to be performed"))
    optional.add_argument("-b", "--baseline", type=int, nargs = '+',
                          help=("Specify baseline volume"))
    parser.add_argument('--version', action='version',
                        version='iisc.py 1.0.0')
    parser._action_groups.append(optional)
    args = parser.parse_args(args)
    return args

# Helper functions

# getting coordinates from 1D (xdim*ydim*zdim) space to 3D xdim X ydim X zdim


def get_coordinates(x_dim, y_dim, z_dim):
    coordinates = np.zeros(x_dim*y_dim*z_dim)
    pos = 0
    for i in range(x_dim):
        for j in range(y_dim):
            for k in range(z_dim):
                coordinates[pos] = (i, j, k)
                pos += 1
    return coordinates

# Loading mask (taken from isc_cli.py github)


def load_mask(mask_arg):
    # Load mask from file
    mask = nib.load(mask_arg).get_fdata().astype(bool)
    # Get indices of voxels in mask
    mask_indices = np.where(mask)
    logger.info("finished loading mask from "
                "'{0}'".format(mask_arg))
    return mask, mask_indices


# def diff_calculate(func_data, k=1, baseline):
def diff_calculate(func_data, baseline):

    print(func_data.shape)
    # voxel_data_diff = np.subtract(func_data, np.roll(func_data, k, axis=1))
    voxel_data_diff = func_data
    baseline = np.broadcast_to(baseline,func_data.shape)
    print(sum(baseline[:,0]))
    print(sum(baseline[:,1]))
    voxel_data_diff[baseline!=0] = 100*(np.subtract(func_data[baseline!=0],baseline[baseline!=0]))/baseline[baseline!=0]
	# voxel_data_diff[baseline==0] = 100*(voxel_data_diff[baseline==0])
    # voxel_data_diff[baseline==0] = 100*(voxel_data_diff[baseline==0])


    # voxel_data_diff = func_data
    # for i in range(func_data.shape[0]):
    # 	for j in range(func_data.shape[1]):

	# 	    if baseline[i][j] != 0:
	# 	    	voxel_data_diff[i][j] = 100*np.subtract(func_data[i][j], baseline[i][j])/baseline[i][j]
    # # for i in range(k):
    # #     voxel_data_diff[:, i] = 0
    return voxel_data_diff


def normal_diff_calculate(vt_sum_counter, vt_ssum_counter, func_data, k, baseline):
    # diff_vt_i = diff_calculate(func_data, k)
    diff_vt_i = diff_calculate(func_data, baseline)
    vt_sum_counter += diff_vt_i
    vt_ssum_counter += np.square(diff_vt_i)
    return vt_sum_counter, vt_ssum_counter


def bin_diff_calculate(vt_pos_counter, vt_neg_counter, func_data, k=1):
    # global vt_pos_counter, vt_neg_counter
    #     V, T = func_data.shape[0], func_data.shape[1]
    voxel_data_bin = np.sign(diff_calculate(func_data, k))
    vt_pos_counter += np.where(voxel_data_bin == 1, 1, 0)
    vt_neg_counter += np.where(voxel_data_bin == -1, 1, 0)
    return (vt_pos_counter, vt_neg_counter)


def file_to_input_fns(input_file):
    input_fns = []
    with open(input_file) as f:
        for line in f.readlines():
            input_fns.append(line.split("\n")[0])
    return input_fns


def load_diff_data(input_arg, baseline, mask=None, delta=1, do_normal=True):
    # global vt_pos_counter, vt_neg_counter
    global nsubs, xdim, ydim, zdim
    input_fns = [fn for fns in [glob(fn) for fn in input_arg]
                 for fn in fns]
    if len(input_fns) < 2:
        raise ValueError("--input requires two or more "
                         "input files for IISC computation")
    count = 0
    # data = []
    affine, data_shape = None, None
    nsubs = len(input_fns)
    for input_fn in input_fns:
        # Load in the NIfTI image using NiBabel and check shapes
        try:
           input_img = nib.load(input_fn)
           if not data_shape:
               data_shape = input_img.shape
               vt_pos_counter = np.zeros((np.product(data_shape[:3]), data_shape[3])).astype(int)
               vt_neg_counter = np.zeros((np.product(data_shape[:3]), data_shape[3])).astype(int)
               vt_sum_counter = np.zeros(
                    (np.product(data_shape[:3]), data_shape[3])).astype(float)
               vt_ssum_counter = np.zeros(
                    (np.product(data_shape[:3]), data_shape[3])).astype(float)
               shape_fn = input_fn
           if len(input_img.shape) != 4:
               raise ValueError("input files should be 4-dimensional "
                             "(three spatial dimensions plus time)")
           if input_img.shape != data_shape:
               raise ValueError("input files have mismatching shape: "
                             "file '{0}' with shape {1} does not "
                             "match file '{2}' with shape "
                             "{3}".format(input_fn,
                                          input_img.shape,
                                          shape_fn,
                                          data_shape))
           logger.debug("input file '{0}' NIfTI image is "
                     "shape {1}".format(input_fn, data_shape))

           # Save the affine and header from the first image
           if affine is None:
               affine, header = input_img.affine, input_img.header
               logger.debug("using affine and header from "
                         "file '{0}'".format(input_fn))

            # Get data from image and apply mask (if provided)
            # Converting the 4D (x,y,z,t) -> 2D (voxel, t)
           input_data = input_img.get_fdata()
           if isinstance(mask, np.ndarray):
               input_data = input_data[mask]

           input_data = input_data.reshape((
                np.product(input_data.shape[:3]),
                input_data.shape[3]))
            # difference calculation for IISC from instantaneous_isc.py
           baseline_data = np.zeros((np.product(data_shape[:3]), 1))
           for base_ind in baseline:
               fdata = input_data[:,base_ind:base_ind+1]/len(baseline)
               baseline_data = baseline_data + fdata
               # print(sum(baseline_data))
           # baseline_data = baseline_data/len(base_ind)


           # baseline_data = input_data[:,baseline:baseline+1]
           #baseline_data = np.sum(input_data[:,:],axis = 1)/(input_data.shape[1])
           #baseline_data = baseline_data.reshape(baseline_data.shape[0],1)
           print(baseline_data.shape)
           if (do_normal):
                vt_sum_counter, vt_ssum_counter = normal_diff_calculate(
                    vt_sum_counter, vt_ssum_counter, input_data, delta, baseline_data)
           logger.info("finished loading data from "
                    "'{0}'".format(input_fn))
            # At the end of this we have equivalents of voxel_data_sum_pos and voxel_data_sum_neg that were defined in the original code
        except IOError:
           print("{0} has IO error. Ignoring subject".format(input_fn))
           io_error.append(input_fn)
           pass
    with open('io_error.txt','w') as f:
        for elem in io_error:
            f.write("{0}\n".format(elem))
    return vt_pos_counter, vt_neg_counter, vt_sum_counter, vt_ssum_counter, affine, header



def calc_normal_pvals(vt_sum_counter, vt_ssum_counter, tail=1):
    global nsubs
    sample_mean = vt_sum_counter/nsubs
    sample_smean = vt_ssum_counter/nsubs
    sample_sigma = np.sqrt((sample_smean - np.square(sample_mean))/(nsubs - 1))
    # t_array = sample_mean/sample_sigma
    t_array = np.where(sample_sigma == 0, np.inf, sample_mean/(sample_sigma))
    if (tail == 2):
        return 2*t.sf(np.abs(t_array), nsubs-1)
    else:
        return t.sf(t_array, nsubs-1)



# Correction for multiple comparisons (currently only for positive probability)


def multiple_comp(vt_data, prob_data,sign_indicator):
    # global vt_data
    global nsubs
    # global xdim, ydim, zdim
    V, T = prob_data.shape

    voxel_data_sum_sign = np.sign(vt_data - sign_indicator - 0.0)
    # corr_prob_map1 = np.zeros((xdim, ydim, zdim, T))
    # corr_prob_map2 = np.zeros((xdim, ydim, zdim, T))
    # prob_map = np.zeros((xdim, ydim, zdim, T))
    # coordinates = get_coordinates(xdim, ydim, zdim)
    corr_prob_map1, corr_prob_map2, prob_map, corr_prob_data1, corr_prob_data2 = np.zeros(
        (V, T)), np.zeros((V, T)), np.zeros((V, T)), np.zeros((V, T)), np.zeros((V, T))

    for i in range(T):
        t1, corr_prob_data1[:, i] = fdrcorrection0(
            prob_data[:, i], alpha=0.05)
        # Multiply the direction
        # print("Greater than threshold values %d: %d" % (i, np.sum(t1)))
        t2, corr_prob_data2[:, i] = fdrcorrection0(
            prob_data[:, i], alpha=0.10)

    corr_prob_map1 = np.where(
        corr_prob_data1 != 0, voxel_data_sum_sign * np.log10(corr_prob_data1) * (-1), 0)
    corr_prob_map2 = np.where(
        corr_prob_data2 != 0, voxel_data_sum_sign * np.log10(corr_prob_data2) * (-1), 0)
    prob_map = np.where(
        prob_data != 0, voxel_data_sum_sign * np.log10(prob_data) * (-1), 0)

    return (corr_prob_map1, corr_prob_map2, prob_map, corr_prob_data1, corr_prob_data2)

# Function to convert array back to NIfTI image


def to_nifti(iisc, affine, header, mask_indices):
    # Output ISCs image shape
    i, j, k = header.get_data_shape()[:3]
    T = iisc.shape[1]
    nifti_shape = (i, j, k, T)
    # Reshape masked data
    if mask_indices:
        nifti_iisc = np.zeros(nifti_shape)
        for t in range(T):
            nifti_iisc[mask_indices, t] = iisc[:, t]
    else:
        nifti_iisc = iisc.reshape(nifti_shape)
    # Construct output NIfTI image
    nifti_img = nib.Nifti1Image(nifti_iisc, affine)
    return nifti_img


# Function to save NIfTI images

# Save one data at a time
def save_data(iisc, affine, header, output_name, mask_indices=None):
    output_img = to_nifti(iisc, affine, header, mask_indices)
    nib.save(output_img, output_name)
    logger.info("saved IISC output to {0}".format(output_name))


def main(args):
    # Set up the logger
    logging.basicConfig(level=20)

    # Parse the arguments
    args = parse_arguments(args)
    print(args.do_normal)
    # Get optional mask
    mask, mask_indices = load_mask(args.mask) if (args.mask) else None, None

    # Get optional delta
    time_delta = args.delta if (args.delta) else 1

    # Get optional do_normal
    do_normal = args.do_normal if (args.do_normal) else False

    # Get optional tail
    tail = args.tail if (args.tail) else 1

    # Get optional baseline volume
    # if(args.baseline is not None):
    #     base_ind = args.baseline
    # else:
    #     base_ind = 0
    base_ind = args.baseline if (args.baseline) else [0]
    # print(base_ind)

    # Get optional file input
    input_fns = file_to_input_fns(args.input[0]) if (args.file) else args.input

    # Load data
    vt_pos_counter, vt_neg_counter, vt_sum_counter, vt_ssum_counter, affine, header = load_diff_data(
        input_fns, base_ind, mask=mask, delta=time_delta, do_normal=do_normal)

    # compute probabilities

    # How many volumes can the probability calculation be done on in one go without memory error.
    Tbar = 240
    # normal
    if (do_normal):
        V, T = vt_sum_counter.shape
        if (T <= Tbar):
            normal_pvals = calc_normal_pvals(
                vt_sum_counter, vt_ssum_counter, tail=tail)
        else:
            normal_pvals = np.zeros((V, T))
            for i in range(int(T/Tbar)):
                normal_pvals[:, Tbar*i:Tbar*(i+1)] = calc_normal_pvals(
                    vt_sum_counter[:, Tbar*i:Tbar*(i+1)], vt_ssum_counter[:, Tbar*i:Tbar*(i+1)], tail=tail)
        logger.info("Finished calculating normal probabilities")

    # multiple comparisons
    # for positive probabilities
    if (do_normal):
        corr_prob_map3, corr_prob_map4, prob_map2, corr_prob_data1, corr_prob_data2 = multiple_comp(
            vt_sum_counter, normal_pvals,0)
    logger.info("Finished correcting for multiple comparisons")

    save_folder = args.output + "/IISC_analysis_" + str(time_delta)
    # Make the output folder
    try:
        os.mkdir(save_folder)
    except:
        if (os.path.exists(save_folder)):
            pass
        else:
            save_folder = args.output
    # iiscs = [corr_prob_map1, corr_prob_map2, prob_map,
    #          vt_pos_counter/nsubs,
    #          corr_prob_map3, corr_prob_map4, prob_map2,
    #          vt_sum_counter/nsubs]
    # output_names = ["corr_prob_05.nii.gz", "corr_prob_10.nii.gz",
    #                 "prob.nii.gz", "frac_pos.nii.gz", "frac_neg.nii.gz"]

    # output_fns = list(
    #     map(lambda x: save_folder + x, output_names))

    if (do_normal):
        #save_data(corr_prob_data1, affine, header, save_folder +
        #          "/normal_corr_pvals_05.nii.gz", mask_indices=mask_indices)
        save_data(corr_prob_data2, affine, header, save_folder +
                  "/normal_corr_pvals_10.nii.gz", mask_indices=mask_indices)
        #save_data(corr_prob_map3, affine, header, save_folder +
        #          "/normal_corr_prob_05.nii.gz", mask_indices=mask_indices)
        save_data(corr_prob_map4, affine, header, save_folder +
                  "/normal_corr_prob_10.nii.gz", mask_indices=mask_indices)
        save_data(prob_map2, affine, header, save_folder +
                  "/normal_prob.nii.gz", mask_indices=mask_indices)
        save_data(normal_pvals, affine, header, save_folder +
                  "/normal_pvals.nii.gz", mask_indices=mask_indices)
        #save_data(vt_sum_counter/nsubs, affine, header, save_folder +
        #          "/sample_mean.nii.gz", mask_indices=mask_indices)
        #save_data(vt_ssum_counter/nsubs, affine, header, save_folder +
        #          "/sample_smean.nii.gz", mask_indices=mask_indices)

    # save_data(iiscs, affine, header, output_fns, mask_indices=mask_indices)


if __name__ == '__main__':
    main(sys.argv[1:])

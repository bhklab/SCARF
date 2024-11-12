import os, torch

import pandas as pd
import numpy as np
import SimpleITK as sitk

import monai.metrics as met


def calc_metrics(ground_truth: sitk.Image, prediction: sitk.Image):
    # Define list of labels for different organs at risk (OARs)
    labels = ['Brainstem', 'SpinalCord', 'Esophagus', 'Larynx', 'Mandible', 'Parotid_L', 'Parotid_R', 'Acoustic_L', 'Acoustic_R',
              'BrachialPlex_R', 'BrachialPlex_L', 'Lens_L', 'Lens_R', 'Eye_L', 'Eye_R', 'Nrv_Optic_L', 'Nrv_Optic_R', 'OpticChiasm', 'Lips']

    # Extract image spacing information from the prediction image
    spacing = prediction.GetSpacing()
    print(f'Image spacing: {spacing}')

    # Convert SimpleITK images to numpy arrays for processing
    prediction = sitk.GetArrayFromImage(prediction)
    ground_truth = sitk.GetArrayFromImage(ground_truth)

    # Initialize lists to hold metric values for each label
    hd = []
    dice = []
    surf_dist = []
    jaccard = []
    added_path_length = []
    false_neg_path_length = []
    false_neg_vol = []
    oars = []

    # Calculate metrics for each label
    for idx, label in enumerate(labels):
        # Prepare binary masks for current label in prediction and ground truth
        slc = np.copy(prediction)
        y = np.copy(ground_truth)
        
        slc[slc != idx + 1] = 0
        slc[slc != 0] = 1
        y[y != idx + 1] = 0
        y[y != 0] = 1

        # Calculate overlap-based metrics (Dice, Jaccard)
        dc = (np.sum(slc[y == 1]) * 2.0) / (np.sum(slc) + np.sum(y))
        j = np.sum(slc[y == 1]) / (np.sum(slc) + np.sum(y) - np.sum(slc[y == 1]))

        # Calculate path-based metrics (False Negative Volume, Added Path Length, False Negative Path Length)
        fnv = FalseNegativeVolume(slc, y)
        apl = AddedPathLength(slc, y)
        fnpl = FalseNegativePathLength(slc, y)

        # Convert numpy arrays to torch tensors for computing distance-based metrics
        slc = torch.from_numpy(slc)
        y = torch.from_numpy(y)

        # Calculate Hausdorff and average surface distance metrics
        h = met.compute_hausdorff_distance(slc.unsqueeze(0).unsqueeze(
            0), y.unsqueeze(0).unsqueeze(0), percentile=95, include_background=False, spacing=spacing)
        s = met.compute_average_surface_distance(slc.unsqueeze(0).unsqueeze(0), y.unsqueeze(0).unsqueeze(0), include_background=False, spacing=spacing)

        # Append calculated metrics to lists
        dice.append(dc)
        hd.append(h[0].item())
        surf_dist.append(s[0].item())
        jaccard.append(j)
        added_path_length.append(apl)
        false_neg_path_length.append(fnpl)
        false_neg_vol.append(fnv)
        oars.append(label)

    # Create a DataFrame to summarize all computed metrics
    metrics_df = pd.DataFrame({
        'OAR': oars,
        'VolDice': dice,
        '95HD': hd,
        'SurfDist': surf_dist,
        'APL': added_path_length,
        'FNPL': false_neg_path_length,
        'FNV': false_neg_vol
    })

    return metrics_df
"""
TAKEN FROM: https://github.com/kkiser1/Autosegmentation-Spatial-Similarity-Metrics/

Each function takes "auto" and "gt" arguments, which are respectively the autosegmentation and ground truth 
segmentation represented as three-dimensional NumPy arrays. The array dimensions should be the dimensions 
of the original image, and each array element should be 0 if its corresponding image pixel is not part of 
the segmentation or 1 if it is.
"""

def FalseNegativeVolume(auto, gt):
    '''
    Returns the false negative volume, in pixels
    
    Steps:
    1. Find pixels where the mask is present in gt but not in auto (wherever gt is 1 but auto is 0)
    2. Convert comparison from bool to int
    3. Compute # pixels
    '''
    
    fnv = (gt > auto).astype(int).sum()
    return fnv


def AddedPathLength(auto, gt):
    '''
    Returns the added path length, in pixels
    
    Steps:
    1. Find pixels at the edge of the mask for both auto and gt
    2. Count # pixels on the edge of gt that are not in the edge of auto
    '''
    
    # Check if auto and gt have same dimensions. If not, then raise a ValueError
    if auto.shape != gt.shape:
        raise ValueError('Shape of auto and gt must be identical!')

    # edge_auto has the pixels which are at the edge of the automated segmentation result
    edge_auto = getEdgeOfMask(auto)
    # edge_gt has the pixels which are at the edge of the ground truth segmentation
    edge_gt = getEdgeOfMask(gt)
    
    # Count # pixels on the edge of gt that are on not in the edge of auto
    apl = (edge_gt > edge_auto).astype(int).sum()
    
    return apl 


def FalseNegativePathLength(auto, gt):
    '''
    Returns the false negative path length, in pixels
    
    Steps:
    1. Find pixels at the edge of the mask for gt
    2. Count # pixels on the edge of gt that are not in the auto mask volume
    '''
    
    # Check if auto and gt have same dimensions. If not, then raise a ValueError
    if auto.shape != gt.shape:
        raise ValueError('Shape of auto and gt must be identical!')
    
    # edge_gt has the pixels which are at the edge of the ground truth segmentation
    edge_gt = getEdgeOfMask(gt)
    
    # Count # pixels where the edges in grount truth == 1 and auto == 0
    fnpl = (edge_gt > auto).astype(int).sum() 
    
    return fnpl

def getEdgeOfMask(mask):
    '''
    Computes and returns edge of a segmentation mask
    '''
    # edge has the pixels which are at the edge of the mask
    edge = np.zeros_like(mask)
    
    # mask_pixels has the pixels which are inside the mask of the automated segmentation result
    mask_pixels = np.where(mask > 0)

    for idx in range(0,mask_pixels[0].size):

        x = mask_pixels[0][idx]
        y = mask_pixels[1][idx]
        z = mask_pixels[2][idx]

        # Count # pixels in 3x3 neighborhood that are in the mask
        # If sum < 27, then (x, y, z) is on the edge of the mask
        if mask[x-1:x+2, y-1:y+2, z-1:z+2].sum() < 27:
            edge[x,y,z] = 1
            
    return edge


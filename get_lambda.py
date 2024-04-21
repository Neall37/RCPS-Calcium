import json
import matplotlib.pyplot as plt
from numpy import array, zeros
from tifffile import imread, imwrite
from glob import glob
import numpy as np
import cv2
from tqdm import tqdm
import torch.nn.functional as F
from scipy.stats import norm
from scipy.optimize import brentq
from data import fourier_pass
from skimage.filters import gaussian
from scipy.ndimage import median_filter
from skimage.restoration import denoise_nl_means
from skimage import morphology
from skimage.filters import threshold_local


def empirical_risk_percell(T, masks):
    empirical_risks = []

    # Iterate over each mask in the dataset
    for i in range(len(masks)):
        # print(i)
        # Extract the current prediction
        current_T = T[i]

        # Ensure predictions are binary
        current_T = (current_T > 0).astype(int)

        # Get the regions for the current mask
        regions = masks[i]["regions"]
        image_risk_sum = 0.0


        for region in regions:
            # Extract the coordinates of the current region
            coords = region["coordinates"]
            cell_mask = np.zeros_like(current_T)
            cell_mask[tuple(zip(*coords))] = 1
            cell_size = cell_mask.sum()

            # Calculate the size of the missed region for the current cell
            missed_size = np.logical_and(cell_mask, np.logical_not(current_T)).sum()

            # Calculate the risk for the current cell
            if cell_size > 0:  # Avoid division by zero
                cell_risk = missed_size / cell_size
                image_risk_sum += cell_risk

        # Calculate the empirical risk for the current image
        image_risk = image_risk_sum / len(regions)
        empirical_risks.append(image_risk)

    # Calculate and return the mean and standard deviation of empirical risks across all images
    mean_risk = np.mean(empirical_risks)
    std_risk = np.std(empirical_risks)

    return mean_risk, std_risk



def tomask(coords,dims):
    mask = zeros(dims)
    y_coords, x_coords = zip(*coords)
    mask[np.array(y_coords), np.array(x_coords)] = 1
    return mask


def risk_UBC(regions, masks, gamma, delta, lam):
    Tlam = regions >= -lam
    Rhat, sigmahat = empirical_risk_percell(Tlam, masks)
    t = -norm.ppf(delta) * sigmahat / np.sqrt(regions.shape[0])
    return Rhat, Rhat + t

def _condition(regions, masks, gamma, delta, lam):
    Tlam = regions >= -lam
    Rhat, sigmahat = empirical_risk_percell(Tlam, masks)
    t = -norm.ppf(delta) * sigmahat / np.sqrt(regions.shape[0])
    return Rhat + t - gamma

# Use optimization to find lambda_hat
def get_lambda_hat_clt_perpolyp_01(regions, masks, gamma, delta):
    def _condition_in(lam):
        Tlam = regions >= -lam
        Rhat, sigmahat = empirical_risk(Tlam, masks)
        t = -norm.ppf(delta) * sigmahat / np.sqrt(regions.shape[0])

        return Rhat + t - gamma

    return brentq(_condition_in, -0.01, -0.99, xtol=1e-3, rtol=1e-3)


if __name__ == '__main__':
    imgs = imread('cali.tif')

    #masks = array([tomask(s['coordinates'],dims) for s in regions])
    with open('label_masks_001.json') as f:
        masks = json.load(f)

    gamma = 0.1
    delta = 0.1
    lambdas = np.linspace(-0.99, -0.5, 20)
    emp_risks = []
    UBC_risks = []
    for lam in tqdm(lambdas):
        emp_risk, true_risk = risk_UBC(imgs, masks, gamma, delta, lam)
        emp_risks.append(emp_risk)
        UBC_risks.append(true_risk)

    # Sample data
    # lambdas = [0.1, 0.2, 0.3, 0.4, 0.5]  # Replace with your actual lambda values
    # emp_risks = [0.5, 0.4, 0.35, 0.33, 0.31]  # Replace with your actual empirical risks
    # UBC_risks = [0.6, 0.5, 0.45, 0.4, 0.35]  # Replace with your actual upper bound risks

    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, emp_risks, label='Empirical Risk', color='#444444', linewidth=3)
    plt.plot(lambdas, UBC_risks, label='Upper Bound Risk', color='#EBE1A4', linewidth=3)

    # Labeling the axes
    plt.xlabel('Lambda', fontsize=16)  # Adjust font size for x-axis label
    plt.ylabel('Risk', fontsize=16)  # Adjust font size for y-axis label

    # Setting the title with increased font size
    plt.title('Risk Across Lambda Values', fontsize=18)

    # Adding a grid for better readability
    plt.grid(False)

    # Adding a legend with increased font size
    plt.legend(fontsize=14)

    # Show the plot
    plt.show()


import numpy as np
from skimage.measure import compare_psnr as psnr_metric
from skimage.measure import compare_ssim as ssim_metric


def mse(gt, y_hat):
    mse = np.square(gt - y_hat).sum()
    return mse/10000

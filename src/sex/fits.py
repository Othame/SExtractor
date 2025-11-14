from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clip
import numpy as np
from photutils.aperture import CircularAperture, aperture_photometry
from sigmaex import SigmaEx

def find_valid_region(images):
    """
    查找三维图像中所有切片的有效图像区域（非 NaN 区域）共同的矩形区域
    """ 
    # 初始化全为True的mask，形状与单个图像一致
    valid_mask = np.ones_like(images[0], dtype=bool)

    # 累积有效区域的布尔掩码
    for image in images:
        valid_mask &= ~np.isnan(image)

    # 获取有效区域的边界
    non_zero_coords = np.argwhere(valid_mask)
    y_min, x_min = non_zero_coords.min(axis=0)
    y_max, x_max = non_zero_coords.max(axis=0) + 1

    return y_min, y_max, x_min, x_max

def crop_images_to_valid_region(images, valid_region):
    """
    根据有效区域的边界裁剪三维图像stack
    """
    y_min, y_max, x_min, x_max = valid_region
    return images[:, y_min:y_max, x_min:x_max]

def sigma_clipping_zaxis(image, sigma : float = 3.0):
    # 计算深度维度的均值和标准差
    with np.errstate(invalid='ignore'):
        mean = np.nanmean(image, axis = 0)
        std = np.nanstd(image, axis = 0)
        median = np.nanmedian(image, axis = 0, keepdims = True)
    median = np.broadcast_to(median, image.shape)
    
    # 计算 sigma clipping 的上下界
    lower_bound = mean - sigma * std
    upper_bound = mean + sigma * std
    
    # 创建逻辑掩码，标记需要替换的位置
    mask = (image <= lower_bound) | (image >= upper_bound) | np.isnan(image)

    # 使用逻辑掩码直接替换需要处理的元素
    image[mask] = median[mask]
    
    return image

def mse_select_bad_frame(stack):
    mean_image = np.nanmean(stack, axis = 0)

    # Initialize an array to store MSE values
    num_frames = stack.shape[0]
    mse_values = np.zeros(num_frames)

    # 3. Compute MSE for each frame compared to the mean image
    for i in range(num_frames):
        frame = stack[i]
        # Create a mask of valid pixels (not NaN in both frame and mean_image)
        valid_mask = ~np.isnan(frame) & ~np.isnan(mean_image)
        # Calculate the differences only on valid pixels
        diff = frame[valid_mask] - mean_image[valid_mask]
        mse = np.mean(diff ** 2)
        mse_values[i] = mse

    # Sort the frames based on MSE values in ascending order
    sorted_indices = np.argsort(mse_values)
    sorted_stack = stack[sorted_indices]

    # sorted_stack[np.isnan(sorted_stack)] = 0
    sorted_mse_values = mse_values[sorted_indices]
    
    return sorted_stack, sorted_mse_values

def get_sigma(data, ap_size = 2.5, n_rand_ap = 100000, clip_sigma = 3.0) -> float:
    sigma_clipped_data = sigma_clip(data, sigma=clip_sigma, maxiters=None)
    sigma_clipped_data_with_nan = np.ma.filled(sigma_clipped_data, np.nan)
    res = np.array([])
    while len(res) < n_rand_ap:
        pos = np.array((np.random.randint(ap_size, data.shape[1] - ap_size,n_rand_ap), 
                        np.random.randint(ap_size, data.shape[0] - ap_size,n_rand_ap))).T
        aps = CircularAperture(pos, r=ap_size)
        tmp_table = aperture_photometry(sigma_clipped_data_with_nan, aps)['aperture_sum']
        res = np.append(res,tmp_table[~np.isnan(tmp_table)])
    res = res[:n_rand_ap]
    sigmaex_obj = SigmaEx(res)
    return sigmaex_obj.gaussian_fit_sigma, sigmaex_obj

class Fits:
    def __init__(self, fits_path : str):
        self.fits_path = fits_path
        self.data = None
        self.header = None
        self.wcs = None
        self.load_fits()

    def load_fits(self):
        with fits.open(self.fits_path) as hdul:
            self.data = hdul[0].data
            self.header = hdul[0].header
            self.wcs = WCS(self.header)
    
    def get_sigma(self, ap_size = 2.5, n_rand_ap = 100000, clip_sigma = 3.0) -> float:
        sigma_clipped_data = sigma_clip(self.data, sigma=clip_sigma, maxiters=None)
        sigma_clipped_data_with_nan = np.ma.filled(sigma_clipped_data, np.nan)
        res = np.array([])
        while len(res) < n_rand_ap:
            pos = np.array((np.random.randint(ap_size, self.data.shape[1] - ap_size,n_rand_ap), 
                            np.random.randint(ap_size, self.data.shape[0] - ap_size,n_rand_ap))).T
            aps = CircularAperture(pos, r=ap_size)
            tmp_table = aperture_photometry(sigma_clipped_data_with_nan, aps)['aperture_sum']
            res = np.append(res,tmp_table[~np.isnan(tmp_table)])
        res = res[:n_rand_ap]
        return SigmaEx(res).gaussian_fit_sigma

    def write_to(self, output_path : str):
        hdu = fits.PrimaryHDU(self.data, header=self.header)
        hdu.writeto(output_path, overwrite=True)
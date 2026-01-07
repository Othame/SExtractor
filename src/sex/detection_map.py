import pandas as pd
from astropy.io import fits
from photutils.aperture import SkyCircularAperture, aperture_photometry
import astropy.units as u   
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import numpy as np
from regions import Regions, EllipseSkyRegion, CircleSkyRegion
from scipy.spatial import cKDTree
from typing import Callable, List
from tqdm import tqdm
from photutils.background import Background2D, MedianBackground
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAperture, CircularAnnulus

def convert_to_Ds9SkyReg(row):
    center_sky = SkyCoord(row['ALPHA_J2000'], row['DELTA_J2000'], unit='deg', frame='icrs')
    height = 2.5 * row['A_WORLD'] * 2 * u.deg
    width = 2.5 * row['B_WORLD'] * 2 * u.deg
    angle = row['THETA_J2000'] * u.deg
    region_sky = EllipseSkyRegion(center=center_sky,
                                  height=height, width=width,
                                  angle=angle)
    return region_sky

def convert_to_Ds9SkyReg_Aperture(row, ap_size : float = 2.5, pixel_size : float = 0.04, with_label : bool = True):
    center_sky = SkyCoord(row['ALPHA_J2000'], row['DELTA_J2000'], unit='deg', frame='icrs')
    radius = (ap_size * pixel_size) * u.arcsec
    region_sky = CircleSkyRegion(center=center_sky, radius=radius)
    if with_label and 'ABMAG_APER' in row:
        region_sky.meta['label'] = f"{row['ABMAG_APER']:.2f}"
    return region_sky


def write_regions_with_labels(regions: List[CircleSkyRegion], filename: str, with_label : bool):
    """
    regions: list of CircleSkyRegion
    filename: str
    """
    with open(filename, "w") as f:
        f.write("# Region file format: DS9 astropy/regions\n")
        f.write("icrs\n")
        for reg in regions:
            c = reg.center
            ra = c.ra.deg
            dec = c.dec.deg
            radius_deg = reg.radius.to(u.deg).value
            label = reg.meta.get('label', None)
            if label is not None and with_label:
                f.write(f"circle({ra:.8f},{dec:.8f},{radius_deg:.8e}) # text={{{label}}}\n")
            else:
                f.write(f"circle({ra:.8f},{dec:.8f},{radius_deg:.8e})\n")

class DetectionMap(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @property
    def _constructor(self):
        return DetectionMap
    
    @staticmethod
    def from_csv(csv_path : str) -> "DetectionMap":
        df = pd.read_csv(csv_path, index_col=0)
        return DetectionMap(df)
    
    def apply_mask(self, 
                     mask_path : str,
                     ap_size : float = 2.5,
                     pixel_size : float = 0.04,
                     drop_masked = True,
                     mask_trans_func : Callable = None):
        # 读取mask fits文件
        with fits.open(mask_path) as hdul:
            mask_data = hdul[0].data
            mask_bool = mask_trans_func(mask_data) if mask_trans_func is not None else mask_data
            wcs = WCS(hdul[0].header)
        
        # 一次性构造所有源的天球坐标
        ra = np.asarray(self['ALPHA_J2000'].values, dtype=float)
        dec = np.asarray(self['DELTA_J2000'].values, dtype=float)
        positions = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')

        # 构造一个 Sky aperture（半径单位用角秒；ap_size(像素) * pixel_size(arcsec/pix)）
        r = (ap_size * pixel_size) * u.arcsec
        aperture = SkyCircularAperture(positions, r=r)

        # 一次性做 aperture photometry（vectorized）
        phot = aperture_photometry(mask_bool, aperture, wcs=wcs)
        # 取出每个源 aperture 内的 mask 求和（布尔 True 会当 1 计数）
        mask_sum = np.asarray(phot['aperture_sum'], dtype=float)

        if drop_masked:
            # 过滤：aperture 内没有被 mask 到（和为 0）的源
            masked = self[mask_sum == 0].reset_index(drop=True)
        else:
            # 不过滤，只标记被mask的源
            masked = self.copy()
            masked['MASKED'] = (mask_sum != 0)
        return masked
    
    def compute_photometry(self, fits_path : str, 
                    zero_point : float, 
                    psf_factor : float = 0.7583614,
                    ap_size : float = 2.5,
                    pixel_size : float = 0.04,
                    max_ABmag : float = 33.0,
                    drop_faint : bool = True,
                    bkg_sub : bool = False,
                    bkg_box_size : int = 32,
                    bkg_filter_size : int = 3,
                    verbose : bool = False):
        '''
        给定fits_path, 计算每个源的photometry
        '''
        if self.empty:
            print("Warning: No sources found in the detection map")
            return self
        self = self.drop(columns=['NJY_APER', 'ABMAG_APER'], errors='ignore')
        
        with fits.open(fits_path) as hdul:
            data = hdul[0].data
            wcs = WCS(hdul[0].header)

        if bkg_sub:
            bkg_estimator = MedianBackground()
            bkg = Background2D(data,
                            box_size=(bkg_box_size, bkg_box_size),
                            filter_size=(bkg_filter_size, bkg_filter_size),
                            bkg_estimator=bkg_estimator)
            data = data - bkg.background

        # 批量构造天球坐标
        ra  = np.asarray(self['ALPHA_J2000'].values, dtype=float)
        dec = np.asarray(self['DELTA_J2000'].values, dtype=float)
        coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')

        # 构造 Sky aperture（ap_size 为像素半径；pixel_size 为 arcsec/pixel）
        r = (ap_size * pixel_size) * u.arcsec
        aperture = SkyCircularAperture(coords, r=r)

        # 批量做 aperture photometry（自动做 WCS 投影）
        phot_tbl = aperture_photometry(data, aperture, wcs=wcs)
        # aperture_sum 是每个源的总和（单位按 data 而定，这里当作“计数/流量”）
        flux_sum = np.asarray(phot_tbl['aperture_sum'], dtype=float) / float(psf_factor)

        # 转换到 nJy 与 ABmag（注意非正值不能转为 ABmag）
        zp_njy = (zero_point * u.ABmag).to(u.nJy).value  # 标量
        flux_njy = flux_sum * zp_njy                     # (N,)

        mag = np.full_like(flux_njy, 99.0, dtype=float)
        pos = flux_njy > 0
        if np.any(pos):
            mag[pos] = (flux_njy[pos] * u.nJy).to(u.ABmag).value

        # 拼回 DataFrame
        out = self.copy()
        out['NJY_APER']  = flux_njy
        out['ABMAG_APER'] = mag

        if verbose:
            print(out)
        if drop_faint:
            # 过滤太暗源
            out = out[out['ABMAG_APER'] < max_ABmag]
        else:
            # 不过滤，只标记太暗的源
            out = out.copy()
            out['FAINT'] = (out['ABMAG_APER'] >= max_ABmag)
        return out
    
    def compute_photometry_bkg2d_annulus(
        self, 
        fits_path: str,
        zero_point: float,
        psf_factor: float = 1.0,

        # aperture / annulus in pixels
        ap_size: float = 2.5,
        ann_r_in: float = 6.0,
        ann_r_out: float = 10.0,

        # Background2D params
        use_bkg2d: bool = True,
        bkg_box_size: int = 32,
        bkg_filter_size: int = 3,
        bkg_mask_bool: np.ndarray | None = None,   # (H,W) True=masked (可选)

        # speed/robustness
        method: str = "center",     # "center" 快；"exact" 更精确但慢
        batch_size: int = 5000,
        sigma_clip: float = 3.0,    # annulus局部背景的sigma-clip
        drop_faint: bool = True,
        max_ABmag: float = 33.0,
        verbose: bool = False,
    ):
        """
        对 df (含 ALPHA_J2000/DELTA_J2000) 做：
        1) (可选) Background2D 扣大尺度背景
        2) 每源 annulus sigma-clipped median 做局部背景
        3) aperture flux 扣掉局部背景，再除 psf_factor，转 nJy/ABmag
        """

        if self.empty:
            if verbose:
                print("Warning: empty catalog")
            return self.copy()

        with fits.open(fits_path) as hdul:
            data = hdul[0].data.astype(np.float32)
            wcs = WCS(hdul[0].header)

        H, W = data.shape

        # --- 0) WCS -> pixel positions (一次性) ---
        ra = np.asarray(self["ALPHA_J2000"].values, float)
        dec = np.asarray(self["DELTA_J2000"].values, float)
        coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        x, y = wcs.world_to_pixel(coords)  # float arrays

        positions = np.vstack([x, y]).T
        out = self.copy()

        # --- 2) Background2D（整图一次）---
        if use_bkg2d:
            bkg_estimator = MedianBackground()
            bkg = Background2D(
                data,
                box_size=(bkg_box_size, bkg_box_size),
                filter_size=(bkg_filter_size, bkg_filter_size),
                bkg_estimator=bkg_estimator,
                mask=bkg_mask_bool
            )
            img = data - bkg.background
        else:
            img = data

        # --- 3) 批量 aperture + annulus 测光 ---
        ap_area = np.pi * (ap_size ** 2)
        zp_njy = (zero_point * u.ABmag).to(u.nJy).value

        flux_out = np.full(len(out), -np.inf, dtype=float)
        mag_out = np.full(len(out), -np.inf, dtype=float)
        ann_success = np.ones(len(out), dtype=np.int8)

        for i0 in range(0, len(out), batch_size):
            i1 = min(i0 + batch_size, len(out))
            pos_b = positions[i0:i1]

            aper = CircularAperture(pos_b, r=ap_size)
            ann = CircularAnnulus(pos_b, r_in=ann_r_in, r_out=ann_r_out)

            # aperture sum
            tab_ap = aperture_photometry(img, aper, method=method)
            ap_sum = np.asarray(tab_ap["aperture_sum"], float)

            # annulus 像素样本：逐源做 sigma-clipped median（这里是主要耗时点）
            # 但 128x128 + batch 化后通常还可以接受；做 purity 初测光够用
            bkg_med = np.zeros(i1 - i0, dtype=float)
            ann_success_batch = np.ones(i1 - i0, dtype=np.int8)

            ann_masks = ann.to_mask(method=method)  # list of masks

            for k, m in enumerate(ann_masks):
                cut = m.cutout(img)  # annulus 的 bounding box cutout
                if cut is None:
                    ann_success_batch[k] = 0
                    continue

                ann_data = m.multiply(img)  # same shape as cutout region (outside annulus -> 0)
                sel = m.data > 0  # annulus pixels
                vals = ann_data[sel]
                vals = vals[np.isfinite(vals)]
                if vals.size < 20:
                    ann_success_batch[k] = 0
                    continue

                # sigma-clipped median as local background
                _, med, _ = sigma_clipped_stats(vals, sigma=sigma_clip)
                bkg_med[k] = med

            # local background subtraction
            flux = ap_sum - bkg_med * ap_area

            # PSF / aperture correction factor
            flux = flux / float(psf_factor)

            # convert to nJy and AB mag
            flux_njy = flux * zp_njy
            ok = np.isfinite(flux_njy) & (flux_njy > 0)

            mag = np.full_like(flux_njy, -np.inf, dtype=float)
            mag[ok] = (flux_njy[ok] * u.nJy).to(u.ABmag).value

            flux_out[i0:i1] = flux_njy
            mag_out[i0:i1] = mag
            ann_success[i0:i1] = ann_success_batch

        # 回填结果给 good_pos对应的index
        out["NJY_APER"] = flux_out
        out["ABMAG_APER"] = mag_out
        out["ANNULUS"] = ann_success == 1

        if drop_faint:
            out = out[np.isfinite(out["ABMAG_APER"]) & (out["ABMAG_APER"] < max_ABmag)]
        else:
            out = out.copy()
            out['FAINT'] = (out['ABMAG_APER'] >= max_ABmag) | np.isinf(out['ABMAG_APER'])
        if verbose:
            print(out)

        return out
    def filter_marginal_sources(self, 
                                 margin_size_pixel : float = 10,
                                 image_size_pixel : tuple = (128, 128),
                                 drop_marginal = True):
        '''
        过滤det_map中的边缘源
        '''
        x_image = self["X_IMAGE"] - 1
        y_image = self["Y_IMAGE"] - 1
        mask = (x_image < margin_size_pixel) | (x_image > image_size_pixel[1] - margin_size_pixel) | (y_image < margin_size_pixel) | (y_image > image_size_pixel[0] - margin_size_pixel)
        if drop_marginal:
            filtered = self[mask].reset_index(drop=True)
        else:
            filtered = self.copy()
            filtered['MARGINAL'] = mask
        return filtered
    def filter_overlap_sources(self):
        '''
        过滤det_map中的星
        '''
        ref_coords = np.vstack([self["X_IMAGE"], self["Y_IMAGE"]]).T
        ref_tree = cKDTree(ref_coords)
        dist_threshold = self['A_IMAGE'].max()
        n_sources = len(self)
        mask = np.ones(n_sources, dtype=bool)
        
        for i in tqdm(range(n_sources)):
            if mask[i] == False:
                continue
            
            close_idxs = ref_tree.query_ball_point(ref_coords[i], r=dist_threshold)
            
            if len(close_idxs) <= 1:
                continue

            # 获取当前源的椭圆参数
            x0, y0 = self.iloc[i]["X_IMAGE"], self.iloc[i]["Y_IMAGE"]
            a, b = self.iloc[i]["A_IMAGE"], self.iloc[i]["B_IMAGE"]
            theta = self.iloc[i]["THETA_IMAGE"]
            theta_rad = np.deg2rad(theta)

            # 获取所有邻居坐标（排除自身 i）
            j_idxs = [j for j in close_idxs if j != i]
            xy = ref_coords[j_idxs]  # shape (N, 2)
            dx = xy[:, 0] - x0
            dy = xy[:, 1] - y0

            # 坐标投影（旋转到椭圆主轴系）
            xp = dx * np.cos(theta_rad) + dy * np.sin(theta_rad)
            yp = -dx * np.sin(theta_rad) + dy * np.cos(theta_rad)

            # 归一化椭圆判定
            inside = (xp / a) ** 2 + (yp / b) ** 2 <= 1

            if np.any(inside):
                mask[i] = False
        self = self[mask].reset_index(drop=True)
        return self
    
    def to_Ds9SkyReg(self, reg_path : str):
        reg_list = []
        for _, row in self.iterrows():
            region = convert_to_Ds9SkyReg(row)
            reg_list.append(region)
        reg = Regions(reg_list)
        reg.write(reg_path, format='ds9', overwrite=True)
        
    def to_Ds9SkyReg_Aperture(self, reg_path : str, with_label : bool = True, ap_size : float = 2.5, pixel_size : float = 0.04):
        reg_list = []
        for _, row in self.iterrows():
            region = convert_to_Ds9SkyReg_Aperture(row, ap_size, pixel_size, with_label)
            reg_list.append(region)
        write_regions_with_labels(reg_list, reg_path, with_label)
    
    @staticmethod
    def from_Ds9SkyReg_Aperture(reg_path : str):
        reg_list = Regions.read(reg_path, format='ds9')
        import re
        source_list = []
        for reg in reg_list:
            # 尝试从reg.meta['text']或reg.meta['label']中读取abmag
            abmag = None
            text = reg.meta.get('text', None)
            if text is not None:
                # 匹配如 {29.79} 或 29.79
                m = re.search(r'([0-9]+\.[0-9]+)', str(text))
                if m:
                    abmag = float(m.group(1))
            source_list.append({
                'ALPHA_J2000': reg.center.ra.deg, 
                'DELTA_J2000': reg.center.dec.deg,
                'ABMAG_APER': abmag
            })
        return DetectionMap(pd.DataFrame(source_list))
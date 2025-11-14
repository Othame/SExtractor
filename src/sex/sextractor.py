import os
from typing import List, Dict, Tuple
import tifffile as tiff
from astropy.io import fits
from regions import Regions, PixCoord, EllipsePixelRegion, EllipseSkyRegion
from tqdm import tqdm
import astropy.units as u
from astropy.table import Table
import subprocess
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from astropy.coordinates import SkyCoord
from .detection_map import DetectionMap
import yaml

JWST_ZP = 27.461825709242483
SUBARU_ZP = 27.11087032

def convert_to_ds9reg_pixel(row):
    center_pix = PixCoord(row['X_IMAGE']-1, row['Y_IMAGE']-1)
    height = 2.5 * row['A_IMAGE'] * 2
    width = 2.5 * row['B_IMAGE'] * 2
    angle = (90 + row['THETA_IMAGE']) * u.deg
    region_pix = EllipsePixelRegion(center=center_pix,
                            height=height, width=width,
                            angle=angle)
    return region_pix

class SExtractor:
    def __init__(self, 
                 sex_path : str = r"/usr/local/bin/sex"):
        
        self.sex_path : str = sex_path
        self.cat_name : str = "source.cat"
        self.ds9reg_name : str = "source-detection.reg"
        self.sub_dir_names : Dict[str, str] = {
            "config" : "config",
            "catalog" : "catalog",
            "check" : "check"
        }
    
    def run(self,
            config_path : str,
            work_dir : str,
            fits_path : str,
            zero_point: float = SUBARU_ZP,
            detect_thres : float = 0.5,
            detect_minarea : int = 15,
            to_ds9reg : bool = True):

        sub_dirs = self.prepare_work_sub_dirs(work_dir=work_dir)
        self.write_config(config_path=config_path,config_dir=sub_dirs["config"])
        self.run_sex(fits_path=fits_path, sub_dirs=sub_dirs, zero_point=zero_point, detect_thres=detect_thres, detect_minarea=detect_minarea)
        if to_ds9reg:
            self.convert_cat_to_ds9reg(catalog_dir=sub_dirs["catalog"])

    def prepare_work_sub_dirs(self, work_dir : str) -> Dict[str, str]:
        '''
        准备工作目录
        '''
        if not os.path.exists(work_dir):
            print(f"Directory '{work_dir}' does not exist. Creating it...")
        os.makedirs(work_dir, exist_ok=True)
        
        sub_dirs = {}
        
        for key, value in self.sub_dir_names.items():
            sub_dirs[key] = os.path.join(work_dir, value)
            os.makedirs(sub_dirs[key], exist_ok=True)
        
        return sub_dirs
    
    def write_config(self, config_path : str, config_dir : str):
        '''
        将config中的内容写入config_dir中
        '''
        with open(config_path, 'r', encoding='utf-8') as f:
            work_config = yaml.safe_load(f)
        for filename, content in work_config.items():
            file_path = os.path.join(config_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
    
    def run_sex(self, fits_path: str, sub_dirs : Dict[str, str], zero_point : float, detect_thres : float, detect_minarea : int):
        '''
        给定一个fits, 运行sex, 生成对应的cat
        sex的工作目录就是config_dir, file_name另外指定, 输出路径由cat_path指定
        '''
        
        cat_path : str =  os.path.join(sub_dirs["catalog"], self.cat_name)
        background_path : str = os.path.join(sub_dirs["check"], "background.fits")
        bkg_sub_path : str = os.path.join(sub_dirs["check"], "bkg_sub.fits")
        segmentation_path : str = os.path.join(sub_dirs["check"], "segmentation.fits")
        
        cmd = [
            self.sex_path, fits_path,
            "-DETECT_THRESH", str(detect_thres),
            "-DETECT_MINAREA", str(detect_minarea),
            "-ANALYSIS_THRESH", str(detect_thres),
            "-CATALOG_NAME", cat_path,
            "-MAG_ZEROPOINT", str(zero_point),
            "-CHECKIMAGE_NAME", f"{background_path},{bkg_sub_path},{segmentation_path}"
        ]
        
        subprocess.run(
            cmd,
            cwd=sub_dirs["config"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
    
    def convert_cat_to_ds9reg(self, catalog_dir : str):
        '''
        将catalog_dir中的sources.cat转换为ds9reg文件
        '''
        cat_path : str = os.path.join(catalog_dir, self.cat_name)
        cat = Table.read(cat_path, format="ascii.sextractor")
        reg_path : str = os.path.join(catalog_dir, self.ds9reg_name)
        reg_list = []
        for _, row in cat.iterrows():
            region_pix = convert_to_ds9reg_pixel(row)
            reg_list.append(region_pix)
        reg = Regions(reg_list)
        reg.write(reg_path, overwrite=True)
        
class MultiThresholdSExtractor:
    def __init__(self, 
                 sex_path : str = r"/usr/local/bin/sex",
                 max_workers : int = 100):
        
        self.sex = SExtractor(sex_path=sex_path)
        self.max_workers = max_workers
    
    def run(self,
            config_path : str,
            work_dir : str,
            fits_path : str,
            thresholds : List[float],
            dist_threshold : float = 3.0,
            zero_point: float = SUBARU_ZP,
            minarea : int = 15,
            verbose : bool = False,
            ) -> DetectionMap:
        
        sub_work_dirs = self.run_sex(config_path=config_path, work_dir=work_dir, fits_path=fits_path, thresholds=thresholds, zero_point=zero_point, minarea=minarea)
        csv_files = self.convert_cat_to_csv(sub_work_dirs)
        det_map = self.match_and_expand_catalog(work_dir=work_dir, csv_files=csv_files, thresholds=thresholds, dist_threshold=dist_threshold)
        if verbose:
            print(f"Filtering overlap sources... #Sources: {len(det_map)}")
        det_map = det_map.filter_overlap_sources()
        if verbose:
            print(f"Detection Finished #Sources: {len(det_map)}")
        return det_map
    
    def run_sex(self, 
            config_path : str,
            work_dir : str,
            fits_path : str,
            thresholds : List[float],
            zero_point: float,
            minarea : int) -> List[str]:
  
        def run_with_threshold(threshold):
            sub_work_dir = os.path.join(work_dir, f"thres_{threshold}")
            os.makedirs(sub_work_dir, exist_ok=True)
            self.sex.run(
                config_path=config_path,
                work_dir=sub_work_dir,
                fits_path=fits_path,
                zero_point=zero_point,
                detect_thres=threshold,
                detect_minarea=minarea,
                to_ds9reg=False
            )
            return (threshold, sub_work_dir)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            sub_work_dirs = list(tqdm(executor.map(run_with_threshold, thresholds), total=len(thresholds)))
        
        return {thres: dir for thres, dir in sub_work_dirs}
    
    def convert_cat_to_csv(self, sub_work_dirs : List[Tuple[float, str]]):
        '''
        将sub_work_dirs中的cat文件转换为csv文件
        '''
        csv_files = {}
        for threshold, work_dir in sub_work_dirs.items():
            cat_path = os.path.join(work_dir, self.sex.sub_dir_names["catalog"], self.sex.cat_name)
            csv_path = cat_path.replace(".cat", ".csv")
            cat = Table.read(cat_path, format="ascii.sextractor")
            cat["THRES"] = threshold
            cat.write(csv_path, format="ascii.csv", overwrite=True)
            csv_files[threshold] = csv_path
        return csv_files
    
    def match_and_expand_catalog(self, work_dir : str, csv_files : Dict[float, str], thresholds : List[float], dist_threshold : float):
        '''
        从最大的threshold开始，依次扩充csv_files中的csv文件
        '''
        thresholds_sorted = sorted(thresholds, reverse=True)
        # 读取最大threshold作为参考星表
        det_map = pd.read_csv(csv_files[thresholds_sorted[0]])
        
        # 遍历Thresholds
        for threshold in thresholds_sorted[1:]:
            # 读取提取星表
            ext_df = pd.read_csv(csv_files[threshold])
            ext_coords = np.vstack([ext_df["X_IMAGE"], ext_df["Y_IMAGE"]]).T

            # 构建参考星表的KDTree
            ref_coords = np.vstack([det_map["X_IMAGE"], det_map["Y_IMAGE"]]).T
            ref_tree = cKDTree(ref_coords)

            # 查询每个提取星到参考星表的最近距离
            dists, _ = ref_tree.query(ext_coords, k=1)

            # 选出距离大于阈值的星
            new_sources = ext_df[dists > dist_threshold]

            # 将新找到的星加入参考星表
            det_map = pd.concat([det_map, new_sources], ignore_index=True)

        # 对NUMBER列重新编号
        det_map.drop(columns=['NUMBER'], inplace=True)

        return DetectionMap(det_map)
        
    
    
    
from scipy.spatial import cKDTree
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from .detection_map import DetectionMap

def cross_match(ref_path, ext_path, threshold_arcsec=0.04, verbose=False):
    if isinstance(ref_path, DetectionMap):
        ref_df = ref_path
    else:
        ref_df = DetectionMap.from_csv(ref_path)
    if isinstance(ext_path, DetectionMap):
        ext_df = ext_path
    else:
        ext_df = DetectionMap.from_csv(ext_path)

    ref_coords = SkyCoord(ra=ref_df['ALPHA_J2000'].values * u.deg, dec=ref_df['DELTA_J2000'].values * u.deg, frame='icrs')
    ext_coords = SkyCoord(ra=ext_df['ALPHA_J2000'].values * u.deg, dec=ext_df['DELTA_J2000'].values * u.deg, frame='icrs')

    ref_xyz = ref_coords.cartesian.xyz.value.T
    ext_xyz = ext_coords.cartesian.xyz.value.T

    # 构建KDTree
    tree = cKDTree(ref_xyz)

    # 这里也用SkyCoord进行转换
    threshold = threshold_arcsec * u.arcsec
    threshold_rad = threshold.to(u.rad).value  # 转换为 float 型弧度值

    # 查询匹配
    dists, ref_indices = tree.query(ext_xyz, k=1, distance_upper_bound=threshold_rad)

    # 记录匹配成功的df1和df2的index
    matched_indices = {}

    for ext_idx, (ref_idx, dist) in enumerate(zip(ref_indices, dists)):
        if ref_idx < len(ref_df) and dist < threshold_rad:
            if ref_idx not in matched_indices:
                matched_indices[ref_idx] = (ext_idx, dist)
            else:
                if dist < matched_indices[ref_idx][1]:
                    matched_indices[ref_idx] = (ext_idx, dist)

    matched_ref_indices = list(matched_indices.keys())
    matched_ext_indices = [matched_indices[ref_idx][0] for ref_idx in matched_ref_indices]

    # matched_df1_idx为df1中匹配成功的index，matched_df2_idx为对应df2中匹配的index
    if verbose:
        print(f"matched: {len(matched_ref_indices)}")
        print(f"ref: {len(ref_df)}")
        print(f"ext: {len(ext_df)}")
    
    # 从df1和df2中用匹配到的index提取数据
    matched_ref_df = ref_df.iloc[matched_ref_indices].copy()
    matched_ext_df = ext_df.iloc[matched_ext_indices].copy()

    matched_ref_df['matched_idx'] = list(matched_ext_df.index)
    matched_ext_df['matched_idx'] = list(matched_ref_df.index)
    
    # 提取unmatched部分
    all_ref_indices = set(range(len(ref_df)))
    all_ext_indices = set(range(len(ext_df)))
    matched_ref_set = set(matched_ref_indices)
    matched_ext_set = set(matched_ext_indices)

    unmatched_ref_indices = list(all_ref_indices - matched_ref_set)
    unmatched_ext_indices = list(all_ext_indices - matched_ext_set)

    unmatched_ref_df = ref_df.iloc[unmatched_ref_indices].copy()
    unmatched_ext_df = ext_df.iloc[unmatched_ext_indices].copy()

    return DetectionMap(matched_ref_df), DetectionMap(matched_ext_df), DetectionMap(unmatched_ref_df), DetectionMap(unmatched_ext_df)





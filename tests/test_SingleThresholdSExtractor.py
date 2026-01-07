from sex.sextractor import SingleThresholdSExtractor
from astropy.io import fits
from astropy.wcs import WCS
test_fits_path = r"/data1/zc/Subaru/calexp/9813/HSC-I/fits/calexp-HSC-I-9813-4,4.fits"
with fits.open(test_fits_path) as hdul:
    wcs = WCS(hdul[1].header)
    data = hdul[1].data
recover_fits_path = r"/data1/zc/Subaru/sex/test/src/test.fits"
fits.writeto(recover_fits_path, data, wcs.to_header(), overwrite=True)
sex = SingleThresholdSExtractor()
det_map = sex.run(
    config_path = r"/home/zc/workspace/Validation/sex/configs/origin_no_bkg.yaml",
    work_dir=r"/data1/zc/Subaru/sex/test",
    fits_path=recover_fits_path,
    threshold=1.2,
    zero_point=27.0,
    minarea=15,
    verbose=True,
)
det_map.to_csv(r"/data1/zc/Subaru/sex/test/source.csv")
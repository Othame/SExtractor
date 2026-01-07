class HSCParams:
    pixel_size = 0.168 # arcsec
    seeing = 0.7 # arcsec
    psf_factor = 0.8 # 80% encircled energy radius
    photometry_aperture = seeing * 0.7618937 / pixel_size # pixels, 80% encircled energy radius r_80 = 1/2 * FWHM * sqrt(ln5 / ln2)
    match_threshold = 0.3 # arcsec
    zero_point = 32.5 # AB mag
    detection_minarea = 15 # pixels
    detection_thresholds = [55, 50, 48, 45, 43, 40, 38, 35, 33, 30, 28, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7.5, 7, 6.5, 6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1, 0.5]
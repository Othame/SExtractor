class HSCParams:
    pixel_size = 0.168 # arcsec
    seeing = 0.7 # arcsec
    psf_factor = 0.8 # 80% encircled energy radius
    photometry_aperture = seeing * 0.7618937 / pixel_size # pixels, 80% encircled energy radius r_80 = 1/2 * FWHM * sqrt(ln5 / ln2)
    match_threshold = 0.3 # arcsec
    zero_point = 32.5 # AB mag
    detection_minarea = 15 # pixels
import awb

NUMBER_OF_SAMPLES = 100 # use 1000 if not using svd.
WAVELENGTH_RANGE = awb.SpectralFunction.lambdas #maybe add another way to set wavelength resolution
COLOR_EPSILON = 1e-4
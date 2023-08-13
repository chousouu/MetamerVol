import awb

NUMBER_OF_SAMPLES = 1000 #use 1000 with svd optimization, without ---  approx. 3000 
WAVELENGTH_RANGE = awb.SpectralFunction.lambdas #add another way to set wavelength resolution
COLOR_EPSILON = 1e-4
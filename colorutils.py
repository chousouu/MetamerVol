import numpy as np
from constants import COLOR_EPSILON

def get_unique_colors(image):
    """
    receives 2D (width * length, 3) or 3D (length, width, 3) image array 
    and returns unique colors of the image.
    """
    image = np.reshape(image, (-1, 3))
    return np.unique(np.floor(image / COLOR_EPSILON), axis = 0) * COLOR_EPSILON

def get_colour_sys(illum, sens):
    return np.stack([illum * sens for sens in sens.T], axis = 1)    

def get_colour_response(sensitivies, illum, reflectances, lambdas):
    """
    returns colour response 

    Parameters
    ----------
    sensitivities : (3, q) ndarray
    illum : (q,) ndarray
    reflectances: (q,) ndarray 

    q - wavelength resolution. For example, q = int(780 - 350) + 1 

    Returns
    -------
    k : ndarray
        returns size 3 colour response array 
    """
    tristims = list()
    for sens in sensitivies:
        colour_component = np.trapz(y = sens * (illum * reflectances), x = lambdas) 
        tristims.append(colour_component)

    return np.stack(tristims, axis = 0)

def rgb_to_string(rgb):
    return '_'.join([(str(format(_, '.4f'))) for _ in rgb])

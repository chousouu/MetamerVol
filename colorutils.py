import numpy as np
from constants import COLOR_EPSILON

def get_unique_colors(image):
    """
    receives 2D (width * length, 3) or 3D (length, width, 3) image array 
    and returns unique colors of the image.
    """
    def is_equal(color_one, color_two):
        return np.all(abs(color_two - color_one) < COLOR_EPSILON)
    def is_unique(color, new_colors):
        for color in new_colors:
            if is_equal(rgb_pixel, color):
                return False
        return True
    
    image = np.reshape(image, (-1, 3))

    new_colors = []
    for rgb_pixel in image:
        if is_unique(rgb_pixel, new_colors):
            new_colors.append(rgb_pixel)

    return np.array(new_colors)

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

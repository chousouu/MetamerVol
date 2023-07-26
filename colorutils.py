import numpy as np
from constants import COLOR_EPSILON

def clear_array_from_zeros(arr : np.ndarray):
    if np.all(arr == 0):
        raise ValueError("Fully zeroed array!")

    return arr[arr.any(axis = 1)]

def find_non_zero_elem(arr : np.ndarray):
    """
    works on 1d only
    """
    return np.nonzero(arr)[0][0]


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
    illum : q-array
    reflectances: q-array
    sensitivities : 3 x q array
    """
    tristims = list()
    for sens in sensitivies:
        colour_component = np.trapz(y = sens * (illum * reflectances), x = lambdas) 

        tristims.append(colour_component)

    return np.stack(tristims, axis = 0)

def rgb_to_string(rgb):
    return '_'.join([(str(format(_, '.4f'))) for _ in rgb])

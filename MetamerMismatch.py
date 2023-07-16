import numpy as np
import scipy 
import pygel3d.gl_display as gd
from pygel3d import hmesh as hm 
from scipy.spatial import ConvexHull

NUMBER_OF_SAMPLES = 1000
WAVELENGTH_RESOLUTION = 10
COLOR_EPSILON = 0.001

# GENERAL TODO'S (important -> less important):
# 1.PEP8 style guide check
# 2.Make descriptions to funcs
# 3.do we norm r(lambda) ????????????????????UPD:where??
# 4.maybe merge together ocs and mmb funcs? (get_figure_volume/points /// get_figure_s_t[subject to])


def sample_unit_sphere(color_sys, wavelengths, sample_amount):
    """
    # returns k that belongs to R^(2N) for mmb and R^N for ocs
    """
    vector_size = color_sys.shape[1]
    wavelengths_size = np.shape(wavelengths)[0]
    k_sampled = []    
    wave = 0

    for _ in range(sample_amount):
        rand_components = np.random.randn(vector_size - 1, 1)
        k_1 = np.sum(np.multiply(rand_components.T, color_sys[wave, 1:])) / color_sys[wave, 0]
        k_i = np.append(k_1, rand_components)
        k_sampled += (k_i / np.linalg.norm(k_i)).tolist()
        wave = (wave + 1) % wavelengths_size

    return np.reshape(np.array(k_sampled), (vector_size, sample_amount)).T

def solve_linear_problem(objective_func_coef, constrain_func = None, 
                         constrain_func_value = None, bounds = None):
    '''
    ----------
    returns solution vector pair : (x_max, x_min)
    '''
    
    #x = r(lambda)
    #c = Sk  = objective_func_coef
    #S(lambda) = illumination * spectral.sens = colour sys. spectra. size : (,) 
    max_Class = scipy.optimize.linprog(c = objective_func_coef, 
                                   A_eq = constrain_func, 
                                   b_eq = constrain_func_value, 
                                   bounds = bounds)

    min_Class = scipy.optimize.linprog(c = objective_func_coef * -1, 
                                       A_eq = constrain_func, 
                                       b_eq = constrain_func_value, 
                                       bounds = bounds)
    
    if max_Class.status and min_Class.status:
        raise ValueError(f"optimize.lingprog failed to compute with reason :{max_Class.status}.",
                         "Check the reason on scipy's docs")

    # check if x_min = -x_max when one = c = ... *-1 and c = ...
    # as addition : k -> -k is also the solve so c = .... * -1 is alright ig
    x_max = max_Class.x
    x_min = min_Class.x

    return (x_max, x_min)

def get_mmb_points(metameric_color,  #<- metameric colors phi(r) in phi-space
               illum_phi, sens_phi,
               illum_psi, sens_psi,
               sampling_resolution = NUMBER_OF_SAMPLES):

#TODO: HARDCODED NEED TO BE FIXED! (maybe refactor functions)
    wavelengths = np.arange(400, 701, WAVELENGTH_RESOLUTION) 
    S_phi = get_colour_sys(illum_phi, sens_phi)
    S_psi = get_colour_sys(illum_psi, sens_psi) # q x N, q - wavelength resolution
    
    S = np.concatenate((S_phi, S_psi), axis = 1) #q x 2N; q - wavelength resolution

    mmb_extr_points = []

    for k in sample_unit_sphere(S, wavelengths, sampling_resolution):
        max_reflectance, min_reflectance =\
        solve_linear_problem(objective_func_coef = np.dot(S, k), constrain_func = S_phi.T ,
                             constrain_func_value = metameric_color, bounds = (0, 1))

        # scale so the brightest illum color response == 1          
        scale = np.max(np.dot(sens_psi.T, illum_psi))

        max_color_psi = np.dot(max_reflectance, S_psi) / scale
        min_color_psi = np.dot(min_reflectance, S_psi) / scale

        mmb_extr_points.extend([min_color_psi, max_color_psi])

    return mmb_extr_points

def get_ocs_points(illum, sens, sampling_resolution = NUMBER_OF_SAMPLES):
    S = get_colour_sys(illum, sens)

#TODO: HARDCODED NEED TO BE FIXED! (maybe refactor functions)
    wavelengths = np.arange(400, 701, WAVELENGTH_RESOLUTION) 

    ocs_extr_points = []

    for k in sample_unit_sphere(S, wavelengths, sampling_resolution):
        max_reflectance, min_reflectance =\
        solve_linear_problem(objective_func_coef = np.dot(S, k), bounds = (0,1))

        # scale so the brightest illum color response == 1  
        scale = np.max(np.dot(sens.T, illum))
        
        max_color = np.dot(max_reflectance, S) / scale
        min_color = np.dot(min_reflectance, S) / scale

        ocs_extr_points.extend([min_color, max_color])

    return ocs_extr_points

# def get_hulls_dict(colors, 
#                         illum_phi, sens_phi,
#                         illum_psi, sens_psi):
    
#     """
#     colors either a single [r,g,b] (1D)array or [length x width x rgb] (3D) array
#     returns dict {color : convexhull}
#     """

#     for color in colors:
#     hulls.update({rgb_to_string(color) : \
#                   ConvexHull(mm.get_mmb_points(metameric_color = color, 
#                    illum_phi = D65_Illuminant, illum_psi = D65_Illuminant,
#                    sens_phi = canon_sens, sens_psi = cie1964))} )

#---------------------------------------------------------------------------------------------------------------
#TODO: find a way to remove degenerated triangles without transfering convexhull->manifold like def dist() 
def _distance(points, point):
    m = hm.Manifold()
    for i in range(0, len(points), 3):
        m.add_face(points[i : i + 3])

    dist = hm.MeshDistance(m)
    distr = dist.signed_distance(point)
    return distr
#---------------------------------------------------------------------------------------------------------------

def hull_from_points(surface_points):
	#maybe add raisevalue if points are added as manifold class
	return ConvexHull(surface_points)

def _view_Manifold(m):
    gd.Viewer().display(m)
    return

def dist(hull, point): #temp(?) version of func
    m = hm.Manifold()

    for s in hull.simplices:
        m.add_face(hull.points[s])

    dist = hm.MeshDistance(m).signed_distance(point)

    return dist

def get_unique_colors(image):
    """
    receives 2D (width * length, 3) or 3D (length, width, 3) image array 
    and returns unique colors of the image.
    """
    def is_equal(color_one, color_two):
        return all(abs(color_two - color_one) < COLOR_EPSILON)
    def is_unique(color, new_colors):
        for color in new_colors:
            if is_equal(rgb_pixel, color):
                return False
        return True
    
    image = np.reshape(image, (-1, 3))

    new_colors = [image[0, :]]
    for rgb_pixel in image:
        if is_unique(rgb_pixel, new_colors):
            new_colors.append(rgb_pixel)

    return np.array(new_colors)

def get_colour_sys(illum, sens):
    return np.stack([illum * sens for sens in sens.T], axis = 1)    

import numpy as np
import scipy 
import pygel3d.gl_display as gd
from pygel3d import hmesh as hm 
from scipy.spatial import ConvexHull
import utils
import colorutils as cutils
import awb

NUMBER_OF_SAMPLES = 1000
WAVELENGTH_RANGE = np.arange(400, 701, 10)
D65_xyz = np.array([95.047, 100., 108.883]) / 100

# GENERAL TODO'S (important -> less important):
# 1.PEP8 style guide check
# 2.Make descriptions to funcs
# 3.do we norm r(lambda) ????????????????????UPD:where??
# 4.maybe merge together ocs and mmb funcs? (get_figure_volume/points /// get_figure_s_t[subject to])


def sample_unit_sphere(color_sys, sample_amount, wavelengths = WAVELENGTH_RANGE):
    """
    # returns k that belongs to R^(2N) for mmb and R^N for ocs
    """
    vector_size = color_sys.shape[1]
    # wavelengths_size = np.shape(wavelengths)[0]
    wavelengths_size = 431
    k_sampled = []    
    wave = 0

    for _ in range(sample_amount):
        rand_components = np.random.randn(vector_size - 1, 1)
        k_1 = np.sum(np.multiply(rand_components.T, color_sys[wave, 1:])) / color_sys[wave, 0]
        k_i = np.append(k_1, rand_components)
        k_sampled.append((k_i / np.linalg.norm(k_i)).tolist())
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
               S_phi, S_psi,
               sampling_resolution = NUMBER_OF_SAMPLES):
#illum_phi, sens_phi,
#illum_psi, sens_psi,
    # S_phi = cutils.get_colour_sys(illum_phi, sens_phi)
    # S_psi = cutils.get_colour_sys(illum_psi, sens_psi) # q x N, q - wavelength resolution
    S = np.concatenate((S_phi, S_psi), axis = 1) #q x 2N; q - wavelength resolution

    mmb_extr_points = []

    for k in sample_unit_sphere(S, sampling_resolution):
        print(f"{S.shape=}, {k.shape=}, {np.dot(S, k).shape=}, {S_phi.T.shape=}")
        print(S@k)
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
    S = cutils.get_colour_sys(illum, sens)

    ocs_extr_points = []

    for k in sample_unit_sphere(S, sampling_resolution):
        max_reflectance, min_reflectance =\
        solve_linear_problem(objective_func_coef = np.dot(S, k), bounds = (0,1))

        # scale so the brightest illum color response == 1  
        scale = np.max(np.dot(sens.T, illum))
        
        max_color = np.dot(max_reflectance, S) / scale
        min_color = np.dot(min_reflectance, S) / scale

        ocs_extr_points.extend([min_color, max_color])

    return ocs_extr_points

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

def deltaE_Metamer(x, y, x_wp=None, y_wp=None, 
                   sens_before = None, sens_after = None, 
                   illum = None):
    if x_wp is None:
        x_wp = np.ones_like(x)
    if y_wp is None:
        y_wp = np.ones_like(y)

    return calculate_deltaE_Metamer(x / x_wp * D65_xyz, y / y_wp * D65_xyz,
                                    sens_before, sens_after, illum)

#think of how to make deltaE (not just sum of distances)
def calculate_deltaE_Metamer(pred_tristim, dst_tristim,
                             sens_before = None, sens_after = None, 
                             illum = None):
    """
    dst_tristim  : whole image
    pred_tristim : 
    """
    pred_unique = cutils.get_unique_colors(pred_tristim)
    dst_unique = cutils.get_unique_colors(dst_tristim)

    if pred_unique.shape != dst_unique.shape:
        raise ValueError("Shapes are not equal! (Rethink logic?)")

    distances = []

    for i in range(pred_unique.shape[0]):
        color_mmb = get_mmb_points(dst_unique[i],
                                    sens_phi = sens_before, illum_phi = illum,
                                    sens_psi = sens_after , illum_psi = illum)
        hull_dst = hull_from_points(color_mmb)
        distances.append(dist(hull_dst, pred_unique)) #check for points in if value is < 0

    return distances


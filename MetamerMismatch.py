import numpy as np
import scipy 
import pygel3d.gl_display as gd
from pygel3d import hmesh as hm 
from scipy.spatial import ConvexHull
import plot3D
import utils
import colorutils as cutils
import awb
import matplotlib.pyplot as plt
NUMBER_OF_SAMPLES = 100 # 1000
WAVELENGTH_RANGE = awb.SpectralFunction.lambdas

# GENERAL TODO'S (important -> less important):
# 1.PEP8 style guide check
# 2.Make descriptions to funcs
# 3.do we norm r(lambda) ????????????????????UPD:where??
# 4.maybe merge together ocs and mmb funcs? (get_figure_volume/points /// get_figure_s_t[subject to])


def sample_unit_sphere(color_sys, sample_amount = NUMBER_OF_SAMPLES):
    """
    Samples k-vectors of size N that are zero-crossings of linear combination color_sys_i * k_i

    Parameters
    ----------
    color_sys : (q, N) ndarray
        colour system (illum * sensitivities)
    sample_amount : int, optional
        amount of vectors that are going to be generated.

    q - wavelength resolution. For example, q = int(780 - 350) + 1 
    N - number of channels. For example, r,g,b -> N = 3 

    Returns
    -------
    k : ndarray
        returns (N, sample_amount) array
    """
    
    #remove zeroed arrays
    color_sys_clean = cutils.clear_array_from_zeros(color_sys)
    wavelengths_size, vector_size = color_sys_clean.shape
    k_sampled = []    
    wave = 0

    for _ in range(sample_amount):
        i = cutils.find_non_zero_elem(color_sys_clean[wave, :])
        s_except_i, s_i  = np.concatenate((color_sys_clean[wave, :i], color_sys_clean[wave, i+1:])), color_sys_clean[wave, i]
        # print(f"{s_except_i=}, {s_i=}")
        rand_components = np.random.randn(vector_size - 1, 1)
        k_1 = np.sum(np.multiply(rand_components.T, s_except_i)) / s_i
        k_i = np.append(k_1, rand_components)
        k_sampled.append((k_i / np.linalg.norm(k_i)).tolist())
        wave = (wave + 1) % wavelengths_size

    return np.reshape(np.array(k_sampled), (vector_size, sample_amount)).T

def _solve_linear_problem(objective_func_coef, constrain_func = None, 
                         constrain_func_value = None, bounds = None):
    
    max_Class = scipy.optimize.linprog(c = objective_func_coef, 
                                   A_eq = constrain_func, 
                                   b_eq = constrain_func_value, 
                                   bounds = bounds)

    if max_Class.success:
        return max_Class.x
    return None

    # TODO: check if x_min is NEEDED. i think it might not be right. (ask Vasya?)
    # TODO: add optimized version with SVD (?)

    # check if x_min = -x_max when one = c = ... *-1 and c = ...
    # as addition : k -> -k is also the solve so c = .... * -1 is alright ig

def get_mmb_points(metameric_color,
                illum_phi, sens_phi,
                illum_psi, sens_psi,
                sampling_resolution : int = NUMBER_OF_SAMPLES):
    """
    Returns bounds of metamer body in psi-color-space induced by 'metameric_color' in phi-color-space

    Parameters
    ----------
    metameric_color : array_like
        color's [r,g,b] array 
    illum_phi : (q,) ndarray
        illuminations in first(phi) color space
    sens_phi : (q, N) ndarray
        sensitivities in first(phi) color space
    illum_psi : (q,) ndarray
        illuminations in second(psi) color space
    sens_psi : (q, N) ndarray
        sensitivities in second(psi) color space
    samping_resolution : int, optional
        amount of vectors to sample

    q - wavelength resolution. For example, q = int(780 - 350) + 1 
    N - number of channels. For example, r,g,b -> N = 3 

    Returns
    -------
    mmb_extr_points : list
        returns points of metamer body's bound as list of (r,g,b) arrays  
    """

    S_phi = cutils.get_colour_sys(illum_phi, sens_phi)
    S_psi = cutils.get_colour_sys(illum_psi, sens_psi) # q x N, q - wavelength resolution
    S = np.concatenate((S_phi, S_psi), axis = 1) #q x 2N; q - wavelength resolution
    mmb_extr_points = []

    for k in sample_unit_sphere(S, sampling_resolution):        
        max_reflectance = _solve_linear_problem(objective_func_coef = np.dot(S, k), constrain_func = S_phi.T,
                                                constrain_func_value = metameric_color, bounds = (0, 1))
        if max_reflectance is None:
            print(f"failed to compute lingprog {k=}") #tmp
            continue
        scale = cutils.get_colour_response(sens_psi.T, illum_psi, np.ones(WAVELENGTH_RANGE.shape[0]), WAVELENGTH_RANGE)
        # print(scale)
        max_color_psi = cutils.get_colour_response(sens_psi.T, illum_psi, max_reflectance, WAVELENGTH_RANGE) / scale

        mmb_extr_points.extend([max_color_psi])
    
    return mmb_extr_points

def get_ocs_points(illum, sens, sampling_resolution = NUMBER_OF_SAMPLES):
    """
    Returns bounds of object colour solid in color-space

    Parameters
    ----------
    illum : (q,) ndarray
        illuminations in color space
    sens : (q, N) ndarray
        sensitivities in color space
    samping_resolution : int, optional
        amount of vectors to sample

    q - wavelength resolution. For example, q = int(780 - 350) + 1 
    N - number of channels. For example, r,g,b -> N = 3

    Returns
    -------
    ocs_extr_points : list
        returns points of ocs's bound as list of (r,g,b) arrays  
    """
    S = cutils.get_colour_sys(illum, sens)

    ocs_extr_points = []

    for k in sample_unit_sphere(S, sampling_resolution):
        max_reflectance = _solve_linear_problem(objective_func_coef = np.dot(S, k), bounds = (0,1))

        if max_reflectance is None:
            print(f"failed to compute lingprog {k=}") # tmp
            continue

        scale = cutils.get_colour_response(sens.T, illum, [1] * WAVELENGTH_RANGE.shape[0], WAVELENGTH_RANGE)

        max_color = cutils.get_colour_response(sens.T, illum, max_reflectance, WAVELENGTH_RANGE) / scale

        ocs_extr_points.extend([max_color])

    return ocs_extr_points

#---------------------------------------------------------------------------------------------------------------
#TODO: find a way to remove degenerated triangles without transfering convexhull->manifold like def dist() 
def _distance(points, point):
    """
    #TODO: NOT IMPLEMENTED, use dist() instead.
    #TODO: combine dist and _distance()
    finds distance between body generated by bound's 'points' and 'point'

    Parameters
    ----------
    points : (q, N) array_like
        bound's points
    point : (N,) array_like
        point in N-dimensional space 

    Returns
    -------
    k : float
        returns distance.
    """
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

def dist(hull : ConvexHull, point): #temp(?) version of func
    """
    finds distance between body generated by bound's 'points' and 'point'

    Parameters
    ----------
    hull : ConvexHull
        generated hull from scipy.spatial
    point : (N,) array_like
        point in N-dimensional space 

    Returns
    -------
    k : float
        returns distance.
    """
    m = hm.Manifold()

    for s in hull.simplices:
        m.add_face(hull.points[s])

    dist = hm.MeshDistance(m).signed_distance(point)

    return dist

def deltaE_Metamer(x, y, x_wp=None, y_wp=None, 
                   sens_phi = None, sens_psi = None, illum = None):
    if x_wp is None:
        x_wp = np.ones_like(x)
    if y_wp is None:
        y_wp = np.ones_like(y)

    print("calculating deltaE metamer")
    return calculate_deltaE_Metamer(x, y, sens_phi, sens_psi, illum)

#think of how to make deltaE (not just sum of distances)
def calculate_deltaE_Metamer(pred_tristim, src_tristim,
                             sens_phi, sens_psi, illum):
    """
    calculates distance between pred_tristim and metamer body induced by src_tristim

    WARNING: For different illuminations not released yet
    Parameters
    ----------
    pred_tristim : (N, 3) ndarray
        predicted colors of image in psi-color-space
    src_tristim : (N, 3) ndarray
        original colors of image in phi-color-space 
    sens_phi :
        sensitivities in first color space 
    sens_psi : 
        sensitivities in second color space
    illum :
        illuminations in both color space. 

    N = (width * length) of image

    Returns
    -------
    score : float
        returns sum of distances throughout the whole image.
    """

#    TODO: add COLOR_EPSILON to declare from what point colors are the same

    pred_unique = cutils.get_unique_colors(pred_tristim)
    src_unique = cutils.get_unique_colors(src_tristim)

    print(pred_unique.shape , src_unique.shape)
    if pred_unique.shape != src_unique.shape:
        raise ValueError("Shapes are not equal! (Rethink logic?)")

    distances = []
    # tmp_hull_dst = []
    # tmp_color_pred = []
    # ocs = hull_from_points(get_ocs_points(illum, sens_psi))
    for i in range(pred_unique.shape[0]):
        # print("dst unique ", dst_unique[i])
        color_mmb = get_mmb_points(src_unique[i],
                                   illum_phi = illum, sens_phi = sens_phi,
                                   illum_psi = illum, sens_psi = sens_psi)
        hull_dst_i = hull_from_points(color_mmb)
        # print(f"{np.array(color_mmb).max()=}")
        # tmp_hull_dst.append(hull_dst_i)
        # tmp_color_pred.append(dst_unique[i])
        distances.append(dist(hull_dst_i, pred_unique[i]))
    
    # plot3D.plot_scene(tmp_hull_dst, color_predictions = tmp_color_pred, ocs=ocs)
    # raise ValueError
    # print("dist.len", len(distances), np.array(distances).min(), np.array(distances).max())
    score = np.clip(distances, 0, None)
    return score
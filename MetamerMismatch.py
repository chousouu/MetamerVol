import numpy as np
import scipy 
import pygel3d.gl_display as gd
import pygel3d.hmesh as hm 
from scipy.spatial import ConvexHull
import colorutils as cutils
from constants import NUMBER_OF_SAMPLES, WAVELENGTH_RANGE

# GENERAL TODO'S (important -> less important):
# 1.PEP8 style guide check
# 3.do we norm r(lambda) ????????????????????UPD:where??


def sample_unit_sphere(color_sys, sample_amount : int = NUMBER_OF_SAMPLES):
    """
    Samples k-vectors of size N

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
        returns (sample_amount, N) normed array
    """
    vector_size = color_sys.shape[1]

    sampled_ks = np.random.randn(sample_amount, vector_size)

    return sampled_ks / np.linalg.norm(sampled_ks)

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

    # check if x_min = -x_max when one = c = ... *-1 and c = ...
    # as addition : k -> -k is also the solve so c = .... * -1 is alright ig

def _clip_to_zero_one(reflectances):
    reflectances[reflectances > 0.5] = 1
    reflectances[reflectances <= 0.5] = 0

    return reflectances

def _apply_SVD(S, use_SVD : bool):
    S_or_U = S
    if use_SVD:
        S_or_U, _, _ = np.linalg.svd(S, full_matrices = False)

    return S_or_U

def get_mmb_points(metameric_color,
                illum_phi, sens_phi,
                illum_psi, sens_psi,
                sampling_resolution : int = NUMBER_OF_SAMPLES,
                use_SVD : bool = True):
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
    use_SVD : bool, optional
        Use SVD for better count of volume. default = True

    q - wavelength resolution. For example, q = int(780 - 350) + 1 
    N - number of channels. For example, r,g,b -> N = 3 

    Returns
    -------
    mmb_extr_points : list
        returns points of metamer body's bound as list of (r,g,b) arrays  
    """
    S_phi = cutils.get_colour_sys(illum_phi, sens_phi) # q x N
    S_psi = cutils.get_colour_sys(illum_psi, sens_psi) # q x N

    S = np.concatenate((S_phi, S_psi), axis = 1) #q x 2N

    scale_to_sys = cutils.get_colour_response(sens_phi.T, illum_phi, np.ones(WAVELENGTH_RANGE.shape[0]), WAVELENGTH_RANGE)
    scaled_metameric_color = metameric_color * scale_to_sys
    
    mmb_extr_points = []
    
    S_or_U = _apply_SVD(S, use_SVD)

    for k in sample_unit_sphere(S, sampling_resolution):  
        max_reflectance = _solve_linear_problem(objective_func_coef = np.dot(S_or_U, k), constrain_func = S_phi.T,
                                                constrain_func_value = scaled_metameric_color, bounds = (0, 1))

        max_reflectance = _clip_to_zero_one(max_reflectance)
        
        if max_reflectance is None:
            continue
        
        scale = cutils.get_colour_response(sens_psi.T, illum_psi, np.ones(WAVELENGTH_RANGE.shape[0]), WAVELENGTH_RANGE)
        max_color_psi = cutils.get_colour_response(sens_psi.T, illum_psi, max_reflectance, WAVELENGTH_RANGE) / scale

        mmb_extr_points.extend([max_color_psi])
    return mmb_extr_points

def get_ocs_points(illum, sens, 
                   sampling_resolution : int = NUMBER_OF_SAMPLES,
                   use_SVD : bool = True):
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
    use_SVD : bool, optional
        Use SVD for better count of volume. default = True

    q - wavelength resolution. For example, q = int(780 - 350) + 1 
    N - number of channels. For example, r,g,b -> N = 3

    Returns
    -------
    ocs_extr_points : list
        returns points of ocs's bound as list of (r,g,b) arrays  
    """
    S = cutils.get_colour_sys(illum, sens)
    
    S_or_U = _apply_SVD(S, use_SVD)

    ocs_extr_points = []

    for k in sample_unit_sphere(S, sampling_resolution):
        max_reflectance = _solve_linear_problem(objective_func_coef =  np.dot(S_or_U, k), bounds = (0,1))

        max_reflectance = _clip_to_zero_one(max_reflectance)

        if max_reflectance is None:
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

    pred_unique = cutils.get_unique_colors(pred_tristim)
    src_unique = cutils.get_unique_colors(src_tristim)

    if pred_unique.shape != src_unique.shape:
        raise ValueError(f"Shapes are not equal!",
                        "Number of pred_tristim and src_tristim unique colors have to be equal.",
                        f"{pred_unique.shape=}, {src_unique.shape=}")

    distances = []
    for i in range(pred_unique.shape[0]):
        color_mmb = get_mmb_points(src_unique[i],
                                   illum_phi = illum, sens_phi = sens_phi,
                                   illum_psi = illum, sens_psi = sens_psi)
        hull_dst_i = hull_from_points(color_mmb)

        distances.append(dist(hull_dst_i, pred_unique[i]))
    
    score = np.clip(distances, 0, None)
    return score


def get_scene_details(pred_tristim, src_tristim,
                             sens_phi, sens_psi, illum):
    """
    Prepares all information needed for plotting 
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
    src_hulls : list
        list of metamer hulls (type == class ConvexHull) made from 'src_unique'
    pred_unique : ndarray
        array of unique colors made from 'pred_tristim'
    ocs : ConvexHull
        Object color solid in psi-color-space
    """

    pred_unique = cutils.get_unique_colors(pred_tristim)
    src_unique = cutils.get_unique_colors(src_tristim)

    if pred_unique.shape != src_unique.shape:
        raise ValueError(f"Shapes are not equal!",
                        "Number of pred_tristim and src_tristim unique colors have to be equal.",
                        f"{pred_unique.shape=}, {src_unique.shape=}")

    src_hulls = []
    for i in range(pred_unique.shape[0]):
        color_mmb = get_mmb_points(src_unique[i],
                                   illum_phi = illum, sens_phi = sens_phi,
                                   illum_psi = illum, sens_psi = sens_psi)
        src_hulls.append(hull_from_points(color_mmb))

    ocs = hull_from_points(get_ocs_points(illum, sens_psi)) 

    return src_hulls, pred_unique, ocs
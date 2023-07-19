import numpy as np
import scipy 
import pygel3d.gl_display as gd
from pygel3d import hmesh as hm 
from scipy.spatial import ConvexHull
import plot3D
import utils
import colorutils as cutils
import awb
from idk import get_chart_data

NUMBER_OF_SAMPLES = 10 # 1000
WAVELENGTH_RANGE = awb.SpectralFunction.lambdas
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
    #clean from zeroed arrays
    # print(f"before, {color_sys.shape=}")
    color_sys_clean = cutils.clear_array_from_zeros(color_sys)
    # print(f"after, {color_sys_clean.shape=}")
    wavelengths_size, vector_size = color_sys_clean.shape
    # print(f"{wavelengths_size=}, {vector_size=}")
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

    S_phi = cutils.get_colour_sys(illum_phi, sens_phi)
    S_psi = cutils.get_colour_sys(illum_psi, sens_psi) # q x N, q - wavelength resolution
    S = np.concatenate((S_phi, S_psi), axis = 1) #q x 2N; q - wavelength resolution

    mmb_extr_points = []
    # print("shape phi", S.shape)


    for k in sample_unit_sphere(S, sampling_resolution):
        # print(f"{S.shape=}, {k.shape=}, {np.dot(S, k).shape=}, {S_phi.T.shape=}")
        
        # print(f"{S=}\n\n\n\n\n\n\n")
        # print(f"{k=}\n\n\\n\n\n\n\n")
        # print(np.dot(S,k))
        # print(np.all(np.isnan(np.dot(S, k))))    
        # print("sampled k")
        max_reflectance, min_reflectance =\
        solve_linear_problem(objective_func_coef = np.dot(S, k), constrain_func = S_phi.T ,
                             constrain_func_value = metameric_color, bounds = (0, 1))

        # print("solved problem.")
        # scale so the brightest illum color response == 1          
        scale = cutils.get_colour_response(sens_psi.T, illum_psi, [1] * WAVELENGTH_RANGE.shape[0], WAVELENGTH_RANGE)
        # print(scale)
        max_color_psi = cutils.get_colour_response(sens_psi.T, illum_psi, max_reflectance, WAVELENGTH_RANGE) / scale
        min_color_psi = cutils.get_colour_response(sens_psi.T, illum_psi, min_reflectance, WAVELENGTH_RANGE) / scale

        # print(max_reflectance.shape, S_psi.shape)
        # print('shape', (S_psi.T).shape)
        # scale = np.trapz(S_psi.T, WAVELENGTH_RANGE, axis=-1)
        # print(scale)
        # max_color_psi = np.trapz(max_reflectance * S_psi.T, WAVELENGTH_RANGE, axis=-1)  / scale
        # min_color_psi = np.trapz(min_reflectance * S_psi.T, WAVELENGTH_RANGE, axis=-1)  / scale
        # max_color_psi = np.dot(max_reflectance, S_psi) / scale
        # min_color_psi = np.dot(min_reflectance, S_psi) / scale

        mmb_extr_points.extend([min_color_psi, max_color_psi])

    # print("Returning mmb points")
    return mmb_extr_points

def get_ocs_points(illum, sens, sampling_resolution = NUMBER_OF_SAMPLES):
    S = cutils.get_colour_sys(illum, sens)

    ocs_extr_points = []

    for k in sample_unit_sphere(S, sampling_resolution):
        max_reflectance, min_reflectance =\
        solve_linear_problem(objective_func_coef = np.dot(S, k), bounds = (0,1))

        # scale so the brightest illum color response == 1  
        scale = cutils.get_colour_response(sens.T, illum, [1] * WAVELENGTH_RANGE.shape[0], WAVELENGTH_RANGE)

        max_color = cutils.get_colour_response(sens.T, illum, max_reflectance, WAVELENGTH_RANGE) / scale
        min_color = cutils.get_colour_response(sens.T, illum, min_reflectance, WAVELENGTH_RANGE) / scale

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
                   sens_phi = None, sens_psi = None, illum = None):
    if x_wp is None:
        x_wp = np.ones_like(x)
    if y_wp is None:
        y_wp = np.ones_like(y)

    print("calculating deltaE metamer")
    return calculate_deltaE_Metamer(x, y, sens_phi, sens_psi, illum)
    return calculate_deltaE_Metamer(x / x_wp * D65_xyz, y / y_wp * D65_xyz,
                                    sens_phi, sens_psi, illum)

#think of how to make deltaE (not just sum of distances)
def calculate_deltaE_Metamer(pred_tristim, dst_tristim,
                             sens_phi = None, sens_psi = None, illum = None):
    """
    dst_tristim  : whole image
    pred_tristim : 
    """
    pred_unique = cutils.get_unique_colors(pred_tristim)
    dst_unique = cutils.get_unique_colors(dst_tristim)

    print(pred_unique.shape , dst_unique.shape)
    if pred_unique.shape != dst_unique.shape:
        raise ValueError("Shapes are not equal! (Rethink logic?)")

    distances = []
    tmp_hull_dst = []
    tmp_color_pred = []
    ocs = hull_from_points(get_ocs_points(illum, sens_psi))
    for i in range(pred_unique.shape[0]):
        print("dst unique ", dst_unique[i])
        color_mmb = get_mmb_points(dst_unique[i],
                                   illum_phi = illum, sens_phi = sens_phi,
                                   illum_psi = illum, sens_psi = sens_psi)
        
        # print('aaaaaaaaaaaaaaaaaaaa', np.array(color_mmb).max(), np.array(color_mmb).min())
        hull_dst_i = hull_from_points(color_mmb)
        print(f"{np.array(color_mmb).max()=}")
        tmp_hull_dst.append(hull_dst_i)
        tmp_color_pred.append(dst_unique[i])
        distances.append(dist(hull_dst_i, pred_unique[i]))
    
    plot3D.plot_scene(tmp_hull_dst, color_predictions = tmp_color_pred, ocs=ocs)
    raise ValueError
    print("dist.len", len(distances), np.array(distances).min(), np.array(distances).max())
    return np.clip(distances, 0, None)
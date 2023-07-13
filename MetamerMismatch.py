import numpy as np
import scipy 

NUMBER_OF_SAMPLES = 1000
WAVELENGTH_RESOLUTION = 10

# GENERAL TODO'S (important -> less important):
# 1.PEP8 style guide check
# 2.Make descriptions to funcs
# 3.do we norm r(lambda) ????????????????????UPD:where??


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
    
    S_phi = np.stack([illum_phi * sens for sens in sens_phi.T], axis = 1) #color sys; size = q x N
    S_psi = np.stack([illum_psi * sens for sens in sens_psi.T], axis = 1) #
    S = np.concatenate((S_phi, S_psi), axis = 1) #q x 2N; q - wavelength resolution

    mmb_extr_points = []

    # print("shape of unit sphere", sample_unit_sphere(vector_size, sampling_resolution).shape)

    # !!!!is amount of k's component should be equal to 2N / N for mmb/ocs? according to (3) in the statya 
    for k in sample_unit_sphere(S, wavelengths, sampling_resolution):

        # print("S, k, S@K", S.shape, k.shape, np.dot(S, k).shape)
        # print("constr_func = value :", S_phi.shape, "=", metameric_color.shape, metameric_color)
        # print("z0 size : ", metameric_color.ndim, metameric_color.shape, metameric_color)
        # print(f"b = {metameric_color.shape}/ {metameric_color.ndim}, A_Eq = {S_phi.T.shape}, {metameric_color.shape} == {S_phi.T.shape[0]}")

        max_reflectance, min_reflectance =\
        solve_linear_problem(objective_func_coef = np.dot(S, k), constrain_func = S_phi.T ,
                             constrain_func_value = metameric_color, bounds = (0, 1))

        # scale so the brightest illum color response == 1  
        # scale = np.max(S_psi)
        # print(max_reflectance, min_reflectance)
        
        scale = np.max(np.dot(sens_psi.T, illum_psi))

        max_color_psi = np.dot(max_reflectance, S_psi) / scale
        min_color_psi = np.dot(min_reflectance, S_psi) / scale
       
        # print('==========')
        # print(max_color_psi / scale)
        # print('\n\n\n\n\n\n')
        # print([max_color_psi / scale, min_color_psi / scale])

        # print('==========')
        mmb_extr_points.extend([min_color_psi, max_color_psi])
        # print('==========')

    return mmb_extr_points

#TODO: maybe merge together ocs and mmb funcs? (get_figure_volume/points)

def get_ocs_points(illum, sens, sampling_resolution = NUMBER_OF_SAMPLES):
    S = np.stack([illum * sens_ for sens_ in sens.T], axis = 1)

#TODO: HARDCODED NEED TO BE FIXED! (maybe refactor functions)
    wavelengths = np.arange(400, 701, WAVELENGTH_RESOLUTION) 

    ocs_extr_points = []

    for k in sample_unit_sphere(S, wavelengths, sampling_resolution):
        max_reflectance, min_reflectance =\
        solve_linear_problem(objective_func_coef = np.dot(S, k), bounds = (0,1))

        # scale so the brightest illum color response == 1  
        scale = np.max(np.dot(sens.T, illum))
        # scale = np.max(S)
        
        max_color = np.dot(max_reflectance, S) / scale
        min_color = np.dot(min_reflectance, S) / scale

        ocs_extr_points.extend([min_color, max_color])

    return ocs_extr_points


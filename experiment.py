import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import numpy as np
import MetamerMismatch as mm

from dataclasses import dataclass
import argparse
from pathlib import Path
from functools import partial

import awb #;;;;;

from awb import SpectralFunction, SpectralSensitivies, CameraRender
from awb import sensitivies_path, load_illuminants, spectral_data
from awb import STD_CHANNELS_NAMES, DC, SG


from skimage import color
from tqdm    import tqdm

import random, shutil, h5py, os

from awb.spectral_data           import SpectralData, cameras, reflectances, illuminances
from awb.rendering.camera_render import CameraRender, camera_render
from awb.pipeline.camera         import CameraColorConfig
from awb.spectral_data           import illuminances_path

from sklearn.preprocessing   import PolynomialFeatures
from sklearn.linear_model    import LinearRegression
from sklearn.model_selection import LeaveOneOut

from utils import RootPolynomialFeatures, RobustScalableRationalFeatures, Poly_RootPoly_Scalrat, RootPoly_Scalrat

@dataclass
class ExperimentSetup:
    noise: bool = False
    SNR: float = None
    runs: int = 1
    positive: bool = False
    use_wp: bool = False
    error_metric: str = 'metamer'

    @classmethod
    def from_argparse(cls, args):
        return cls(
                runs = args.runs,
                positive = args.positive,
                use_wp = args.use_wp,
                error_metric = args.error_metric
                )

def train_model(X, y, positive=False):
    regression = LinearRegression(positive=positive, fit_intercept=False)
    regression.fit(X, y)

    return regression

def check_model(src_tristim_train, dst_tristim_train,
                 src_tristim_test, feature_generator,
                 positive = False):
    src_features_train = feature_generator.fit_transform(src_tristim_train)
    src_features_test  = feature_generator.fit_transform(src_tristim_test)

    model = train_model(src_features_train, dst_tristim_train, positive)

    predicted = model.predict(src_features_test)
    assert (not positive) or (predicted >= 0).all()

    return np.clip(predicted, 0, None)

def normalize_chart_colors(colors):
    """
    colors : Nx3, N - amount of colors
    """
    if np.any(colors > 1):
        colors /= np.amax(colors, axis = 0)

    return colors

def view(img):
    from PIL import Image
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


def run_once(src, dst, error_fun, settings):
    metrics_df = pd.DataFrame(columns = ['mean', 'median', '95 pt'])

    K_s = range(2, 5)

    loo = LeaveOneOut()

    print("in run_once()")
    for K in K_s:

        feature_name_generator_s = \
        {
            f'PCC_{K}' : PolynomialFeatures(K, include_bias = False),
            f'RPCC_{K}': RootPolynomialFeatures(K),
            f'SRCC_{K}': RobustScalableRationalFeatures(K)
        }

        for feature_name, feature_generator in feature_name_generator_s.items():
            pred = np.empty_like(dst)
            print('===========================')
            print(f"{feature_name}")
            for train_index_s, test_index in loo.split(src):
                cur_X_train, cur_Y_train = src[train_index_s, ...], dst[train_index_s, ...]
                # noise only X and if needed
                cur_X_test, cur_Y_test = src[test_index, ...], dst[test_index, ...]

                pred[test_index, ...] = check_model(cur_X_train,
                                                    cur_Y_train,
                                                    cur_X_test,
                                                    feature_generator,
                                                    positive=settings.positive)

            pred = normalize_chart_colors(pred)
            print("got best model from check_model")
            results = error_fun(pred, src) # CHECK! was dst

            print(f"results : {np.array(results).shape}")
            results_mean = np.mean(results)
            results_median = np.median(results)
            results_95pt = np.percentile(results, 95)

            metrics_df = pd.concat([metrics_df,
                                    pd.DataFrame([[results_mean,
                                                 results_median,
                                                 results_95pt]],
                                   columns=metrics_df.columns,
                                   index=[feature_name])])

# 
# awb.spectral_rendering.DC.render_img_from_colors(src)


            print('===========================')
    return metrics_df

def preprocess_data(src, dst, settings, src_wp=None, dst_wp=None):
    if settings.use_wp:
        src /= src_wp[None, ...]
        dst /= dst_wp[None, ...]

    return src, dst

def run_repeatedly(src, dst, err_fun, settings): # not using
    runs = settings.runs
    if runs > 1 and not settings.noise:
        raise ValueError(f'When performing experiment without noise'
                          'runs should be equal to 1.')

    dfs = []
    for _ in range(runs):
        dfs.append(run_once(src, dst, err_fun, settings))

    total = pd.concat(dfs)

    return total
 
def get_chart_data(chart, render, illum):
    color_reflectance_s = list(chart.reflectances.values())

    tristim_s = []
    for color_reflectance in color_reflectance_s:
        radiance = color_reflectance * illum
        tristim = render.render(radiance)

        tristim_s.append(tristim)

    tristim_s = np.stack(tristim_s, axis = 0)
    return tristim_s

def prepare_and_run(settings, outdir):
    illuminance_data = pd.read_csv(illuminances_path/'cie_2018/std.csv')
    illum = SpectralFunction(illuminance_data['D65'], illuminance_data['wavelength'])

    charts = {'DC': DC(exclude_periphery=True),
              'SG': SG(exclude_periphery=True)}

    dst_camera = SpectralSensitivies.from_csv(sensitivies_path / 'iitp/xyz_matching_fun.csv',
                                               normalized=False)
    dst_render = CameraRender(dst_camera)

    base_src_cameras = {
            'Sony_DXC-930':
            SpectralSensitivies.from_csv(sensitivies_path / 'iitp/Sony_DXC-930.csv',
                                         normalized=False,
                                         channels_names=STD_CHANNELS_NAMES),
            'canon600d': SpectralSensitivies.from_csv(sensitivies_path / 'iitp/canon600d.csv',
                                                      normalized=False,
                                                      channels_names=STD_CHANNELS_NAMES)
            }
    src_cameras = base_src_cameras

    results = dict()

    print("Got all cameras, starting run")
        
    for chart_name, chart in charts.items():
        for camera_name, src_camera in src_cameras.items():
            src_render = CameraRender(src_camera)

            src_data = get_chart_data(chart, src_render, illum)
            dst_data = get_chart_data(chart, dst_render, illum)
            print(f"{src_data.shape=}, {dst_data.shape=}")

            src_wp = src_render.render(illum)
            dst_wp = dst_render.render(illum)

            src_data, dst_data = preprocess_data(src_data, dst_data, settings, src_wp, dst_wp)

            if settings.error_metric == 'metamer':
                src_cam_sens = (src_camera.sensitivities.y).T
                dst_cam_sens = (dst_camera.sensitivities.y).T

                err_fun = partial(mm.deltaE_Metamer, x_wp=dst_wp, y_wp=dst_wp,
                                  sens_phi = src_cam_sens, sens_psi = dst_cam_sens, illum = illum.y)
            else:
                raise ValueError(f'Error metric {settings.error_metric} is not a valid metric.')

            df = run_repeatedly(src_data, dst_data, err_fun, settings) #suda

            print(f'Chart {chart_name} and camera {camera_name}')

            df.to_csv(outdir / f'{camera_name}_{chart_name}.csv')

            results[(chart_name, camera_name)] = df


    res = pd.concat(results.values()).groupby(level = 0)
    std = res.std().round(5)
    mean = res.mean().round(5)

    mean.to_csv(outdir / 'mean.csv')
    std.to_csv(outdir / 'std.csv')

    return mean, std

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outdir', type = Path, default = Path("/home/yasin/iitp/tempdir"))
    parser.add_argument('--runs', type = int, default = 1)
    parser.add_argument('--use_wp', action = 'store_true')
    parser.add_argument('--positive', action = 'store_true')
    parser.add_argument('--error_metric', type=str, default = 'metamer')

    args = parser.parse_args()

    args.outdir.mkdir(exist_ok = True, parents = True)

    settings = ExperimentSetup.from_argparse(args)

    prepare_and_run(settings, args.outdir)
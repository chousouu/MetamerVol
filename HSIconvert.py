import numpy as np
import awb
import h5py
from pathlib import Path
import argparse
from os.path import splitext
from PIL import Image 

PATH_LI = "/home/yasin/iitp/li_ds"
ILLUM_PATH = "cie/std.csv"
SAVE_IMAGES = '/home/yasin/iitp/tempdir'

def read_h5(h5_path : Path, wv_hsi: np.ndarray):
    with h5py.File(h5_path, 'r') as fh:
        hsi_y = np.transpose(fh['img\\']).astype(np.float32)
        hsi = awb.SpectralFunction(hsi_y, wv_hsi, dtype=np.float32)
        
        return hsi

def add_ill(data, ill_path): ## returns SpectralFunction
    illums_path = awb.spectral_data.illuminances_path / ILLUM_PATH
    wv, y_dict = awb.spectral_data.read_csv(illums_path, wavelength_key='wavelength')

    y_all = y_dict[ill_path]
    illums = awb.SpectralFunction(y_all, wv)

    return illums * data

def view(img):
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

def get_colour_responses(image_name, 
                sensitivity_name : str = 'canon', 
                illuminance : str = 'Standard Illuminant D65', 
                outdir : Path = None):
    awb.SpectralFunction.lambdas = np.linspace(400, 730, num=34, endpoint=True) 
    path_to_hsi = PATH_LI + '/' + image_name

    wv_hsi = np.linspace(400, 730, num=34, endpoint=True) 
    data = read_h5((path_to_hsi), wv_hsi)

    with_ill = add_ill(data, illuminance)

    sensitivities = None 
    if sensitivity_name == 'canon':
        sensitivities = awb.SpectralSensitivies.from_csv(
        awb.sensitivies_path / 'iitp/canon600d.csv', 
        normalized=False, channels_names=awb.STD_CHANNELS_NAMES)
    elif sensitivity_name == 'xyz':
        sensitivities = awb.SpectralSensitivies.from_csv(
        awb.sensitivies_path / 'iitp/xyz_matching_fun.csv',
        normalized=False)
    else:
        raise ValueError("There is no such sensitivities!")

    img = awb.spectral_rendering.camera_render(with_ill, sensitivities)
    img = np.clip(img / np.max(img), 0, 1)

    if outdir is not None:
        save_path = splitext(outdir)[0] + '/' + splitext(image_name)[0] + '.png'
        awb.imsave(Path(save_path), img, unchanged=False, out_dtype=np.uint16, gamma_correction=True)
    
    return img


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outdir', type=Path, default=Path(SAVE_IMAGES))
    parser.add_argument('-s', '--sensitivities', type=str, default='canon')
    args = parser.parse_args()

    args.outdir.mkdir(exist_ok=True, parents=True)
    
    for name in ["2019-09-11_047.h5", "2019-09-08_007.h5", "2019-09-09_013.h5",
                 "2019-08-29_021.h5", "2019-08-26_023.h5", "2019-08-25_004.h5",
                 "2019-09-08_015.h5", "2019-08-28_016.h5", "2019-09-08_007.h5"]:
        get_colour_responses(name, sensitivity_name='xyz', outdir = args.outdir)

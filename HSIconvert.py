import numpy as np
import awb
import h5py
from pathlib import Path
import argparse
from os.path import splitext


Path_li = "/home/yasin/iitp/li_ds/"
def read_h5(h5_path : Path, wv_hsi: np.ndarray):
    with h5py.File(h5_path, 'r') as fh:
        hsi_y = np.transpose(fh['img\\']).astype(np.float32)
        hsi = awb.SpectralFunction(hsi_y, wv_hsi, dtype=np.float32)
        
        return hsi


def add_ill(data, ill_path): ## returns SpectralFunction
    illums_path = awb.spectral_data.illuminances_path / ill_path
    wv, y_dict = awb.spectral_data.read_csv(illums_path, wavelength_key='wavelength')
    y_all = np.vstack([y_array for y_array in y_dict.values()])
    illums = awb.SpectralFunction(y_all, wv)

    return illums * data


def convert2png(args, image_name):
    awb.SpectralFunction.lambdas = np.linspace(400, 730, num=34, endpoint=True) 
    path_to_hsi = Path_li + '/' + image_name

    wv_hsi = np.linspace(400, 730, num=34, endpoint=True) 
    data = read_h5((path_to_hsi), wv_hsi)

    with_ill = add_ill(data, 'iitp/D50.csv')

    if (args.sensetivities == 'cam'):
        cam = awb.SpectralSensitivies.from_csv(
        awb.sensitivies_path / 'iitp/canon600d.csv', 
        normalized=False,
        channels_names=awb.STD_CHANNELS_NAMES)
        img = awb.spectral_rendering.camera_render(with_ill, cam)
        img = np.clip(img / np.max(img), 0, 1)
    elif (args.sensetivities == 'xyz'):
        xyz = awb.SpectralSensitivies.from_csv(
        awb.sensitivies_path / 'iitp/xyz_matching_fun.csv',
        normalized=False)
        img = awb.spectral_rendering.camera_render(with_ill, xyz)
        img = np.clip(img / np.max(img), 0, 1)

     
    save_path = splitext(args.outdir)[0] + '/' + image_name + '.png'
    awb.imsave(Path(save_path), img, unchanged=False, out_dtype=np.uint16, gamma_correction=True)
    return save_path


if __name__ == '__main__':
    # HSI_name = "2019-09-11_047.h5" # motorcycle
    # HSI_name = "2019-09-08_007.h5" # another text
    # HSI_name = "2019-09-09_013.h5" # money
    # HSI_name = "2019-08-29_021.h5" # cottons
    # HSI_name = "2019-08-26_023.h5" # (red) watermelon
    # HSI_name = "2019-08-25_004.h5" # child room
    # HSI_name = "2019-09-08_015.h5" # text
    # HSI_name = "2019-08-28_016.h5" # green(red) leaves
    # HSI_name = "2019-09-18_003.h5" # colorchecker image

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outdir', type=Path, default=Path('/home/yasin/iitp/tempdir'))
    parser.add_argument('-s', '--sensetivities', type=str, default='xyz')

    args = parser.parse_args()

    args.outdir.mkdir(exist_ok=True, parents=True)

    for name in ["2019-09-11_047.h5", "2019-09-08_007.h5", "2019-09-09_013.h5",
                 "2019-08-29_021.h5",
                "2019-08-26_023.h5",
                "2019-08-25_004.h5",
                "2019-09-08_015.h5", "2019-08-28_016.h5"]:
        convert2png(args, name)
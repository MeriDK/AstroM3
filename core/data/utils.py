import numpy as np
import pandas as pd
from astropy.io import fits
from util.preprocess_data import clip_outliers
from scipy import stats
from io import BytesIO


def preprocess_spectra(spectra):
    wavelengths, fluxes = spectra[:, 0], spectra[:, 1]
    fluxes = np.interp(np.arange(3850, 9000, 2), wavelengths, fluxes)
    fluxes = (fluxes - fluxes.mean()) / fluxes.std()
    fluxes = fluxes.reshape(1, -1).astype(np.float32)

    return fluxes


def readLRSFits(filename, z_corr=False):
    """
    Read LAMOST fits file
      adapted from https://github.com/fandongwei/pylamost

    Parameters:
    -----------
    filename: str
      name of the fits file
    z_corr: bool.
      if True, correct for measured radial velocity of star

    Returns:
    --------
    spec: numpy array
      wavelength, flux, inverse variance
    """

    hdulist = fits.open(filename)
    len_list = len(hdulist)

    if len_list == 1:
        head = hdulist[0].header
        scidata = hdulist[0].data
        coeff0 = head['COEFF0']
        coeff1 = head['COEFF1']
        pixel_num = head['NAXIS1']
        specflux = scidata[0,]
        ivar = scidata[1,]
        wavelength = np.linspace(0, pixel_num - 1, pixel_num)
        wavelength = np.power(10, (coeff0 + wavelength * coeff1))
        hdulist.close()
    elif len_list == 2:
        head = hdulist[0].header
        scidata = hdulist[1].data
        wavelength = scidata[0][2]
        ivar = scidata[0][1]
        specflux = scidata[0][0]
    else:
        raise ValueError(f'Wrong number of fits files. {len_list} should be 1 or 2')

    if z_corr:
        try:
            # correct for radial velocity of star
            redshift = head['Z']
        except Exception as e:
            print(e, 'Setting redshift to zero')
            redshift = 0.0

        wavelength = wavelength - redshift * wavelength

    return np.vstack((wavelength, specflux, ivar)).T


def preprocess_lc(X, period, clip, seq_len, phased, aux):
    # 2 sort based on HJD
    sorted_indices = np.argsort(X[:, 0])
    X = X[sorted_indices]

    # 3 clip outliers
    # TODO double check clip outliers function
    if clip:
        t, y, y_err = X[:, 0], X[:, 1], X[:, 2]
        if len(t) > 20:
            t, y, y_err, _, _, _, _, _ = clip_outliers(t, y, y_err, measurements_in_flux_units=True,
                                                       initial_clip=(20, 5), clean_only=True)
        X = np.vstack((t, y, y_err)).T

    # Calculate min max before normalization
    log_abs_min = 0 if min(X[:, 1]) == 0 else np.log(abs(min(X[:, 1])))
    log_abs_max = np.log(abs(max(X[:, 1])))

    # 4 normalize
    mean = X[:, 1].mean()
    std = stats.median_abs_deviation(X[:, 1])
    X[:, 0] = (X[:, 0] - X[:, 0].min()) / (X[:, 0].max() - X[:, 0].min())
    X[:, 1] = (X[:, 1] - mean) / std
    X[:, 2] = X[:, 2] / std

    # 5 trim if longer than seq_len
    if X.shape[0] > seq_len:
        start = np.random.randint(0, len(X) - seq_len)
        X = X[start:start + seq_len, :]

        # if self.split == 'train':
        #     start = np.random.randint(0, len(X) - self.seq_len)
        #     X = X[start:start + self.seq_len, :]
        # else:
        #     X = X[:self.seq_len, :]

    # 1 phase
    if phased:
        X = np.vstack(((X[:, 0] % period) / period, X[:, 1], X[:, 2])).T

    # pad if needed and create mask
    mask = np.ones(seq_len)
    if X.shape[0] < seq_len:
        mask[X.shape[0]:] = 0
        X = np.pad(X, ((0, seq_len - X.shape[0]), (0, 0)), 'constant', constant_values=(0,))

    # add aux
    if aux:
        log_abs_mean = np.log(abs(mean))
        log_std = np.log(std)

        # aux = np.tile([log_abs_min, log_abs_max, log_abs_mean, log_std, log_period], (self.seq_len, 1))
        aux = np.tile([log_abs_min, log_abs_max, log_abs_mean, log_std], (seq_len, 1))
        X = np.concatenate((X, aux), axis=-1)

    # 6 convert X and mask from float64 to float32
    X = X.astype(np.float32)
    mask = mask.astype(np.float32)

    return X, mask


def get_vlc(file_name, v_prefix, reader_v):
    csv = BytesIO()
    file_name = file_name.replace(' ', '')
    data_path = f'{v_prefix}/{file_name}.dat'

    csv.write(reader_v.read(data_path))
    csv.seek(0)

    lc = pd.read_csv(csv, sep='\s+', skiprows=2, names=['HJD', 'MAG', 'MAG_ERR', 'FLUX', 'FLUX_ERR'],
                     dtype={'HJD': float, 'MAG': float, 'MAG_ERR': float, 'FLUX': float, 'FLUX_ERR': float})

    return lc[['HJD', 'FLUX', 'FLUX_ERR']].values


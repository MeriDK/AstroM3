import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import stats
from io import BytesIO

from dev.util import ParallelZipFile as ZipFile


class PSMDataset(Dataset):
    def __init__(self, config, split='train'):
        super(PSMDataset, self).__init__()

        self.split = split
        self.mode = config['mode']
        self.data_root = config['data_root']
        self.df = pd.read_csv(os.path.join(self.data_root, f'{config["file"]}_{self.split}_norm.csv'))
        self.reader_v = ZipFile(os.path.join(self.data_root, config['v_zip']))
        self.v_prefix = config['v_prefix']
        self.lamost_spec_dir = os.path.join(self.data_root, config['lamost_spec_dir'])
        self.meta_cols = config['meta_cols']
        self.photo_cols = config['photo_cols']

        self.min_samples = config['min_samples']
        self.max_samples = config['max_samples']
        self.classes = config['classes']
        self.seq_len = config['seq_len']
        self.phased = config['phased']
        self.p_aux = config['p_aux']
        self.p_enc_in = config['p_enc_in']
        self.s_mad = config['s_mad']
        self.s_aux = config['s_aux']
        self.s_err = config['s_err']
        self.s_err_norm = config['s_err_norm']
        self.s_enc_in = config['s_conv_channels'][0]

        self.random_seed = config['random_seed']
        np.random.seed(self.random_seed)

        self._filter_classes()
        self._limit_samples()

        self.id2target = {i: x for i, x in enumerate(sorted(self.df['target'].unique()))}
        self.target2id = {v: k for k, v in self.id2target.items()}
        self.num_classes = len(self.id2target)

    def _filter_classes(self):
        if self.classes:
            self.df = self.df[self.df['target'].isin(self.classes)]

    def _limit_samples(self):
        if self.min_samples:
            value_counts = self.df['target'].value_counts()
            classes_to_remove = value_counts[value_counts < self.min_samples].index
            self.df = self.df[~self.df['target'].isin(classes_to_remove)]

        if self.max_samples:
            value_counts = self.df['target'].value_counts()
            classes_to_limit = value_counts[value_counts > self.max_samples].index

            for class_type in classes_to_limit:
                class_indices = self.df[self.df['target'] == class_type].index
                indices_to_keep = np.random.choice(class_indices, size=self.max_samples, replace=False)
                self.df = self.df.drop(index=set(class_indices) - set(indices_to_keep))

    def get_vlc(self, file_name):
        csv = BytesIO()
        file_name = file_name.replace(' ', '')
        data_path = f'{self.v_prefix}/{file_name}.dat'

        csv.write(self.reader_v.read(data_path))
        csv.seek(0)

        lc = pd.read_csv(csv, sep='\s+', skiprows=2, names=['HJD', 'MAG', 'MAG_ERR', 'FLUX', 'FLUX_ERR'],
                         dtype={'HJD': float, 'MAG': float, 'MAG_ERR': float, 'FLUX': float, 'FLUX_ERR': float})

        return lc[['HJD', 'FLUX', 'FLUX_ERR']].values

    def readLRSFits(self, filename):
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

        return np.vstack((wavelength, specflux, ivar)).T

    def preprocess_lc(self, X, period, aux_values):
        # Sort based on HJD
        sorted_indices = np.argsort(X[:, 0])
        X = X[sorted_indices]

        # Normalize
        mean = X[:, 1].mean()
        mad = stats.median_abs_deviation(X[:, 1])
        X[:, 1] = (X[:, 1] - mean) / mad
        X[:, 2] = X[:, 2] / mad

        # save delta t before scaling
        delta_t = (X[:, 0].max() - X[:, 0].min()) / 365

        if not self.phased:
            # scale time from 0 to 1
            X[:, 0] = (X[:, 0] - X[:, 0].min()) / (X[:, 0].max() - X[:, 0].min())

        # Trim if longer than seq_len
        if X.shape[0] > self.seq_len:
            if self.split == 'train':   # random crop
                start = np.random.randint(0, len(X) - self.seq_len)
            else:  # 'center'
                start = (len(X) - self.seq_len) // 2

            X = X[start:start + self.seq_len, :]

        # Phase
        if self.phased:
            X = np.vstack(((X[:, 0] % period) / period, X[:, 1], X[:, 2])).T

            # Sort again cause phasing ruined the order
            sorted_indices = np.argsort(X[:, 0])
            X = X[sorted_indices]

        # Pad if needed and create mask
        mask = np.ones(self.seq_len)
        if X.shape[0] < self.seq_len:
            mask[X.shape[0]:] = 0
            X = np.pad(X, ((0, self.seq_len - X.shape[0]), (0, 0)), 'constant', constant_values=(0,))

        # Add aux
        if self.p_aux:
            aux_values.append(np.log10(mad))
            aux_values.append(delta_t)

            aux_values = np.tile(aux_values, (self.seq_len, 1))
            X = np.concatenate((X, aux_values), axis=-1)

        # Convert X and mask from float64 to float32
        X = X.astype(np.float32)
        mask = mask.astype(np.float32)

        return X, mask

    def preprocess_spectra(self, spectra):
        wavelengths = spectra[:, 0]
        flux = spectra[:, 1]
        flux_err = spectra[:, 2]

        new_wavelengths = np.arange(3850, 9000, 2)
        flux = np.interp(new_wavelengths, wavelengths, flux)
        flux_err = np.interp(new_wavelengths, wavelengths, flux_err)

        mean = np.mean(flux)

        if self.s_mad:
            std = stats.median_abs_deviation(flux[flux != 0])
        else:
            std = np.std(flux)

        flux = (flux - mean) / std
        spectra = [flux]

        if self.s_err:
            if self.s_err_norm:
                flux_err = flux_err / std

            spectra.append(flux_err)

        if self.s_aux:
            aux_values = np.full_like(flux, np.log10(std))
            spectra.append(aux_values)

        spectra = np.vstack(spectra).astype(np.float32)

        return spectra

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        el = self.df.iloc[idx]
        label = self.target2id[el['target']]

        photometry = torch.zeros((self.seq_len, self.p_enc_in), dtype=torch.float32)
        photometry_mask = torch.zeros(self.seq_len, dtype=torch.float32)
        spectra = torch.zeros((self.s_enc_in, 2575), dtype=torch.float32)
        metadata = torch.zeros(len(self.meta_cols), dtype=torch.float32)

        if self.mode in ('photo', 'all', 'clip'):
            photometry = self.get_vlc(el['name'])
            photometry, photometry_mask = self.preprocess_lc(photometry, el['org_period'], list(el[self.photo_cols]))

        if self.mode in ('spectra', 'all', 'clip'):
            spectra = self.readLRSFits(os.path.join(self.lamost_spec_dir, el['spec_filename']))
            spectra = self.preprocess_spectra(spectra)

        if self.mode in ('meta', 'all', 'clip'):
            metadata = el[self.meta_cols].values.astype(np.float32)

        return photometry, photometry_mask, spectra, metadata, label

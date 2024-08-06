import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import stats
from io import BytesIO

from util.parallelzipfile import ParallelZipFile as ZipFile


class PSMDataset(Dataset):
    def __init__(self, config, split='train'):
        super(PSMDataset, self).__init__()

        self.split = split
        self.data_root = config['data_root']
        self.df = pd.read_csv(os.path.join(self.data_root, f'{config["file"]}_{self.split}_norm.csv'))
        self.reader_v = ZipFile(os.path.join(self.data_root, config['v_zip']))
        self.v_prefix = config['v_prefix']
        self.lamost_spec_dir = os.path.join(self.data_root, config['lamost_spec_dir'])
        self.meta_cols = config['meta_cols']

        self.min_samples = config['min_samples']
        self.max_samples = config['max_samples']
        self.classes = config['classes']
        self.seq_len = config['seq_len']
        self.phased = config['phased']
        self.aux = config['aux']

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

    def preprocess_lc(self, X, period):
        # Sort based on HJD
        sorted_indices = np.argsort(X[:, 0])
        X = X[sorted_indices]

        # Calculate min max before normalization
        log_abs_min = 0 if min(X[:, 1]) == 0 else np.log(abs(min(X[:, 1])))
        log_abs_max = np.log(abs(max(X[:, 1])))

        # Normalize
        mean = X[:, 1].mean()
        std = stats.median_abs_deviation(X[:, 1])
        X[:, 0] = (X[:, 0] - X[:, 0].min()) / (X[:, 0].max() - X[:, 0].min())
        X[:, 1] = (X[:, 1] - mean) / std
        X[:, 2] = X[:, 2] / std

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

        # Pad if needed and create mask
        mask = np.ones(self.seq_len)
        if X.shape[0] < self.seq_len:
            mask[X.shape[0]:] = 0
            X = np.pad(X, ((0, self.seq_len - X.shape[0]), (0, 0)), 'constant', constant_values=(0,))

        # Add aux
        if self.aux:
            log_abs_mean = np.log(abs(mean))
            log_std = np.log(std)

            aux = np.tile([log_abs_min, log_abs_max, log_abs_mean, log_std], (self.seq_len, 1))
            X = np.concatenate((X, aux), axis=-1)

        # Convert X and mask from float64 to float32
        X = X.astype(np.float32)
        mask = mask.astype(np.float32)

        return X, mask

    def preprocess_spectra(self, spectra):
        wavelengths, fluxes = spectra[:, 0], spectra[:, 1]
        fluxes = np.interp(np.arange(3850, 9000, 2), wavelengths, fluxes)
        fluxes = (fluxes - fluxes.mean()) / fluxes.std()
        fluxes = fluxes.reshape(1, -1).astype(np.float32)

        return fluxes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        el = self.df.iloc[idx]
        label = self.target2id[el['target']]

        photometry = self.get_vlc(el['name'])
        spectra = self.readLRSFits(os.path.join(self.lamost_spec_dir, el['spec_filename']))
        metadata = el[self.meta_cols].values.astype(np.float32)

        photometry, photometry_mask = self.preprocess_lc(photometry, el['period'])
        spectra = self.preprocess_spectra(spectra)

        return photometry, photometry_mask, spectra, metadata, label

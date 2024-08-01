import os
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import stats
from torch.utils.data import Dataset
from core.data.utils import readLRSFits, preprocess_spectra


class SpectraVDataset(Dataset):
    def __init__(self, data_root, lamost_spec_dir, spectra_v_file, split='train', classes=None, z_corr=False):
        self.data_root = data_root
        self.lamost_spec_dir = os.path.join(data_root, lamost_spec_dir)
        self.spectra_v_file = os.path.join(data_root, spectra_v_file)
        self.split = split
        self.z_corr = z_corr

        self.df = pd.read_csv(self.spectra_v_file)
        self.df = self.df[['edr3_source_id', 'variable_type', 'spec_filename']]

        if classes:
            self.df = self.df[self.df['variable_type'].isin(classes)]

        self._split()

        self.id2target = {i: x for i, x in enumerate(sorted(self.df['variable_type'].unique()))}
        self.target2id = {v: k for k, v in self.id2target.items()}
        self.num_classes = len(self.id2target)

    def _split(self):
        total_size = len(self.df)
        train_size = int(total_size * 0.7)
        val_size = int(total_size * 0.15)

        shuffled_df = self.df.sample(frac=1, random_state=42)

        if self.split == 'train':
            self.df = shuffled_df[:train_size]
        elif self.split == 'val':
            self.df = shuffled_df[train_size:train_size + val_size]
        elif self.split == 'test':
            self.df = shuffled_df[train_size + val_size:]
        else:
            self.df = shuffled_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        el = self.df.iloc[idx]
        variable_type, spec_filename = el['variable_type'], el['spec_filename']
        label = self.target2id[variable_type]

        # read spectra
        spectra = self.readLRSFits(os.path.join(self.lamost_spec_dir, spec_filename))
        original_wavelengths, fluxes = spectra[:, 0], spectra[:, 1]

        # interpolate
        wavelengths = np.arange(3850, 9000, 2)
        fluxes = np.interp(wavelengths, original_wavelengths, fluxes)

        # normalize
        fluxes = (fluxes - fluxes.mean()) / fluxes.std()

        # reshape so batches are [N, 1, (9000-3850)//2]
        fluxes = fluxes.reshape(1, -1).astype(np.float32)

        return fluxes, label

    def readLRSFits(self, filename):
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

        if self.z_corr:
            try:
                # correct for radial velocity of star
                redshift = head['Z']
            except Exception as e:
                print(e, 'Setting redshift to zero')
                redshift = 0.0

            wavelength = wavelength - redshift * wavelength

        return np.vstack((wavelength, specflux, ivar)).T


class VPSMDatasetV2Spectra(Dataset):
    def __init__(self, split='train', data_root='/home/mariia/AstroML/data/asassn/', file='preprocessed_data/full/spectra_and_v',
                 lamost_spec_dir='Spectra/v2', min_samples=None, max_samples=None, classes=None,
                 z_corr=False, random_seed=42):

        self.data_root = data_root
        self.df = pd.read_csv(os.path.join(data_root, f'{file}_{split}_norm.csv'))
        self.lamost_spec_dir = os.path.join(data_root, lamost_spec_dir)

        self.min_samples = min_samples
        self.max_samples = max_samples
        self.classes = classes
        self.z_corr = z_corr

        self.random_seed = random_seed
        np.random.seed(random_seed)

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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        el = self.df.iloc[idx]
        label = self.target2id[el['target']]

        spectra = readLRSFits(os.path.join(self.lamost_spec_dir, el['spec_filename']), self.z_corr)
        spectra = preprocess_spectra(spectra)

        return spectra, label

from astropy.io import fits
import json
import os
from io import BytesIO
from scipy import stats
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from util.parallelzipfile import ParallelZipFile as ZipFile
from util.preprocess_data import clip_outliers


METADATA_COLS = [
    'mean_vmag', 'amplitude', 'period', 'phot_g_mean_mag', 'e_phot_g_mean_mag', 'lksl_statistic',
    'rfr_score', 'phot_bp_mean_mag', 'e_phot_bp_mean_mag', 'phot_rp_mean_mag', 'e_phot_rp_mean_mag',
    'bp_rp', 'parallax', 'parallax_error', 'parallax_over_error', 'pmra', 'pmra_error', 'pmdec',
    'pmdec_error', 'j_mag', 'e_j_mag', 'h_mag', 'e_h_mag', 'k_mag', 'e_k_mag', 'w1_mag', 'e_w1_mag',
    'w2_mag', 'e_w2_mag', 'w3_mag', 'w4_mag', 'j_k', 'w1_w2', 'w3_w4', 'pm', 'ruwe'
]

CLASSES = ['CWA', 'CWB', 'DCEP', 'DCEPS', 'DSCT', 'EA', 'EB', 'EW',
           'HADS', 'M', 'ROT', 'RRAB', 'RRC', 'RRD', 'RVA', 'SR']


class VPSMDataset(Dataset):
    def __init__(self,
                 # general
                 data_root='/home/mariia/AstroML/data/asassn/', file='spectra_v_merged_fixed.csv', split='train',
                 min_samples=None, max_samples=None, classes=None, random_seed=42, verbose=True,

                 # photometry
                 v_zip='asassnvarlc_vband_complete.zip', v_prefix='vardb_files', seq_len=200,
                 phased=False, clip=False, aux=False,

                 # spectra
                 lamost_spec_dir='Spectra/v2', spectra_v_file='spectra_v_merged.csv', z_corr=False):

        self.data_root = data_root
        self.df = pd.read_csv(os.path.join(data_root, file))
        self.metadata_cols = METADATA_COLS
        self.all_cols = self.metadata_cols + ['edr3_source_id', 'variable_type', 'spec_filename', 'asassn_name']
        self.df = self.df[self.all_cols]

        self.reader_v = ZipFile(os.path.join(data_root, v_zip))
        self.v_prefix = v_prefix

        self.lamost_spec_dir = os.path.join(data_root, lamost_spec_dir)
        self.spectra_v_file = os.path.join(data_root, spectra_v_file)
        self.z_corr = z_corr

        self.seq_len = seq_len
        self.split = split
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.classes = classes
        self.phased = phased
        self.clip = clip
        self.aux = aux
        self.verbose = verbose

        self.random_seed = random_seed
        np.random.seed(random_seed)

        self._drop_nan()
        self._filter_classes()
        self._limit_samples()
        self._split()
        self._normalize_metadata()

        self.id2target = {i: x for i, x in enumerate(sorted(self.df['variable_type'].unique()))}
        self.target2id = {v: k for k, v in self.id2target.items()}
        self.num_classes = len(self.id2target)

    def _drop_nan(self):
        if self.verbose:
            print('Dropping nan values...', end=' ')

        self.df.dropna(axis=0, how='any', inplace=True)

        if self.verbose:
            print(f'Done. Left with {len(self.df)} rows.')

    def _filter_classes(self):
        if self.classes:
            if self.verbose:
                print(f'Leaving only classes: {self.classes}... ', end='')

            self.df = self.df[self.df['variable_type'].isin(self.classes)]

            if self.verbose:
                print(f'{len(self.df)} objects left.')

    def _limit_samples(self):
        if self.max_samples or self.min_samples:
            if self.verbose:
                print(
                    f'Removing objects that have more than {self.max_samples} or less than {self.min_samples} samples... ',
                    end='')

            value_counts = self.df['variable_type'].value_counts()

            if self.min_samples:
                classes_to_remove = value_counts[value_counts < self.min_samples].index
                self.df = self.df[~self.df['variable_type'].isin(classes_to_remove)]

            if self.max_samples:
                classes_to_limit = value_counts[value_counts > self.max_samples].index
                for class_type in classes_to_limit:
                    class_indices = self.df[self.df['variable_type'] == class_type].index
                    indices_to_keep = np.random.choice(class_indices, size=self.max_samples, replace=False)
                    self.df = self.df.drop(index=set(class_indices) - set(indices_to_keep))

            if self.verbose:
                print(f'{len(self.df)} objects left.')

    def _split(self):
        unique_ids = self.df['edr3_source_id'].unique()
        train_ids, temp_ids = train_test_split(unique_ids, test_size=0.2, random_state=self.random_seed)
        val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=self.random_seed)

        if self.split == 'train':
            self.df = self.df[self.df['edr3_source_id'].isin(train_ids)]
        elif self.split == 'val':
            self.df = self.df[self.df['edr3_source_id'].isin(val_ids)]
        elif self.split == 'test':
            self.df = self.df[self.df['edr3_source_id'].isin(test_ids)]
        else:
            print('Split is not train, val, or test. Keeping the whole dataset')

        self.df = self.df.reset_index()

        if self.verbose:
            print(f'{self.split} split is selected: {len(self.df)} objects left.')

    def _normalize_metadata(self):
        if self.split == 'train':
            self.scaler = StandardScaler()
            self.scaler.fit(self.df[self.metadata_cols])
            joblib.dump(self.scaler, 'scaler.pkl')
        else:
            self.scaler = joblib.load('scaler.pkl')

        self.df[self.metadata_cols] = self.scaler.transform(self.df[self.metadata_cols])

    def get_vlc(self, file_name):
        csv = BytesIO()
        file_name = file_name.replace(' ', '')
        data_path = f'{self.v_prefix}/{file_name}.dat'

        csv.write(self.reader_v.read(data_path))
        csv.seek(0)

        lc = pd.read_csv(csv, sep='\s+', skiprows=2, names=['HJD', 'MAG', 'MAG_ERR', 'FLUX', 'FLUX_ERR'],
                         dtype={'HJD': float, 'MAG': float, 'MAG_ERR': float, 'FLUX': float, 'FLUX_ERR': float})

        return lc[['HJD', 'FLUX', 'FLUX_ERR']].values

    def preprocess_lc(self, X, period):
        # 2 sort based on HJD
        sorted_indices = np.argsort(X[:, 0])
        X = X[sorted_indices]

        # 3 clip outliers
        # TODO double check clip outliers function
        if self.clip:
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
        if X.shape[0] > self.seq_len:
            start = np.random.randint(0, len(X) - self.seq_len)
            X = X[start:start + self.seq_len, :]

            # if self.split == 'train':
            #     start = np.random.randint(0, len(X) - self.seq_len)
            #     X = X[start:start + self.seq_len, :]
            # else:
            #     X = X[:self.seq_len, :]

        # 1 phase
        if self.phased:
            X = np.vstack(((X[:, 0] % period) / period, X[:, 1], X[:, 2])).T

        # pad if needed and create mask
        mask = np.ones(self.seq_len)
        if X.shape[0] < self.seq_len:
            mask[X.shape[0]:] = 0
            X = np.pad(X, ((0, self.seq_len - X.shape[0]), (0, 0)), 'constant', constant_values=(0,))

        # add aux
        if self.aux:
            log_abs_mean = np.log(abs(mean))
            log_std = np.log(std)

            # aux = np.tile([log_abs_min, log_abs_max, log_abs_mean, log_std, log_period], (self.seq_len, 1))
            aux = np.tile([log_abs_min, log_abs_max, log_abs_mean, log_std], (self.seq_len, 1))
            X = np.concatenate((X, aux), axis=-1)

        # 6 convert X and mask from float64 to float32
        X = X.astype(np.float32)
        mask = mask.astype(np.float32)

        return X, mask

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

    def preprocess_spectra(self, spectra):
        wavelengths, fluxes = spectra[:, 0], spectra[:, 1]
        fluxes = np.interp(np.arange(3850, 9000, 2), wavelengths, fluxes)
        fluxes = (fluxes - fluxes.mean()) / fluxes.std()
        fluxes = fluxes.reshape(1, -1).astype(np.float32)

        return fluxes

    def __len__(self):
        return len(self.df)

    def get_el(self, idx):
        el = self.df.iloc[idx]
        label = self.target2id[el['variable_type']]

        photometry = self.get_vlc(el['asassn_name'])
        photometry, photometry_mask = self.preprocess_lc(photometry, el['period'])

        spectra = self.readLRSFits(os.path.join(self.lamost_spec_dir, el['spec_filename']))
        spectra = self.preprocess_spectra(spectra)

        metadata = el[self.metadata_cols].values.astype(np.float32)

        return photometry, photometry_mask, spectra, metadata, label

    def __getitem__(self, idx):
        photometry, photometry_mask, spectra, metadata, label = self.get_el(idx)

        if np.random.rand() < 0.5:
            # negative sample
            idx2 = self.df[self.df['variable_type'] != self.id2target[label]].sample(n=1).index[0]
            y = 0
        else:
            # positive sample
            idx2 = self.df[(self.df.index != idx) &
                           (self.df['variable_type'] == self.id2target[label])].sample(n=1).index[0]
            y = 1

        photometry2, photometry_mask2, spectra2, metadata2, label2 = self.get_el(idx2)

        return (photometry, photometry_mask, spectra, metadata), (photometry2, photometry_mask2, spectra2, metadata2), y

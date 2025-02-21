import json
import os
from io import BytesIO
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from scipy import stats

from dev.util import ParallelZipFile as ZipFile
from dev.multimodal.dataset2 import VPSMDatasetV2


class VGDataset(Dataset):
    def __init__(self, data_root, vg_file, v_zip='asassnvarlc_vband_complete.zip', g_zip='g_band_lcs.zip',
                 v_prefix='vardb_files', g_prefix='g_band_lcs', scales='mean-mad',
                 seq_len=200, split='train', min_samples=None, max_samples=None, classes=None, random_seed=42,
                 phased=False, periodic=False, clip=False, verbose=True, aux=False):
        self.data_root = data_root
        self.df = pd.read_csv(os.path.join(data_root, vg_file))
        self.reader_v = ZipFile(os.path.join(data_root, v_zip))
        self.reader_g = ZipFile(os.path.join(data_root, g_zip))
        self.v_prefix = v_prefix
        self.g_prefix = g_prefix
        self.scales = scales

        self.seq_len = seq_len
        self.split = split
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.classes = classes
        self.phased = phased
        self.periodic = periodic
        self.clip = clip
        self.aux = aux
        self.verbose = verbose

        self.random_seed = random_seed
        np.random.seed(random_seed)

        self._filter_classes()
        self._filter_periodic()
        self._limit_samples()
        self._shuffle_data()
        self._split()

        self.id2target = {i: x for i, x in enumerate(sorted(self.df['target'].unique()))}
        self.target2id = {v: k for k, v in self.id2target.items()}
        self.num_classes = len(self.id2target)

    def _split(self):
        unique_ids = self.df['id'].unique()
        train_ids, temp_ids = train_test_split(unique_ids, test_size=0.2, random_state=self.random_seed)
        val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=self.random_seed)

        if self.split == 'train':
            self.df = self.df[self.df['id'].isin(train_ids)]
        elif self.split == 'val':
            self.df = self.df[self.df['id'].isin(val_ids)]
        elif self.split == 'test':
            self.df = self.df[self.df['id'].isin(test_ids)]
        else:
            print('Split is not train, val, or test. Keeping the whole dataset')

        if self.verbose:
            print(f'{self.split} split is selected: {len(self.df)} objects left.')

    def _filter_classes(self):
        if self.classes:
            if self.verbose:
                print(f'Leaving only classes: {self.classes}... ', end='')

            self.df = self.df[self.df['target'].isin(self.classes)]

            if self.verbose:
                print(f'{len(self.df)} objects left.')

    def _filter_periodic(self):
        if self.periodic:
            if self.verbose:
                print(f'Removing objects without periods... ', end='')

            self.df = self.df[~self.df['period'].isna()]

            if self.verbose:
                print(f'{len(self.df)} objects left.')

    def _limit_samples(self):
        if self.max_samples or self.min_samples:
            if self.verbose:
                print(
                    f'Removing objects that have more than {self.max_samples} or less than {self.min_samples} samples... ',
                    end='')

            value_counts = self.df['target'].value_counts()

            if self.min_samples:
                classes_to_remove = value_counts[value_counts < self.min_samples].index
                self.df = self.df[~self.df['target'].isin(classes_to_remove)]

            if self.max_samples:
                classes_to_limit = value_counts[value_counts > self.max_samples].index
                for class_type in classes_to_limit:
                    class_indices = self.df[self.df['target'] == class_type].index
                    indices_to_keep = np.random.choice(class_indices, size=self.max_samples, replace=False)
                    self.df = self.df.drop(index=set(class_indices) - set(indices_to_keep))

            if self.verbose:
                print(f'{len(self.df)} objects left.')

    def _shuffle_data(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def get_vlc(self, file_name):
        csv = BytesIO()
        data_path = f'{self.v_prefix}/{file_name}.dat'

        csv.write(self.reader_v.read(data_path))
        csv.seek(0)

        lc = pd.read_csv(csv, sep='\s+', skiprows=2, names=['HJD', 'MAG', 'MAG_ERR', 'FLUX', 'FLUX_ERR'],
                         dtype={'HJD': float, 'MAG': float, 'MAG_ERR': float, 'FLUX': float, 'FLUX_ERR': float})

        return lc[['HJD', 'FLUX', 'FLUX_ERR']].values

    def get_glc(self, file_name):
        csv = BytesIO()
        data_path = f'{self.g_prefix}/{file_name}.dat'

        csv.write(self.reader_g.read(data_path))
        csv.seek(0)

        lc = pd.read_csv(csv, sep='\s+', skiprows=2,
                         names=['HJD', 'camera', 'mag', 'mag_err', 'flux', 'flux_err', 'FWHM', 'IMAGE'],
                         dtype={'HJD': float, 'camera': 'object', 'mag': 'object', 'mag_err': 'object',
                                'flux': float, 'flux_err': float, 'FWHM': 'object', 'IMAGE': 'object'})

        return lc[['HJD', 'flux', 'flux_err']].values

    def preprocess(self, X, period, band):
        # 2 sort based on HJD
        sorted_indices = np.argsort(X[:, 0])
        X = X[sorted_indices]

        # 3 clip outliers
        # # TODO double check clip outliers function
        # if self.clip:
        #     t, y, y_err = X[:, 0], X[:, 1], X[:, 2]
        #     if len(t) > 20:
        #         t, y, y_err, _, _, _, _, _ = clip_outliers(t, y, y_err, measurements_in_flux_units=True,
        #                                                    initial_clip=(20, 5), clean_only=True)
        #     X = np.vstack((t, y, y_err)).T

        # Calculate min max before normalization
        log_abs_min = 0 if min(X[:, 1]) == 0 else np.log(abs(min(X[:, 1])))
        log_abs_max = np.log(abs(max(X[:, 1])))

        # 4 normalize
        if self.scales.endswith('.json'):
            with open(os.path.join(self.data_root, self.scales)) as f:
                s = json.load(f)
                mean, std = s[band]['mean'], s[band]['std']
        elif self.scales == 'mean-std':
            mean, std = X[:, 1].mean(), X[:, 1].std()
        elif self.scales == 'mean-mad':
            mean = X[:, 1].mean()
            std = stats.median_abs_deviation(X[:, 1])
        else:
            raise NotImplementedError(f'Unsupported scales {self.scales}')

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
            log_period = 0 if pd.isna(period) else np.log(period)

            # aux = np.tile([log_abs_min, log_abs_max, log_abs_mean, log_std, log_period], (self.seq_len, 1))
            aux = np.tile([log_abs_min, log_abs_max, log_abs_mean, log_std], (self.seq_len, 1))
            X = np.concatenate((X, aux), axis=-1)

        # 6 convert X and mask from float64 to float32
        X = X.astype(np.float32)
        mask = mask.astype(np.float32)

        return X, mask

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        el = self.df.iloc[idx]

        X = self.get_vlc(el['name']) if el['band'] == 'v' else self.get_glc(el['name'])
        X, mask = self.preprocess(X, el['period'], el['band'])
        y = self.target2id[el['target']]

        return X, mask, y


class VPSMDatasetV2Photo(Dataset):
    def __init__(self, split='train', data_root='/home/mariia/AstroML/data/asassn/', file='preprocessed_data/full/spectra_and_v',
                 v_zip='asassnvarlc_vband_complete.zip', v_prefix='vardb_files', min_samples=None, max_samples=None,
                 classes=None, seq_len=200, phased=False, clip=False, aux=False, random_seed=42):

        self.dataset = VPSMDatasetV2(split=split, data_root=data_root, file=file, v_zip=v_zip, v_prefix=v_prefix,
                                     min_samples=min_samples, max_samples=max_samples, classes=classes, seq_len=seq_len,
                                     phased=phased, clip=clip, aux=aux, random_seed=random_seed)
        self.id2target = self.dataset.id2target
        self.target2id = self.dataset.target2id
        self.num_classes = self.dataset.num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        photometry, photometry_mask, spectra, metadata, label = self.dataset[idx]
        return photometry, photometry_mask, label

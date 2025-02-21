import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from dev.util import ParallelZipFile as ZipFile
from dev.data.utils import preprocess_spectra, readLRSFits, preprocess_lc, get_vlc, add_noise, aug_metadata


METADATA_COLS = [
    'mean_vmag', 'amplitude', 'period', 'phot_g_mean_mag', 'e_phot_g_mean_mag', 'lksl_statistic',
    'rfr_score', 'phot_bp_mean_mag', 'e_phot_bp_mean_mag', 'phot_rp_mean_mag', 'e_phot_rp_mean_mag',
    'bp_rp', 'parallax', 'parallax_error', 'parallax_over_error', 'pmra', 'pmra_error', 'pmdec',
    'pmdec_error', 'j_mag', 'e_j_mag', 'h_mag', 'e_h_mag', 'k_mag', 'e_k_mag', 'w1_mag', 'e_w1_mag',
    'w2_mag', 'e_w2_mag', 'w3_mag', 'w4_mag', 'j_k', 'w1_w2', 'w3_w4', 'pm', 'ruwe'
]

CLASSES = ['CWA', 'CWB', 'DCEP', 'DCEPS', 'DSCT', 'EA', 'EB', 'EW',
           'HADS', 'M', 'ROT', 'RRAB', 'RRC', 'RRD', 'RVA', 'SR']


class VPSMDatasetV2(Dataset):
    def __init__(self, split='train', data_root='/home/mariia/AstroML/data/asassn/', file='preprocessed_data/full/spectra_and_v',
                 v_zip='asassnvarlc_vband_complete.zip', v_prefix='vardb_files', lamost_spec_dir='Spectra/v2',
                 min_samples=None, max_samples=None, classes=None, seq_len=200, phased=False, clip=False, aux=False,
                 z_corr=False, noise=False, noise_coef=1, random_seed=42):

        self.split = split
        self.data_root = data_root
        self.df = pd.read_csv(os.path.join(data_root, f'{file}_{split}_norm.csv'))
        self.reader_v = ZipFile(os.path.join(data_root, v_zip))
        self.v_prefix = v_prefix
        self.lamost_spec_dir = os.path.join(data_root, lamost_spec_dir)
        self.metadata_cols = METADATA_COLS

        self.min_samples = min_samples
        self.max_samples = max_samples
        self.classes = classes
        self.seq_len = seq_len
        self.phased = phased
        self.clip = clip
        self.aux = aux
        self.z_corr = z_corr
        self.noise = noise
        self.noise_coef = noise_coef

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

        photometry = get_vlc(el['name'], self.v_prefix, self.reader_v)
        spectra = readLRSFits(os.path.join(self.lamost_spec_dir, el['spec_filename']), self.z_corr)
        metadata = el[self.metadata_cols].values.astype(np.float32)

        if self.noise:
            photometry = add_noise(photometry, noise_coef=self.noise_coef)
            spectra = add_noise(spectra, noise_coef=self.noise_coef)
            metadata = aug_metadata(metadata, noise_coef=self.noise_coef)

        crop = 'random' if self.split == 'train' else 'center'
        photometry, photometry_mask = preprocess_lc(photometry, el['period'], self.clip, self.seq_len, self.phased,
                                                    self.aux, crop=crop)
        spectra = preprocess_spectra(spectra)

        return photometry, photometry_mask, spectra, metadata, label

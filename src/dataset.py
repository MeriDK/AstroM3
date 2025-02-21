from torch.utils.data import Dataset
import numpy as np
from scipy import stats


class HFPSMDataset(Dataset):
    def __init__(self, ds, classes, seq_len, mode='all', split='train'):
        """
        Custom PyTorch dataset for handling multi-modal astronomical data.

        Args:
            ds (datasets.DatasetDict): The Hugging Face dataset dictionary.
            classes (list): List of unique class labels.
            seq_len (int): Sequence length for photometry data.
            mode (str): Determines which modalities are processed ('all', 'photo', 'spectra', 'meta' 'clip').
            split (str): Dataset split to use ('train', 'validation', 'test').
        """
        super(HFPSMDataset, self).__init__()

        self.ds = ds[split]
        self.ds = self.ds.with_format('numpy')
        self.seq_len = seq_len
        self.mode = mode
        self.split = split

        # Create mappings between class labels and numerical indices
        self.id2label = {i: x for i, x in enumerate(sorted(classes))}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.num_classes = len(classes)

    def preprocess_lc(self, X, aux_values):
        """
        Preprocess photometry (light curve) data.

        Steps:
        - Remove duplicate time entries.
        - Sort by Heliocentric Julian Date (HJD).
        - Normalize flux and flux error using mean and median absolute deviation (MAD).
        - Scale time values between 0 and 1.
        - Trim or pad sequences to maintain consistent sequence length.
        - Append auxiliary features (log MAD and time span delta_t).
        - Convert data to float32.

        Args:
            X (numpy.ndarray): Original photometry data of shape (N, 3) where columns are (HJD, flux, flux_error).
            aux_values (numpy.ndarray): Auxiliary metadata features.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: Preprocessed photometry data and corresponding mask.
        """
        # Remove duplicate entries
        X = np.unique(X, axis=0)

        # Sort based on HJD
        sorted_indices = np.argsort(X[:, 0])
        X = X[sorted_indices]

        # Normalize flux and flux error
        mean = X[:, 1].mean()
        mad = stats.median_abs_deviation(X[:, 1])
        X[:, 1] = (X[:, 1] - mean) / mad
        X[:, 2] = X[:, 2] / mad

        # Compute delta_t (time span of the light curve in years)
        delta_t = (X[:, 0].max() - X[:, 0].min()) / 365

        # Scale time from 0 to 1
        X[:, 0] = (X[:, 0] - X[:, 0].min()) / (X[:, 0].max() - X[:, 0].min())

        # Trim sequence if longer than seq_len
        if X.shape[0] > self.seq_len:
            if self.split == 'train':  # Random crop for training
                start = np.random.randint(0, len(X) - self.seq_len)
            else:  # Center crop for validation/test
                start = (len(X) - self.seq_len) // 2

            X = X[start:start + self.seq_len, :]

        # Pad sequences if they are shorter than seq_len
        mask = np.ones(self.seq_len)    # Mask to indicate real vs. padded values
        if X.shape[0] < self.seq_len:
            mask[X.shape[0]:] = 0   # Zero mask for padded values
            X = np.pad(X, ((0, self.seq_len - X.shape[0]), (0, 0)), 'constant', constant_values=(0,))

        # Add MAD and delta_t to auxiliary metadata features
        aux_values = np.concatenate((aux_values, [np.log10(mad), delta_t]))

        # Add auxiliary features to the sequence
        aux_values = np.tile(aux_values, (self.seq_len, 1))
        X = np.concatenate((X, aux_values), axis=-1)

        # Convert to float32
        X = X.astype(np.float32)
        mask = mask.astype(np.float32)

        return X, mask

    @staticmethod
    def preprocess_spectra(spectra):
        """
        Preprocess spectral data.

        Steps:
        - Interpolate flux and flux error to a fixed wavelength grid (3850 to 9000 Å).
        - Normalize flux using mean and median absolute deviation (MAD).
        - Append MAD as an auxiliary feature.
        - Convert to float32.

        Args:
            spectra (numpy.ndarray): Original spectra data of shape (N, 3) where columns are (wavelength, flux, flux_error).

        Returns:
            numpy.ndarray: Preprocessed spectral data of shape (3, num_wavelengths).
        """
        wavelengths = spectra[:, 0]
        flux = spectra[:, 1]
        flux_err = spectra[:, 2]

        # Interpolate flux and flux error onto a fixed grid
        new_wavelengths = np.arange(3850, 9000, 2)
        flux = np.interp(new_wavelengths, wavelengths, flux)
        flux_err = np.interp(new_wavelengths, wavelengths, flux_err)

        # Normalize flux and flux error
        mean = np.mean(flux)
        mad = stats.median_abs_deviation(flux[flux != 0])

        flux = (flux - mean) / mad
        flux_err = flux_err / mad
        aux_values = np.full_like(flux, np.log10(mad))  # Store MAD as an auxiliary feature

        # Stack processed data into a single array and convert to float32
        spectra = np.vstack([flux, flux_err, aux_values]).astype(np.float32)

        return spectra

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.ds)

    def __getitem__(self, idx):
        """
        Retrieves an individual sample from the dataset.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, int]:
            - photometry (numpy.ndarray): Processed light curve data.
            - photometry_mask (numpy.ndarray): Mask indicating real vs. padded values in photometry.
            - spectra (numpy.ndarray): Processed spectral data.
            - metadata (numpy.ndarray): Auxiliary metadata.
            - label (int): Integer class label.
        """
        el = self.ds[idx]

        # Get the label (convert from string to integer)
        label = self.label2id[el['label']]

        # Convert metadata features to NumPy array
        metadata = np.array(list(el['metadata']['meta_cols'].values()), dtype=np.float32)

        # Convert photometry metadata features to NumPy array
        photo_cols = np.array(list(el['metadata']['photo_cols'].values()))

        # Create placeholder arrays (ensures correct batching even if preprocessing is skipped)
        photometry = np.zeros((self.seq_len, len(photo_cols) + 5), dtype=np.float32)
        photometry_mask = np.ones(self.seq_len, dtype=np.float32)
        spectra = np.zeros((3, 2575), dtype=np.float32)

        # Preprocess photometry only if required by mode
        if self.mode in ('photo', 'all', 'clip'):
            photometry, photometry_mask = self.preprocess_lc(el['photometry'], photo_cols)

        # Preprocess spectra only if required by mode
        if self.mode in ('spectra', 'all', 'clip'):
            spectra = self.preprocess_spectra(el['spectra'])

        return photometry, photometry_mask, spectra, metadata, label

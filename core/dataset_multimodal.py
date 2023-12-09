import os
import logging
import tarfile
from io import BytesIO

import numpy as np
import pandas as pd
from filelock import FileLock
from astropy.io import fits

import torch
from torch.utils.data import Dataset

from util.parallelzipfile import ParallelZipFile as ZipFile
from util.preprocess_data import clip_outliers

pd.options.mode.chained_assignment = None
logging.captureWarnings(True)


# These are ASAS-SN data-specific parameters.
raw_data_files = {
    "v": {
        "tab": "asassn_catalog_full.csv",
        "lcs": "asassnvarlc_vband_complete.zip",
        "prefix": "vardb_files/",
        "filekey": "asassn_name",
        "keyfill": "",
        "flux_headers": ["HJD", "FLUX", "FLUX_ERR"],
        "mag_headers": ["HJD", "MAG", "MAG_ERR"],
    },
    "g": {
        "tab": "asassn_variables_x.csv",
        "lcs": "g_band_lcs-001.tar",
        "prefix": "g_band_lcs/",
        "filekey": "ID",
        "keyfill": "_",
        "flux_headers": ["HJD", "flux", "flux_err"],
        "mag_headers": ["HJD", "mag", "mag_err"],
    },
}

metadata_cols = {
    "g": [
        "Mean_gmag",
        "Amplitude",
        "Period",
        "parallax",
        "parallax_error",
        "parallax_over_error",
        "pm",
        "pmra",
        "pmra_error",
        "pmdec",
        "pmdec_error",
        "ruwe",
        "phot_g_mean_mag",
        "e_phot_g_mean_mag",
        "phot_bp_mean_mag",
        "e_phot_bp_mean_mag",
        "phot_rp_mean_mag",
        "e_phot_rp_mean_mag",
        "bp_rp",
        "FUVmag",
        "e_FUVmag",
        "NUVmag",
        "e_NUVmag",
        "W1mag",
        "W2mag",
        "W3mag",
        "W4mag",
        "Jmag",
        "Hmag",
        "Kmag",
        "e_W1mag",
        "e_W2mag",
        "e_W3mag",
        "e_W4mag",
        "e_Jmag",
        "e_Hmag",
        "e_Kmag",
    ],
    "v": [
        "mean_vmag",
        "amplitude",
        "period",
        "phot_g_mean_mag",
        "e_phot_g_mean_mag",
        "lksl_statistic",
        "rfr_score",
        "phot_bp_mean_mag",
        "e_phot_bp_mean_mag",
        "phot_rp_mean_mag",
        "e_phot_rp_mean_mag",
        "bp_rp",
        "parallax",
        "parallax_error",
        "parallax_over_error",
        "pmra",
        "pmra_error",
        "pmdec",
        "pmdec_error",
        "j_mag",
        "e_j_mag",
        "h_mag",
        "e_h_mag",
        "k_mag",
        "e_k_mag",
        "w1_mag",
        "e_w1_mag",
        "w2_mag",
        "e_w2_mag",
        "w3_mag",
        "e_w3_mag",
        "w4_mag",
        "e_w4_mag",
        "j_k",
        "w1_w2",
        "w3_w4",
        "apass_vmag",
        "e_apass_vmag",
        "apass_bmag",
        "e_apass_bmag",
        "apass_gpmag",
        "e_apass_gpmag",
        "apass_rpmag",
        "e_apass_rpmag",
        "apass_ipmag",
        "e_apass_ipmag",
        "FUVmag",
        "e_FUVmag",
        "NUVmag",
        "e_NUVmag",
        "pm",
        "ruwe",
    ],
}

bookkeeping_cols = {
    "v": [
        "id",
        "source_id",
        "asassn_name",
        "other_names",
        "raj2000",
        "dej2000",
        "l",
        "b",
        "epoch_hjd",
        "gdr2_id",
        "allwise_id",
        "apass_dr9_id",
        "edr3_source_id",
        "galex_id",
        "tic_id",
    ],
    "g": [
        "ID",
        "RAJ2000",
        "DEJ2000",
        "l",
        "b",
        "EpochHJD",
        "EDR3_source_id",
        "GALEX_ID",
        "TIC_ID",
        "AllWISE_ID",
        "ML_probability",
        "class_probability",
    ],
}

# outcome columns
target_cols = {"g": ["ML_classification"], "v": ["variable_type"]}

# These are the columns that report the (measured) period of the sources.
period_col = {"g": "Period", "v": "period"}

# What are the columns in the two datafiles that can be used to uniquely identify a source?
merge_key = {"g": "EDR3_source_id", "v": "edr3_source_id"}


def collate_fn(
    batch, data_keys=["lcs", "metadata", "spectra", "classes"], fill_value=-9999
):
    """
    return a list of tensors with data and masks, for this batch
    """
    data = []
    masks = []
    for k in data_keys:
        if k in ["metadata", "classes"]:
            key_batch = [torch.Tensor(t[k]) for t in batch]
            kb = torch.nn.utils.rnn.pad_sequence(
                key_batch, batch_first=True, padding_value=fill_value
            ).squeeze(1)

        if k in ["spectra", "lcs"]:
            spec = []
            max_spec = max([len(obj[k][0]) for obj in batch])
            book = []
            for on, obj in enumerate(batch):
                obj_spec = obj[k][0]
                s = []
                ls = []
                for i in range(len(obj_spec)):
                    shape = list(obj_spec[i].shape)
                    # check for possible empty data vector
                    if shape[0] == 0:
                        shape[0] += 1
                        v = fill_value * torch.ones(shape)
                    else:
                        v = obj_spec[i]
                    s.append(torch.Tensor(v))
                    ls.append(s[-1].shape)

                if len(obj_spec) > 0:
                    last_shape = s[-1].shape
                else:
                    last_shape = (1, 1)
                    book.append((on, last_shape))

                for j in range(i + 1, max_spec):
                    s.append(fill_value * torch.ones(last_shape))
                    ls.append(s[-1].shape)

                spec.append(
                    torch.nn.utils.rnn.pad_sequence(s, padding_value=fill_value)
                )

            try:
                max_shape = tuple(np.max(np.array([s.shape for s in spec]), axis=0))
            except Exception as err:
                raise err from None

            for i, s in enumerate(spec):
                if s.shape[0] <= 1:
                    book.append(("here", i, s.shape))
                    spec[i] = fill_value * torch.ones(max_shape)
            try:
                kb = torch.nn.utils.rnn.pad_sequence(spec, padding_value=fill_value)
            except Exception as err:
                for i, ss in enumerate(spec):
                    print(i, ss.shape, type(ss))
                raise err from None

            # return a tensor like batch_size, lcs/spectrum_number, [nu, flux, fluxerr],
            # value
            kb = torch.permute(kb, (1, 2, 3, 0))

        # make a mask and fill in.
        isnan = torch.isnan(kb)
        key_mask = torch.where(kb == fill_value, True, False) | isnan
        kb[key_mask] = fill_value
        data.append(kb)
        masks.append(key_mask)

    return data, masks


class ASASSNVarStarDataset(Dataset):
    def __init__(
        self,
        data_root,
        prediction_length,
        mode="train",
        use_errors=True,
        use_bands=["v", "g"],
        merge_type="inner",
        lc_type="flux",
        rng=None,
        return_phased=True,
        lock_phase=None,
        clean=True,
        recalc_period=False,
        verbose=False,
        lamost_spec_file="Spectra/lamost_spec.csv",
        lamost_spec_dir="Spectra/v2",
        only_sources_with_spectra=True,
        prime=True,
        initial_clean_clip=[20, 5],
        only_periodic=True,
        period_cache="periods.csv",
        return_items_as_list=False,
        fill_value=-9999,
        lockfile="g_band_zip.lock",
    ):
        """
        Multi-modal ASAS-SN dataset of variable stars

        data_root = root directory of the data (Path object)
        prediction_length = max length of the prediction window (in numbers of time points)
        mode = bookkeeping column to use to split the data into train/val/test
        use_errors = use the reported errors in the light curves
        use_bands = which bands to use. Must be a list of one or two bands.
                    Default: ["v", "g"]
        merge_type = SQL style for how to merge the two bands. Default: "inner"
        lc_type = "flux" or "mag". Default: "flux"
        return_phased = return the phased light curve instead of the original light curves
        lock_phase = if not None, use the period from this band to phase both bands
        clean = remove outliers from the light curves
        recalc_period = refit the L-S to find the best period and save the period to disk
        verbose = print out more information
        lamost_spec_file = filename of the LAMOST spectra csv
        lamost_spec_dir = directory of the LAMOST spectra
        only_sources_with_spectra = only keep sources that have spectra
        prime = prime the tarballs by doing an initial scan
        initial_clean_clip = initial outlier clip parameters (used if `clean` is True)
        only_periodic = only keep sources that are marked as periodic
        period_cache = filename of the period cache csv (used if `recalc_period` is True)
        return_items_as_list = return the items as a list instead of a dictionary
        fill_value = value to use for missing data
        lockfile = lockfile for the zip files
        """
        self.data_root = data_root
        self.prediction_length = prediction_length
        self.use_errors = use_errors
        self.return_phased = return_phased
        self.recalc_period = recalc_period
        self.clean = clean
        self.verbose = verbose
        self.use_bands = use_bands
        if not isinstance(use_bands, list):
            raise Exception("`use_bands` must be a list like ['v', 'g']")
        self.merge_type = merge_type
        self.lc_type = lc_type
        self.lamost_spec_file = lamost_spec_file
        self.lamost_spec_dir = lamost_spec_dir
        self.only_sources_with_spectra = only_sources_with_spectra
        self.lock_phase = lock_phase
        self.initial_clean_clip = initial_clean_clip
        self.only_periodic = only_periodic
        self.period_cache = period_cache
        self.return_items_as_list = return_items_as_list
        self.fill_value = fill_value
        self.lock = FileLock(lockfile)

        # set the random seed if need be
        if rng is None:
            self.rng = np.random.default_rng(42)
        else:
            self.rng = rng

        self._check_and_open_data_files()
        self._merge_bands()

        if self.recalc_period and self.period_cache is not None:
            fname = self.data_root / self.period_cache
            try:
                self.period_recalc_df = pd.read_csv(fname)
                if self.verbose:
                    print("Opened period cache file")
            except Exception:
                self.period_recalc_df = pd.DataFrame(columns=["id", "p", "band"])

        if prime:
            self._prime()

        # shuffle
        self.df = self.df.sample(frac=1, random_state=self.rng)

    def __del__(self):
        if self.recalc_period and self.period_cache is not None:
            # read in what's on disk just in case it changed by another process.
            if (self.data_root / self.period_cache).exists():
                tmp = pd.read_csv(self.data_root / self.period_cache)
                merged = pd.concat(
                    [tmp, self.period_recalc_df], ignore_index=True
                ).drop_duplicates(keep="last", ignore_index=True)
            else:
                merged = self.period_recalc_df
            merged.to_csv(self.data_root / self.period_cache, index=False)
            if self.verbose:
                print("Wrote period cache file.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if not isinstance(idx, list):
            idx = [idx]

        sources = self.df.iloc[idx]

        return_dict = {}

        # bookkeeping - not to be used in learning
        return_dict["bookkeeping_data"] = sources[self.bookkeeping_all].values.tolist()

        # classes
        targets = sources[self.target_all].values
        for k, v in self.target_lookup.items():
            targets[targets == v] = k

        return_dict["classes"] = targets.astype(np.int32)

        # light curves
        return_dict["lcs"] = self.get_light_curves(sources)

        # spectra
        return_dict["spectra"] = self.get_spectra(sources)

        if self.return_phased:
            phased_all = []
            i = 0
            Ps = {band: [] for band in self.use_bands}
            for ind, source in sources.iterrows():
                source_id = source[merge_key[self.use_bands[0]]]
                band_periods = {
                    band: source[period_col[band]] for band in self.use_bands
                }
                used_cache = {band: False for band in self.use_bands}
                for band_num, band in enumerate(self.use_bands):
                    recalc = self.recalc_period
                    if len(self.use_bands) > 1:
                        other_band = list(
                            filter(lambda x: x != band, self.use_bands.copy())
                        )[0]
                    else:
                        other_band = band
                    if self.lock_phase is not None:
                        if band == self.lock_phase:
                            if np.isnan(band_periods[band]):
                                # switch to the other band
                                band_periods[band] = band_periods[other_band]
                        else:
                            band_periods[band] = band_periods[other_band]
                            used_cache[band] = True
                            continue

                    # failsafe: if the period is nan then recalc
                    if np.isnan(band_periods[band]):
                        recalc = True

                    if recalc:
                        # print(f"{band} P={band_periods[band]} ",flush=True, end="")
                        lcs = return_dict["lcs"][i][band_num]
                        t, y, dy = lcs[:, 0], lcs[:, 1], lcs[:, 2]
                        do_calc = True
                        if self.period_cache is not None and self.recalc_period:
                            tmp = self.period_recalc_df.query(
                                f"(id == '{source_id}') & (band == '{band}')"
                            )["p"].values
                            if len(tmp) > 0:
                                P_best = tmp[-1]
                                do_calc = False
                                used_cache[band] = True
                                if self.verbose:
                                    print(" got period from cache ")
                            else:
                                do_calc = True
                        if do_calc and len(t) > 20:
                            t, y, dy, _, P_best, _, _, _ = clip_outliers(
                                t,
                                y,
                                dy,
                                measurements_in_flux_units=self.lc_type == "flux",
                                initial_clip=self.initial_clean_clip,
                                clean_only=False,
                                max_iter=2,
                            )
                        # print(f" ({P_best}) ", flush=True, end="")
                        # we may have found a harmonic. Fix it.
                        if (
                            abs((P_best - band_periods[band]) / band_periods[band])
                            > 0.01
                            and P_best < band_periods[band]
                        ):
                            P_best *= 2
                        elif (
                            abs(P_best - 1) < 0.01
                            or abs(P_best - 2) < 0.01
                            or abs(P_best - 3) < 0.01
                        ):
                            # keep the original P if we're too close to 1 or 2 or 3 day Periods
                            P_best = band_periods[band]
                            if np.isnan(P_best):
                                P_best = band_periods[other_band]

                        band_periods[band] = P_best
                        # print(f"P={band_periods[band]} ",flush=True)

                # now that we have the period, fold
                phased = []
                for band_num, band in enumerate(self.use_bands):
                    if len(self.use_bands) > 1:
                        other_band = list(
                            filter(lambda x: x != band, self.use_bands.copy())
                        )[0]
                    else:
                        other_band = band
                    P_best = (
                        band_periods[band]
                        if band == self.lock_phase
                        else band_periods[other_band]
                    )
                    lcs = return_dict["lcs"][i][band_num]
                    t, y, dy = lcs[:, 0], lcs[:, 1], lcs[:, 2]
                    phased.append(np.vstack(((t % P_best) / P_best, y, dy)).T)
                    Ps[band].append(P_best)
                    if (
                        self.period_cache is not None
                        and self.recalc_period
                        and not used_cache[band]
                    ):
                        # append the recalc periods to the cache
                        self.period_recalc_df.loc[len(self.period_recalc_df)] = [
                            source_id,
                            P_best,
                            band,
                        ]

                phased_all.append(phased)
                i += 1

            # update the sources table with the new periods
            sources[period_col[band]] = Ps[band]
            return_dict["phased"] = phased_all

        # metadata
        return_dict["metadata"] = sources[self.metadata_all].values

        if not self.return_items_as_list:
            if self.return_phased:
                return_dict["lcs"] = return_dict["phased"]
            return return_dict
        else:
            if self.return_phased:
                lcs = return_dict["phased"]
            else:
                lcs = return_dict["lcs"]
            return (
                lcs,
                return_dict["metadata"],
                return_dict["spectra"],
                return_dict["classes"],
                return_dict["bookkeeping_data"],
            )

    def _prime(self):
        """This takes about 1 minute. After that getting light curves is fast"""
        if self.verbose:
            print("Priming tarballs by doing initial scan...", flush=True, end="")
        self.get_light_curves(self.df.sample(random_state=self.rng))
        if self.verbose:
            print("done.", flush=True)

    def get_light_curves(self, rows):
        """Given df row(s), return the light curves as numpy arrays"""

        light_curves = []
        for ind, row in rows.iterrows():
            row_lc = []
            for band in self.use_bands:
                if pd.isna(row[raw_data_files[band]["filekey"]]):
                    name = "missing"
                else:
                    name = row[raw_data_files[band]["filekey"]].replace(
                        " ", raw_data_files[band]["keyfill"]
                    )
                if self.lcs[band][1] == "tar" and band == "g":
                    try:
                        f = self.lcs[band][0].getmember(
                            f"{raw_data_files[band]['prefix']}{name}.dat"
                        )
                    except KeyError:
                        print(f"Cannot find {raw_data_files[band]['prefix']}{name}")
                        row_lc.append(self.fill_value * np.ones((10, 3)))
                    try:
                        with self.lock:
                            csv_file = pd.read_csv(
                                self.lcs[band][0].extractfile(f),
                                sep="\t",
                                on_bad_lines="warn",
                            )
                            rez = csv_file[
                                raw_data_files[band][f"{self.lc_type}_headers"]
                            ].values
                            row_lc.append(rez)
                    except Exception as err:
                        print(err)
                        row_lc.append(self.fill_value * np.ones((10, 3)))

                elif self.lcs[band][1] == "zip" and band == "v":
                    try:
                        csv = BytesIO()
                        csv.write(
                            self.lcs[band][0].read(
                                f"{raw_data_files[band]['prefix']}{name}.dat"
                            )
                        )
                        csv.seek(0)
                        row_lc.append(
                            pd.read_csv(csv, sep=" ", skiprows=1)[
                                raw_data_files[band][f"{self.lc_type}_headers"]
                            ].values
                        )
                    except KeyError:
                        print(f"Cannot find {raw_data_files[band]['prefix']}{name}")
                        row_lc.append(self.fill_value * np.ones((10, 3)))

                else:
                    raise Exception("Dont know how to get data from such files")

                t, y, yerr = row_lc[-1][:, 0], row_lc[-1][:, 1], row_lc[-1][:, 2]
                if self.clean and len(row_lc[-1][:, 0]) > 20:
                    t, y, yerr, _, _, _, _, _ = clip_outliers(
                        t,
                        y,
                        yerr,
                        measurements_in_flux_units=self.lc_type == "flux",
                        initial_clip=self.initial_clean_clip,
                        clean_only=True,
                    )
                row_lc[-1] = np.vstack((t, y, yerr)).T

            light_curves.append(row_lc)
        return light_curves

    def get_spectra(self, rows):
        """Given df row(s), return the spectra as numpy arrays"""
        spectra = []
        for ind, row in rows.iterrows():
            row_spectra = []
            rowid = row[merge_key[self.use_bands[0]]]
            rez = self.spec_df.query(f"edr3_source_id == '{rowid}'")
            if len(rez) == 0:
                spectra.append([])
                continue
            for si, spect in rez.iterrows():
                filename = (
                    self.data_root / self.lamost_spec_dir / spect["spec_filename"]
                )
                if os.path.exists(filename):
                    row_spectra.append(self._readLRSFits(filename))
            spectra.append(row_spectra)
        return spectra

    def _merge_bands(self):
        """
        Merge the two bands into a single dataframe
        """
        if len(self.use_bands) == 1:
            self.df = self.dfs[self.use_bands[0]]
        elif (
            len(self.use_bands) == 2 and "v" in self.use_bands and "g" in self.use_bands
        ):
            if self.verbose:
                print("Merging bands...", flush=True, end="")
            self.df = self.dfs["v"].merge(
                self.dfs["g"],
                how=self.merge_type,
                left_on=merge_key["v"],
                right_on=merge_key["g"],
                suffixes=("_vband", "_gband"),
            )
            if self.verbose:
                print("done.", flush=True)

        else:
            raise Exception("Dont know how to merge these bands")

        if self.only_periodic:
            try:
                self.df = self.df[self.df["periodic"]]
            except Exception:
                print("No `periodic` column in the dataframe. Proceeding.")

        target_cols
        # make a list of columns to save for bookkeeping
        df_cols = self.df.columns
        self.bookkeeping_all = []
        for band in self.use_bands:
            for col in bookkeeping_cols[band]:
                if col in df_cols:
                    self.bookkeeping_all.append(col)
                elif f"{col}_{band}band" in df_cols:
                    self.bookkeeping_all.append(f"{col}_{band}band")

        self.metadata_all = []
        for band in self.use_bands:
            for col in metadata_cols[band]:
                if col in df_cols:
                    self.metadata_all.append(col)
                elif f"{col}_{band}band" in df_cols:
                    self.metadata_all.append(f"{col}_{band}band")

        self.target_all = []
        for band in self.use_bands:
            self.target_all += target_cols[band]

        targets = self.df[[target_cols[band][0] for band in self.use_bands]].values.ravel()
        targets = targets[~pd.isnull(targets)]

        self.target_lookup = {
            i: x
            for i, x in enumerate(
                np.unique(targets))
        }

    def _readLRSFits(self, filename, z_corr=True):
        """
        Read LAMOST fits file
          adapted from https://github.com/fandongwei/pylamost
        """

        hdulist = fits.open(filename)
        len_list = len(hdulist)
        if len_list == 1:
            head = hdulist[0].header
            scidata = hdulist[0].data
            coeff0 = head["COEFF0"]
            coeff1 = head["COEFF1"]
            pixel_num = head["NAXIS1"]
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

        if z_corr:
            try:
                # correct for radial velocity of star
                redshift = head["Z"]
            except Exception:
                redshift = 0.0
            wavelength = wavelength - redshift * wavelength

        return np.vstack((wavelength, specflux, ivar)).T

    def _check_and_open_data_files(self):
        """
        Check that the data files exist and open them
        """
        if len(self.use_bands) == 0:
            raise Exception("Need a least one bandpass to use")
        if not os.path.isdir(self.data_root):
            raise Exception(f"{self.data_root} is not a valid data directory.")

        dfs = {}
        lcs = {}
        for band in self.use_bands:
            if not os.path.exists(self.data_root / raw_data_files[band]["tab"]):
                raise Exception(f"Missing tabular data for {band}.")
            if not os.path.exists(self.data_root / raw_data_files[band]["lcs"]):
                raise Exception(f"Missing light curve data for {band}.")

            if self.verbose:
                print(f"Opening {band} data files...", flush=True, end="")

            dfs[band] = pd.read_csv(self.data_root / raw_data_files[band]["tab"])
            if self.verbose:
                print(f" Found {len(dfs[band])} sources. ", end="")

            lcs_file_type = "".join(
                (self.data_root / raw_data_files[band]["lcs"]).suffixes
            )

            if lcs_file_type == ".zip":
                lcs[band] = (
                    ZipFile(self.data_root / raw_data_files[band]["lcs"]),
                    "zip",
                )
            elif lcs_file_type in [".tar.gz", ".tgz", ".tar"]:
                lcs[band] = (
                    tarfile.open(self.data_root / raw_data_files[band]["lcs"], "r"),
                    "tar",
                )
            else:
                raise Exception(
                    f"Dont know how to open {self.data_root / raw_data_files[band]['lcs']}"
                )

            if self.verbose:
                print("done.", flush=True)

        if (
            self.lamost_spec_file is not None
            and (self.data_root / self.lamost_spec_file).exists()
        ):
            if self.verbose:
                print("Opening spectra csv...", flush=True, end="")
            self.spec_df = pd.read_csv(self.data_root / self.lamost_spec_file)
            if self.verbose:
                print("done.", flush=True)
            if self.only_sources_with_spectra:
                sources_with_spectra = pd.unique(self.spec_df["edr3_source_id"])
                for band in self.use_bands:
                    if self.verbose:
                        print(
                            f"Keeping only {band} band sources with spectra...",
                            flush=True,
                            end="",
                        )
                    dfs[band] = dfs[band][
                        dfs[band][merge_key[band]].isin(sources_with_spectra)
                    ]
                    if self.verbose:
                        print(f" Left with {len(dfs[band])} sources. ", end="")
                        print("done.", flush=True)
        self.lcs = lcs
        self.dfs = dfs

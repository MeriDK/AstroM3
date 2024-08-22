import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


ORIGINAL_METADATA_COLS = [
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
    "w4_mag",
    "j_k",
    "w1_w2",
    "w3_w4",
    "pm",
    "ruwe",
    "l",
    "b",
]

DROP_METADATA_COLS = [
    "mean_vmag",
    "amplitude",
    "period",
    "phot_g_mean_mag",
    "lksl_statistic",
    "rfr_score",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "j_mag",
    "h_mag",
    "k_mag",
    "w1_mag",
    "w2_mag",
    "w3_mag",
    "w4_mag",
    "l",
    "b",
]

METADATA_FUNC = {
    "abs": [
        "mean_vmag",
        "phot_g_mean_mag",
        "phot_bp_mean_mag",
        "phot_rp_mean_mag",
        "j_mag",
        "h_mag",
        "k_mag",
        "w1_mag",
        "w2_mag",
        "w3_mag",
        "w4_mag",
    ],
    "cos": ["l"],
    "sin": ["b"],
}


class MetaVDataset(Dataset):
    def __init__(
        self,
        file,
        split="train",
        classes=None,
        min_samples=None,
        max_samples=None,
        random_seed=42,
        verbose=True,
    ):

        self.file = file
        self.split = split
        self.verbose = verbose
        self.classes = classes
        self.min_samples = min_samples
        self.max_samples = max_samples

        self.random_seed = random_seed
        np.random.seed(random_seed)

        self._load_and_transform_data()
        self._drop_duplicates()
        self._filter_classes()
        self._limit_samples()
        self._split()
        self._normalize()

        self.id2target = {
            i: x for i, x in enumerate(sorted(self.df["variable_type"].unique()))
        }
        self.target2id = {v: k for k, v in self.id2target.items()}
        self.num_classes = len(self.id2target)

    def _load_and_transform_data(self):
        self.df = pd.read_csv(self.file)
        self.df = self.df[ORIGINAL_METADATA_COLS + ["edr3_source_id", "variable_type"]]
        self._drop_nan()

        # now transform the data
        for transformation_type, value in METADATA_FUNC.items():
            if transformation_type == "abs":
                # make a column with the parallax, setting the default value to 1 if the original
                # value is less than 0
                self.df["parallax_tmp"] = self.df["parallax"].copy()
                self.df.loc[self.df["parallax"] < 0, "parallax_tmp"] = 1
                for col in value:
                    # use the parallax (in mas)to determine the absolute magnitude
                    self.df[col + "_abs"] = (
                        self.df[col] - 10 + 5 * np.log10(self.df["parallax_tmp"])
                    )
                # drop the temporary parallax column
                self.df.drop(columns=["parallax_tmp"], inplace=True)
            elif transformation_type == "cos":
                for col in value:
                    self.df[col + "_t"] = np.cos(np.radians(self.df[col]))
            elif transformation_type == "sin":
                for col in value:
                    self.df[col + "_t"] = np.sin(np.radians(self.df[col]))

        # now drop the columns in DROP_METADATA_COLS from df
        self.df.drop(columns=DROP_METADATA_COLS, inplace=True)

        # get the current column names
        self.metadata_cols = self.df.columns.tolist()
        # remove the edr3_source_id and variable_type from the metadata_cols
        self.metadata_cols.remove("edr3_source_id")
        self.metadata_cols.remove("variable_type")

    def _drop_nan(self):
        if self.verbose:
            print("Dropping nan values...", end=" ")

        self.df.dropna(axis=0, how="any", inplace=True)

        if self.verbose:
            print(f"Done. Left with {len(self.df)} rows.")

    def _drop_duplicates(self):
        if self.verbose:
            print("Dropping duplicated values...", end=" ")

        self.df.drop_duplicates(subset=["edr3_source_id"], keep="last", inplace=True)

        if self.verbose:
            print(f"Done. Left with {len(self.df)} rows.")

    def _filter_classes(self):
        if self.classes:
            if self.verbose:
                print(f"Leaving only classes: {self.classes}... ", end="")

            self.df = self.df[self.df["variable_type"].isin(self.classes)]

            if self.verbose:
                print(f"{len(self.df)} objects left.")

    def _limit_samples(self):
        if self.max_samples or self.min_samples:
            if self.verbose:
                print(
                    f"Removing objects that have more than {self.max_samples} or less than {self.min_samples} "
                    f"samples... ",
                    end="",
                )

            value_counts = self.df["variable_type"].value_counts()

            if self.min_samples:
                classes_to_remove = value_counts[value_counts < self.min_samples].index
                self.df = self.df[~self.df["variable_type"].isin(classes_to_remove)]

            if self.max_samples:
                classes_to_limit = value_counts[value_counts > self.max_samples].index
                for class_type in classes_to_limit:
                    class_indices = self.df[
                        self.df["variable_type"] == class_type
                    ].index
                    indices_to_keep = np.random.choice(
                        class_indices, size=self.max_samples, replace=False
                    )
                    self.df = self.df.drop(
                        index=set(class_indices) - set(indices_to_keep)
                    )

            if self.verbose:
                print(f"{len(self.df)} objects left.")

    def _split(self):
        unique_ids = self.df["edr3_source_id"].unique()
        train_ids, temp_ids = train_test_split(
            unique_ids, test_size=0.2, random_state=self.random_seed
        )
        val_ids, test_ids = train_test_split(
            temp_ids, test_size=0.5, random_state=self.random_seed
        )

        if self.split == "train":
            self.df = self.df[self.df["edr3_source_id"].isin(train_ids)]
        elif self.split == "val":
            self.df = self.df[self.df["edr3_source_id"].isin(val_ids)]
        elif self.split == "test":
            self.df = self.df[self.df["edr3_source_id"].isin(test_ids)]
        else:
            print("Split is not train, val, or test. Keeping the whole dataset")

        if self.verbose:
            print(f"{self.split} split is selected: {len(self.df)} objects left.")

    def _normalize(self):
        if self.split == "train":
            self.scaler = StandardScaler()
            self.scaler.fit(self.df[self.metadata_cols])
            joblib.dump(self.scaler, "scaler.pkl")
        else:
            self.scaler = joblib.load("scaler.pkl")

        self.df[self.metadata_cols] = self.scaler.transform(self.df[self.metadata_cols])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        el = self.df.iloc[idx]
        X = el[self.metadata_cols].values.astype(np.float32)
        y = self.target2id[el["variable_type"]]

        return X, y


class VPSMDatasetV2Meta(Dataset):
    def __init__(
        self,
        split="train",
        data_root="/home/mariia/AstroML/data/asassn/",
        file="preprocessed_data/full/spectra_and_v",
        min_samples=None,
        max_samples=None,
        classes=None,
        random_seed=42,
    ):

        self.data_root = data_root
        self.df = pd.read_csv(os.path.join(data_root, f"{file}_{split}_norm.csv"))
        self.metadata_cols = METADATA_COLS

        self.min_samples = min_samples
        self.max_samples = max_samples
        self.classes = classes

        self.random_seed = random_seed
        np.random.seed(random_seed)

        self._filter_classes()
        self._limit_samples()

        self.id2target = {
            i: x for i, x in enumerate(sorted(self.df["target"].unique()))
        }
        self.target2id = {v: k for k, v in self.id2target.items()}
        self.num_classes = len(self.id2target)

    def _filter_classes(self):
        if self.classes:
            self.df = self.df[self.df["target"].isin(self.classes)]

    def _limit_samples(self):
        if self.min_samples:
            value_counts = self.df["target"].value_counts()
            classes_to_remove = value_counts[value_counts < self.min_samples].index
            self.df = self.df[~self.df["target"].isin(classes_to_remove)]

        if self.max_samples:
            value_counts = self.df["target"].value_counts()
            classes_to_limit = value_counts[value_counts > self.max_samples].index

            for class_type in classes_to_limit:
                class_indices = self.df[self.df["target"] == class_type].index
                indices_to_keep = np.random.choice(
                    class_indices, size=self.max_samples, replace=False
                )
                self.df = self.df.drop(index=set(class_indices) - set(indices_to_keep))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        el = self.df.iloc[idx]
        label = self.target2id[el["target"]]
        metadata = el[self.metadata_cols].values.astype(np.float32)

        return metadata, label

from torch.utils.data import Dataset
from core.multimodal.dataset2 import VPSMDatasetV2


class VPSMDatasetV2Spectra(Dataset):
    def __init__(self, split='train', data_root='/home/mariia/AstroML/data/asassn/',
                 file='preprocessed_data/full/spectra_and_v', v_zip='asassnvarlc_vband_complete.zip',
                 v_prefix='vardb_files', min_samples=None, max_samples=None, classes=None, seq_len=200, phased=False,
                 clip=False, aux=False, random_seed=42):

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
        return spectra, label

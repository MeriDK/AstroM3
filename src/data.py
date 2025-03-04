from datasets import load_dataset
import torch
import torch.nn.functional as F
import random


def process_photometry(examples, seq_len=200, how='center'):
    """
    Processes photometry data by trimming or padding sequences to a fixed length.
    """
    photometry_all = []
    mask_all = []

    for photometry in examples['photometry']:
        if type(photometry) == list:
            photometry = torch.tensor(photometry, dtype=torch.float32)
        mask = torch.ones(seq_len, dtype=torch.float32)

        if photometry.shape[0] > seq_len:
            # Determine trimming start index
            if how == 'center':
                start_idx = (photometry.shape[0] - seq_len) // 2
            else:  # 'random'
                start_idx = random.randint(0, photometry.shape[0] - seq_len)

            # Trim sequence
            photometry = photometry[start_idx: start_idx + seq_len]
        else:
            # Update mask for padded values
            mask[photometry.shape[0]:] = 0

            # Pad sequence with zeros
            padding_length = seq_len - photometry.shape[0]
            photometry = F.pad(photometry, (0, 0, 0, padding_length), value=0)

        photometry_all.append(photometry)
        mask_all.append(mask)

    # Update example dictionary
    examples['photometry'] = photometry_all
    examples['photometry_mask'] = mask_all

    return examples


def process_train(examples, seq_len=200, how='random'):
    """
    Processes training data by trimming or padding sequences to a fixed length.
    Converts all inputs to torch tensors.
    """
    examples = process_photometry(examples, seq_len=seq_len, how=how)
    examples['spectra'] = [torch.tensor(el) for el in examples['spectra']]
    examples['metadata'] = [torch.tensor(el) for el in examples['metadata']]
    examples['label'] = [torch.tensor(el) for el in examples['label']]

    return examples


def get_datasets(dataset, name, seq_len):
    """
    Loads datasets from Hugging Face Hub, trims/pads photometry, converts input to torch tensors.
    """
    train = load_dataset(dataset, name=name, split='train')
    val = load_dataset(dataset, name=name, split='validation')
    test = load_dataset(dataset, name=name, split='test')

    val = val.with_format('torch')
    test = test.with_format('torch')

    val = val.map(process_photometry, batched=True, fn_kwargs={'seq_len': seq_len, 'how': 'center'})
    test = test.map(process_photometry, batched=True, fn_kwargs={'seq_len': seq_len, 'how': 'center'})

    train.set_transform(lambda examples: process_train(examples, seq_len=seq_len, how='random'))

    return train, val, test

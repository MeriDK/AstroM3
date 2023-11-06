import joblib
import numpy as np
import argparse
import os
import json


# TODO check why we normalize only 1 dim
def normalize(data, use_error=False):
    """
    Standardizes the input data and optionally adds an error dimension.

    Args:
    - data (np.array): The input data to be normalized.
    - use_error (bool): Whether to include an error dimension.

    Returns:
    - tuple: normalized_data, means, scales
    """

    # Extract shape details
    num_samples, time_steps, _ = data.shape

    # Calculate output dimensions
    out_dim = 3 if use_error else 2

    # Initialize normalized data array
    standardized_data = np.zeros((num_samples, time_steps, out_dim))
    standardized_data[:, :, 0] = data[:, :, 0]
    standardized_data[:, :, 1:out_dim] = data[:, :, 1:out_dim]

    # Calculate means and scales (standard deviation) for normalization
    means = np.nanmean(data[:, :, 1], axis=1, keepdims=True)
    scales = np.nanstd(data[:, :, 1], axis=1, keepdims=True)

    # Normalize the data
    standardized_data[:, :, 1] = (data[:, :, 1] - means) / scales

    return standardized_data, means, scales


def train_test_split(y, train_size=0.33):
    """
    Splits the indexes into train and test sets, maintaining the class distribution.

    Args:
    - y (array-like): Labels.
    - train_size (float): Proportion of the dataset to include in the train split.
    - random_state (int): Seed used by the random number generator; -1 for no seeding.

    Returns:
    - tuple: train_idxs, test_idxs
    """

    # Extract unique labels
    unique_labels = np.unique(y)

    # Create a list to hold indexes for each label
    label_based_indexes = [np.where(y == label)[0] for label in unique_labels]

    # Shuffle indices for each label
    for idx_list in label_based_indexes:
        np.random.shuffle(idx_list)

    # Split indexes based on the specified train_size
    train_idxs = []
    test_idxs = []
    for idx_list in label_based_indexes:
        split_point = max(int(train_size * len(idx_list) + 0.5), 1)
        train_idxs.extend(idx_list[:split_point])
        test_idxs.extend(idx_list[split_point:])

    return train_idxs, test_idxs


def filter_data_by_errors(light_curve):
    """Filter light curve data based on error values."""
    valid_data = (light_curve.errors > 0) & (light_curve.errors < 99)
    light_curve.times = light_curve.times[valid_data]
    light_curve.measurements = light_curve.measurements[valid_data]
    light_curve.errors = light_curve.errors[valid_data]


def filter_data_by_max_sample(data, args):
    """Filters data by labels and max sample."""
    labels = [lc.label for lc in data]
    unique_label, count = np.unique(labels, return_counts=True)
    use_label = unique_label[count >= args.min_sample]

    filtered_data = []
    for cls in use_label:
        class_data = [lc for lc in data if lc.label == cls]
        filtered_data.extend(class_data[:min(len(class_data), args.max_sample)])

    return filtered_data


def fix_macho_labels(light_curve):
    """Fix macho labels in light curve data."""
    if 'LPV' in light_curve.label:
        light_curve.label = "LPV"


def sanitize_data(data, args):
    # Filter dataset by error range
    for lc in data:
        filter_data_by_errors(lc)

    # Fix labels if this is macho dataset
    if 'macho' in args.input:
        for lc in data:
            fix_macho_labels(lc)

    # Adjust max sample if this is asassn dataset
    if 'asassn' in args.input:
        args.max_sample = 20000

    # Filter data
    data = filter_data_by_max_sample(data, args)

    # Convert labels to numerical form
    all_labels_string = [lc.label for lc in data]
    unique_label, count = np.unique(all_labels_string, return_counts=True)
    label_to_num = dict(zip(unique_label, range(len(unique_label))))
    all_labels = np.array([label_to_num[lc.label] for lc in data])

    # Determine number of inputs based on use_error flag
    n_inputs = 3 if args.use_error else 2

    return data, all_labels, len(unique_label), n_inputs, label_to_num


def process_data(split, args, n_inputs, label_to_num, scales_all=None):
    """Processes and normalizes light curve data."""
    x_list = [np.c_[chunk.times, chunk.measurements, chunk.errors] for chunk in split]
    periods = np.array([lc.p for lc in split])
    label = np.array([label_to_num[chunk.label] for chunk in split])

    x, means, scales = normalize(np.array(x_list), use_error=args.use_error)
    print('Shape of the dataset array:', x.shape)

    if scales_all is not None:
        mean_x = scales_all[0][:-1]
        std_x = scales_all[1][:-1]
    else:
        mean_x = x.reshape(-1, n_inputs).mean(axis=0)
        std_x = x.reshape(-1, n_inputs).std(axis=0)

    x -= mean_x
    x /= std_x

    x = np.swapaxes(x, 2, 1)
    aux = np.c_[means, scales, np.log10(periods)]

    if args.use_meta and split[0].metadata is not None:
        metadata = np.array([lc.metadata for lc in split])  # Metadata must have same dimension!
        aux = np.c_[aux, metadata]                          # Concatenate metadata
        print('Metadata will be used as auxiliary inputs.')

    if scales_all is not None:
        aux_mean = scales_all[2]
        aux_std = scales_all[3]
    else:
        aux_mean = aux.mean(axis=0)
        aux_std = aux.std(axis=0)

    aux -= aux_mean
    aux /= aux_std

    if scales_all is None:
        scales_all = np.array([np.append(mean_x, 0), np.append(std_x, 0), aux_mean, aux_std])

    return x, label, aux, scales_all


def get_data_args(notebook=False):

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--L', type=int, default=200,
                        help='training sequence length')
    parser.add_argument('--dir', type=str, default='./data',
                        help='dataset directory (default: ./data/)')
    parser.add_argument('--input', type=str, default='macho_raw.pkl',
                        help='dataset filename. file is expected in ./data/')
    parser.add_argument('--output', type=str, default='macho',
                        help='output dataset directory. dir is expected in ./data/')
    parser.add_argument('--frac-train', type=float, default=0.8,
                        help='training sequence length')
    parser.add_argument('--frac-valid', type=float, default=0.25,
                        help='training sequence length')
    parser.add_argument('--use-error', action='store_true', default=False,
                        help='use error as additional dimension')
    parser.add_argument('--use-meta', action='store_true', default=False,
                        help='use meta as auxiliary network input')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--min_sample', type=int, default=50,
                        help='minimum number of pre-segmented light curve per class')
    parser.add_argument('--max_sample', type=int, default=100000,
                        help='maximum number of pre-segmented light curve per class during testing')

    if notebook:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()

    return args


def main():
    args = get_data_args()
    np.random.seed(args.seed)

    data = joblib.load(f'{args.dir}/{args.input}')
    data, all_labels, n_classes, n_inputs, label_to_num = sanitize_data(data, args)

    unique_label, count = np.unique([lc.label for lc in data], return_counts=True)
    print('------------before segmenting into L={}------------'.format(args.L))
    print(unique_label)
    print(count)

    train_idxs, test_idxs = train_test_split(all_labels, train_size=args.frac_train)

    # TODO check what split(args.L, args.L) does
    # TODO check if data[i].label can be None
    train_split = [chunk for i in train_idxs for chunk in data[i].split(args.L, args.L) if data[i].label is not None]
    test_split = [chunk for i in test_idxs for chunk in data[i].split(args.L, args.L) if data[i].label is not None]

    # TODO check what period_fold does and if it can be moved to preprocessing functions
    for lc in train_split:
        lc.period_fold()
    for lc in test_split:
        lc.period_fold()

    unique_label, count = np.unique([lc.label for lc in train_split], return_counts=True)
    print('------------after segmenting into L={}------------'.format(args.L))
    print(unique_label)
    print(count)

    x_train, label_train, aux_train, scales_all = process_data(train_split, args, n_inputs, label_to_num)
    x_test, label_test, aux_test, _ = process_data(test_split, args, n_inputs, label_to_num, scales_all)
    train_idx, val_idx = train_test_split(label_train, 1 - args.frac_valid)

    os.makedirs(f'data/{args.output}', exist_ok=True)
    joblib.dump((x_train[train_idx], aux_train[train_idx], label_train[train_idx]),
                f'data/{args.output}/train.pkl')
    joblib.dump((x_train[val_idx], aux_train[val_idx], label_train[val_idx]),
                f'data/{args.output}/val.pkl')
    joblib.dump((x_test, aux_test, label_test), f'data/{args.output}/test.pkl')
    np.save(f'data/{args.output}/scales.npy', scales_all)

    with open(f'data/{args.output}/info.json', 'w') as f:
        f.write(json.dumps({
            'n_classes': n_classes,
            'n_inputs': n_inputs
        }))


if __name__ == '__main__':
    main()

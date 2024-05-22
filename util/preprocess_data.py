import os
import json
import argparse

import joblib
import numpy as np
from scipy import stats

from cesium.features.lomb_scargle import lomb_scargle_model, get_lomb_signif


def clip_outliers(t, m, merr, max_sigma=5, max_iter=5,
                  measurements_in_flux_units=False,
                  sys_err=0.05, nharm=10, verbose=False, fixed_P=None,
                  max_frac_del=0.15, initial_clip=[100,10], max_err=0.5, clean_only=False):

    """
    Iteratively clips outliers from a light curve using a Lomb-Scargle
    periodogram model. Returns the clipped light curve and the model and mean-mag.

    Args:
    - t (np.array): The input time array.
    - m (np.array): The input magnitude array.
    - merr (np.array): The input magnitude error array.
    - max_sigma (float): The maximum sigma deviation to clip.
    - max_iter (int): The maximum number of iterations.
    - measurements_in_flux_units (bool): Whether the input measurements are in flux units.
    - sys_err (float): The systematic error to use for the model.
    - nharm (int): The number of harmonics to use for the model.
    - verbose (bool): Whether to print out progress.
    - fixed_P (float): A fixed period to use for the model.
    - max_frac_del (float): The maximum fraction of data to clip.
    - initial_clip (list): The initial clipping to use (in units of sigma, above and below
                the median). Set this to None to skip initial clipping.
    - max_err (float): The maximum mag error to allow. default: 0.5
    - clean_only (bool): If True, only clean the data and do not fit a model.

    usage:
     t, y, dy = d.times, d.measurements, d.errors
     t, y, yerr, rez, P, mean_flux_mag, mean_flux_mag_err, mag0 = clip_outliers(t, y, dy, max_sigma=4, max_iter=5)
     sign = get_lomb_signif(rez)

    """

    t = t.copy()
    try:
        mag0 = np.average(m, weights=1/merr)
    except:
        print(f"{m} {type(m)} {merr} {type(merr)}")
        raise
    if not measurements_in_flux_units:
        f = 10**(-0.4*(m - mag0))
        ferr = 0.4*np.log(10)*f*merr
        m = m.copy()
        merr = merr.copy()
    else:
        f = m.copy()
        ferr = merr.copy()

    # ensure merr is >= 0 and not nan
    try:
        goods = np.squeeze(np.argwhere(~np.isnan(merr) & (merr >= 0)))
    except:
        print(f"{m} {type(m)} {merr} {type(merr)}")
        raise
    t = t[goods]
    f = f[goods]
    ferr = ferr[goods]
    m = m[goods]
    merr = merr[goods]

    initial_size = len(t)
    max_removed = 0
    if initial_clip is not None:
        med = np.nanmedian(f)
        mad = stats.median_abs_deviation(f, nan_policy="omit")
        if verbose:
            print(f"initial median = {med:0.3f} mad = {mad:0.3f}")
        bads = np.argwhere((f >= initial_clip[1]*mad + med) | 
                (merr > max_err) | 
                (f < med - initial_clip[0]*mad))[:int(max_frac_del * len(f))]
        max_removed += len(bads)
        goods = np.delete(np.arange(len(f)), bads)
        t = t[goods[:]]
        f = f[goods[:]]
        ferr = ferr[goods[:]]
        m = m[goods[:]]
        merr = merr[goods[:]]
        if verbose:
            print(f"max_removed (initial cut): {max_removed}")
    
    if not clean_only:
        rez = lomb_scargle_model(t, f, ferr, sys_err=sys_err, nharm=nharm,
                                nfreq=2, tone_control=5.0,default_order=1,
                                freq_grid=None, normalize=False)

        if not fixed_P:
            P = 1 / rez["freq_fits"][0]["freq"]
        else:
            P = fixed_P

        freqs_global = rez["freq_fits"][0]["freqs_vector"]
        psd_global = rez["freq_fits"][0]["psd_vector"]

        if verbose:
            print(f"iter: 0 ... P: {P} n: {len(t)}")
    else:
        rez = {}
        P, mean_flux_mag, mean_flux_mag_err, mag0 = None, None, None, None
        freqs_global = []
        psd_global = []
        max_iter = 0

    iter = 0
    while iter < max_iter:
        # run L-S with 1 freq
        df = 1/5000.00
        f0 = max(1/P - 25*df, df)  # periodogram starting (low) frequency
        fe = 1/P + 25*df  # periodogram ending (high) frequency
        numf = int((fe-f0)/df) + 1
        freq_grid_param = {"f0": f0, "df": df, "fmax": fe, "numf": numf}

        rez = lomb_scargle_model(t, f, ferr, sys_err=sys_err, nharm=nharm,
                                 nfreq=1, tone_control=10.0, default_order=1,
                                 freq_grid=freq_grid_param, normalize=False)

        if not fixed_P:
            P = 1 / rez["freq_fits"][0]["freq"]
        else:
            P = fixed_P

        if verbose:
            print(f"iter: {iter+1} ... P: {P} n: {len(t)}")
        resid = f - rez["freq_fits"][0]["model"]
        resid_err = np.sqrt(ferr**2 + rez["freq_fits"][0]["model_error"]**2)

        # deviation from the model in sigma
        scaled_resid = np.abs(resid)/resid_err
        bads = np.argwhere(scaled_resid >= max_sigma)[:(int(max_frac_del * initial_size) - 1)]
        goods = np.delete(np.arange(len(scaled_resid)), bads)
        if (len(goods) == len(scaled_resid)) or (max_removed > int(max_frac_del * initial_size)):
            # no more outliers to clip
            break

        t = t[goods[:]]
        f = f[goods[:]]
        ferr = ferr[goods[:]]
        m = m[goods[:]]
        merr = merr[goods[:]]
        max_removed += len(bads)
        if verbose:
            print(f"max_removed: {max_removed}")

        # run one last time to get the model
        rez = lomb_scargle_model(t, f, ferr, sys_err=sys_err, nharm=nharm,
                                 nfreq=1, tone_control=10.0, default_order=1,
                                 freq_grid=freq_grid_param, normalize=False)

        if not fixed_P:
            P = 1 / rez["freq_fits"][0]["freq"]
        else:
            P = fixed_P
        iter += 1

    if not measurements_in_flux_units:
        y = m
        yerr = merr
        if not clean_only:
            mean_flux_mag = mag0 - 2.5*np.log10(
                                rez["freq_fits"][0]["trend_coef"][0])
            mean_flux_mag_err = 2.5/np.log(10) * \
                                rez["freq_fits"][0]["trend_coef_error"][0]
            rez["freq_fits"][0]["model"] = mag0 - 2.5*np.log10(
                                rez["freq_fits"][0]["model"])
    else:
        y = f
        yerr = ferr
        if not clean_only:
            mean_flux_mag = rez["freq_fits"][0]["trend_coef"][0]
            mean_flux_mag_err = rez["freq_fits"][0]["trend_coef_error"][0]

    rez["freqs_global"] = freqs_global
    rez["psd_global"] = psd_global
    
    return t, y, yerr, rez, P, mean_flux_mag, mean_flux_mag_err, mag0


def normalize(data, use_error=False, measurements_in_flux_units=False):
    """
    Standardizes the input data and optionally adds an error dimension.

    data is a 3D array of shape (num_samples, time_steps, [time, mag, mag_err])
       so data[:, :, 0] is the time axis, data[:, :, 1] is the flux axis, and
         data[:, :, 2] is the error axis.

    Args:
    - data (np.array): The input data to be normalized.
    - use_error (bool): Whether to include an error dimension.
    - measurements_in_flux_units (bool): Whether the input measurements are in flux units.
       If not (ie. they are in mags), then convert to fluxes.

    Returns:
    - tuple: normalized_data, means, scales
    """

    # Extract shape details
    num_samples, time_steps, _ = data.shape

    # Calculate output dimensions
    out_dim = 3 if use_error else 2

    # Initialize normalized data array
    standardized_data = np.zeros((num_samples, time_steps, out_dim))

    #  time axis
    standardized_data[:, :, 0] = data[:, :, 0]

    # convert to fluxes if necessary
    if not measurements_in_flux_units:
        if not use_error:
            weights = np.ones_like(data[:, :, 1])
        else:
            weights = 1.0 / data[:, :, 2]

        mag0 = np.average(data[:, :, 1], weights=weights, keepdims=True)
        f = 10**(-0.4*(data[:, :, 1] - mag0))
        if use_error:
            ferr = 0.4*np.log(10)*f*data[:, :, 2]
    else:
        f = data[:, :, 1]
        if use_error:
            ferr = data[:, :, 2]

    # flux axis
    standardized_data[:, :, 1] = f

    # error axis
    if use_error:
        standardized_data[:, :, 2] = ferr

    # Calculate median and scales (mad) for normalization
    med = np.nanmedian(standardized_data[:, :, 1], axis=1, keepdims=True)
    mad = np.expand_dims(stats.median_abs_deviation(standardized_data[:, :, 1], 
                            axis=1, nan_policy="omit"), axis=1)

    # Normalize the data
    standardized_data[:, :, 1] = (standardized_data[:, :, 1] - med) / mad
    if use_error:
        standardized_data[:, :, 2] = standardized_data[:, :, 2] / mad

    return standardized_data, med, mad


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
    valid_data = (light_curve.errors > 0) & (light_curve.errors < 99) & (light_curve.errors != np.nan)
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

        if args.clip:
            # Clip outliers whilst finding better periods
            t, y, dy = lc.times, lc.measurements, lc.errors
            t, y, yerr, rez, P, _, _, mag0 = clip_outliers(t, y, dy, max_sigma=4, max_iter=5)
            print(f"{lc.name} {lc.p} {P}")
            # make sure the new period is not too different from sidereal day
            if np.abs((P - 0.997)/0.997) >= 0.005:
                # make sure is not a multiple of the orginal period
                if np.abs((0.5*P - lc.p)/lc.p) >= 0.005:
                    lc.p=P
            lc.p_signif=get_lomb_signif(rez)

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
    """Processes and normalizes light curve data.
    
    scales_all is a tuple of (mean_x, std_x, aux_mean, aux_std) if it is not None.
    n_inputs is 3 if use_error is True, otherwise 2.
    args is the argument parser.
    label_to_num is a dictionary of label to number.
    """

    x_list = [np.c_[chunk.times, chunk.measurements, chunk.errors] for chunk in split]
    
    periods = np.array([lc.p for lc in split])
    label = np.array([label_to_num[chunk.label] for chunk in split])
    x, means, scales = normalize(np.array(x_list), use_error=args.use_error)
    print('Shape of the dataset array:', x.shape)

    # Normalize the entire dataset, but leave the time axis alone
    # save the mean and std for later use during testing
    if scales_all is not None:
        global_med = scales_all[0]
        global_mad = scales_all[1]
    else:
        global_med = np.nanmedian(x[:, :, 1])
        global_mad = np.nanmedian(stats.median_abs_deviation(x[:, :, 1], nan_policy="omit"))

    x[:,:,1] -= global_med
    x[:,:,1] /= global_mad

    if args.use_error:
        x[:,:,2] /= global_mad

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
        scales_all = [global_med, global_mad, aux_mean, aux_std]

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
    parser.add_argument('--clip', action='store_true', default=False,
                        help='sigma clip light curves based on folded period')
    parser.add_argument('--phase_norm', action='store_true', default=False,
                        help='normalize phase in the folded light curves so they all go from [0,1]')
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

    for lc in train_split:
        lc.period_fold(normalize=args.phase_norm)
    for lc in test_split:
        lc.period_fold(normalize=args.phase_norm)

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

    # let's save the scales as a pickle file instead of numpy array
    joblib.dump((scales_all), f'data/{args.output}/scales.pkl')

    with open(f'data/{args.output}/info.json', 'w') as f:
        f.write(json.dumps({
            'n_classes': n_classes,
            'n_inputs': n_inputs
        }))


if __name__ == '__main__':
    main()

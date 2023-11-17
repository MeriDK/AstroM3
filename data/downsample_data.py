import joblib
import numpy as np
import shutil


def downsample(data_root, new_data_root, mode):
    print(data_root + f'{mode}.pkl')
    data = joblib.load(data_root + f'{mode}.pkl')
    values, aux, labels = data

    unique, counts = np.unique(labels, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    print('Class distribution before:', class_distribution)

    min_samples = min(class_distribution.values())
    print('Min samples:', min_samples)

    downsampled_values = []
    downsampled_aux = []
    downsampled_labels = []

    for class_label in unique:
        # Find the indices of all samples belonging to the current class
        indices = np.where(labels == class_label)[0]

        # Randomly select 'min_samples' indices from these
        np.random.shuffle(indices)
        selected_indices = indices[:min_samples]

        # Append the selected samples to the downsampled data lists
        downsampled_values.extend(values[selected_indices])
        downsampled_aux.extend(aux[selected_indices])
        downsampled_labels.extend(labels[selected_indices])

    # Convert lists back to numpy arrays
    downsampled_values = np.array(downsampled_values)
    downsampled_aux = np.array(downsampled_aux)
    downsampled_labels = np.array(downsampled_labels)

    print('New shapes:', downsampled_values.shape, downsampled_aux.shape, downsampled_labels.shape)

    joblib.dump([downsampled_values, downsampled_aux, downsampled_labels], new_data_root + f'{mode}.pkl')


def main():
    np.random.seed(42)
    data_root = '/home/mrizhko/AML/AstroML/data/macho/'
    new_data_root = '/home/mrizhko/AML/AstroML/data/macho-balanced/'

    downsample(data_root, new_data_root, mode='train')
    downsample(data_root, new_data_root, mode='val')
    downsample(data_root, new_data_root, mode='test')

    shutil.copy(data_root + 'info.json', new_data_root + 'info.json')
    shutil.copy(data_root + 'scales.pkl', new_data_root + 'scales.pkl')


if __name__ == '__main__':
    main()

# AstroML

## Setup

Before running the scripts, you need to install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Downloading the Data

The folder _data_ contains scripts for downloading and processing the MACHO dataset as used in the Periodic Network 
project. The original codebase can be found at [kmzzhang/periodicnetwork](https://github.com/kmzzhang/periodicnetwork), 
which should be referenced for detailed data preprocessing steps.

To obtain the dataset, execute the following commands in your terminal:

1. Go to the _data_ folder:
   ```sh
   cd data
   
2. Download the MACHO data by executing the shell script:
   ```sh
   sh download_macho.sh

3. Run pre_processing file on the downloaded file:
   ```sh
   python preprocess_data.py

## Training Time Series Transformer Model

All the code is located in _ts-hf-periodic-refactor.ipynb_. It consists of four steps:

1. Creating custom dataset class MachoDataset, which uses the data that was just downloaded and pre-processed in the 
previous steps. You should change _data_root_ path to the location where your data is located.

2. Setting up code for pre-training time series transformer model on the prediction task. For more details about the
model refer to hugging face [docs](https://huggingface.co/docs/transformers/v4.34.1/en/model_doc/time_series_transformer).

3. Evaluation of the pre-trained model on the prediction task. It produces 2 metrics MASE and sMAPE that should be used
for comparison of the pre-trained models.

4. The next step defines Classification Model which is the pre-trained model from the previous steps plus a
classification layer.

5. Confusion matrix for the trained Classification Model.

## To-Do List

Step 1. Right Now (the goal is to achieve at least 90% accuracy):
- [ ] Regenerate data with errors
- [ ] Train the model with different prediction window lengths
- [ ] Fine-tune hyperparameters for pre-training model
- [ ] Fine-tune hyperparameters for the classification model
- [ ] Add additional inputs into the model: aux, errors, aux + errors

Step 2. After that (the goal is to set up training for variable length objects with at least the same 90% on the old 
data and decent results on the new data):
- [ ] Create new data of different lengths and corresponding masks
- [ ] Check the current model's performance on the new data of different lengths
- [ ] Train the model with the new data of different lengths
- [ ] Check the average performance on all data
- [ ] Check the new models' performance on the new data of different lengths

Step 3. After that (at the same time?) (the goal is to set up training for the new modality (spectra) and show better
general performance with all 3 modalities combined than on each separately):
- [ ] Generate the data only with all 3 modalities
- [ ] Set up training for Multimodal proposal sample data
- [ ] Get baseline results with some RNN model (LSTM?)
- [ ] Get baseline results on aux data
- [ ] Get baseline results on spectra
- [ ] Get the results for Time Series Transformer model with: flux, flux + aux
- [ ] Set up training for the multi-modal data flux + aux + spectra

Step 4. Missing modalities (the goal is to set up training for the missing modalities, it's supposed to show the same
results for the data with all modalities and decent results for objects with missing modalities):
- [ ] Add objects with missing modalities creating a new dataset
- [ ] Set up training for the multi-modal data flux + aux + spectra with missing modalities
- [ ] Check the performance on the data with all modalities
- [ ] Check the performance on the data with missing modalities

Step 5. Transfer everything to our real world data.

Optionally:
- [ ] Add scripts for loading ASAS-SN and OGLE-III
- [ ] Repeat the experiments from Step 1 and Step 2 with ASAS-SN and OGLE-III

Other tasks:
- [ ] Move everyting from jupyter notebooks to python files
- [ ] Add weights \& biases
- [ ] Add learning rate scheduler



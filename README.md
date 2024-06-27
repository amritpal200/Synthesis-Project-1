# Synthesis-Project-1

## Overview
The aim of this project is to develop an Artificial Intelligence model capable of detecting anomalies in Apache web server logs.

## Table of Contents
1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Usage](#usage)

## Installation
### Clone the repository
```bash
git clone https://github.com/amritpal200/Synthesis-Project-1.git
cd Synthesis-Project-1
```

### Install the dependencies:
```
pip install -r requirements.txt
```
If you have GPU:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Otherwise for CPU:
```
pip install torch torchvision torchaudio
```

### Donwload the log files
Once you have the repository on your local machine, place your `.log` files in the data folder of this project, in order for the models to process them.

## Project Structure
This project is divided into three folders, each containing different kinds of files.

- `data` folder is used to save the files of data and preprocessed data. It contains a small dataset `sitges_access.csv` for demo purposes.
- `models` folder is used to save the models developed to solve the challenge, apart from other models used to create embeddings on specific features of the data.
- `src` folder is used to save the code files used to preprocess the data, develop, and train the model. The files resulting from the execution of these scripts are saved in the prior folders.

## Usage
To reproduce the results obtained for each model, you will need to execute the following files, all of them from the src folder.

First of all, we need to convert the log files to .csv format to work with them. For this, we need to execute the file `log_to_csv.ipynb`. The resulting .csv files will be saved in the data folder.

### Autoencoders
To preprocess the data for the autoencoders, we need to execute two different files.

The first one is `data_cleaning_autoencoder.ipynb`. This file is used to clean the data. We will execute this file once, and then, we will execute only the last cell again, but change the *file* variable definition from 'file = f"../data/{files[0]}.csv"' to 'file = f"../data/{files[1]}.csv"'. The result of these executions will be two different .csv files called `sitges_access_clean_whole_set_but_last.csv` and `sitges_access_clean_last.csv`. Both files will be saved in the data folder.

The second one is `data_preprocessing_autoencoder.ipynb`. This file is used to preprocess the data once it is cleaned. We will execute this file twice, the first time as it is, and the second time changing the *file* variable definition from 'file = f"../data/{files[0]}.csv"' to 'file = f"../data/{files[1]}.csv"'. The result of these executions will be two different .csv files called `sitges_access_prepared_whole_set_but_last.csv` and `sitges_access_prepared_last.csv`. Both files will be saved in the data folder.

Once we have the data preprocessed, the models can be trained.

#### Normal Autoencoder
To train the normal autoencoder model, we will need to execute the file called `Normal_Autoencoder.ipynb`. The result of the execution will be a file called `normalAutoencoder.pt` that will be saved in the models folder. This resulting file is the model ready to be used.

#### LSTM Autoencoder
To train the LSTM autoencoder model, we will need to execute the file called `LSTM_Autoencoder.ipynb`. The result of the execution will be a file called `LSTMAutoencoder.pt` that will be saved in the models folder. This resulting file is the model ready to be used.

To compare the results of these two models, we will execute the file called `results_autoencoders.ipynb`. The results of this file will be a printed graph for each of the models representing the score of each log given its error value on the datasetset used for results. Additionally, at the end of this file will be printed a *Dataframe* object for each of the models showing the anomalies based on a editable threshold.

### Deeplog
Deeplog requires a complex preprocessing, which is implemented in `data_cleaning.ipynb`. Executing this file will preprocess the `sitges_access.csv` dataset and create a new `sitges_access_clean.csv`.  
> **Note**: in order to come up with such preprocessing, a thorough analysis was made and can be seen in files `data_analysis_X.ipynb`.

An implementation of Deeplog parameter value model as explained in the paper can be found in `deeplog.py`, and a demonstration of how can be trained and used is in `deeplog.ipynb`.

### Combination of Autoencoder and Deeplog
An implementation of the combination of the normal Autoencoder and Deeplog can be found in `combo.ipynb`. The same clean dataset as in Deeplog is used, so a pretrained Deeplog model can be loaded. For consistency, the autoencoder is trained on the same dataset.  
We've trained such combination model and processed a set of logs. Example of URLs from logs that have been detected as highly potential anomalies can be found in the `data` folder.

### Transformer
To train the Transformer model, first obtain scores through Wazuk. For this, use `wazuk.py` located in the `src` folder. Once you have a `.csv` file with the scores, preprocess this file for training. To do this, pass the `.csv` file with Wazuk scores through `data_preprocessing_transformer.py`. This will generate three different `.csv` files: training, validation, and test files. With these files, you can proceed to training using the `transformer_model.pt` file.

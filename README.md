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

### Install the following dependencies:
- numpy
- torch
- pandas
- scikit-learn
- tqdm
- fasttext

### Donwload the log files
Once you have the repository on your local machine, download the log files available in the Campus Virtual and save them in the data folder of this project.

## Project Structure
This project is divided into three folders, each containing different kinds of files.

**data** folder is used to save the files of data and preprocessed data.

**models** folder is used to save the models developed to solve the challenge, apart from other models used to create embeddings on specific features of the data.

**src** folder is used to save the code files used to preprocess the data, develop, and train the model. The files resulting from the execution of these scripts are saved in the prior folders.

## Usage
To reproduce the results obtained for each model, you will need to execute the following files, all of them from the src folder.

First of all, we need to convert the log files to .csv format to work with them. For this, we need to execute the file `log_to_csv.py`. The resulting .csv files will be saved in the data folder.

### Autoencoders
To preprocess the data for the autoencoders, we need to execute two different files.

The first one is `data_cleaning_autoencoder.ipynb`. This file is used to clean the data. We will execute this file once, and then, we will execute only the last cell again, but change the *file* variable definition from 'file = f"../data/{files[0]}.csv"' to 'file = f"../data/{files[1]}.csv"'. The result of these executions will be two different .csv files called `sitges_access_clean_whole_set_but_last.csv` and `sitges_access_clean_last.csv`. Both files will be saved in the data folder.

The second one is `data_preprocessing_autoencoder.ipynb`. This file is used to preprocess the data once it is cleaned. We will execute this file twice, the first time as it is, and the second time changing the *file* variable definition from 'file = f"../data/{files[0]}.csv"' to 'file = f"../data/{files[1]}.csv"'. The result of these executions will be two different .csv files called `sitges_access_prepared_whole_set_but_last.csv` and `sitges_access_prepared_last.csv`. Both files will be saved in the data folder.

Once we have the data preprocessed, the models can be trained.

#### Normal Autoencoder
To train the normal autoencoder model, we will need to execute the file called `Normal_Autoencoder.ipynb`. The result of the execution will be a file called `normalAutoencoder.pt` that will be saved in the models folder. This resulting file is the model ready to be used.

#### LSTM Autoencoder
To train the LSTM autoencoder model, we will need to execute the file called `LSTM_Autoencoder.ipynb`. The result of the execution will be a file called `LSTMAutoencoder.pt` that will be saved in the models folder. This resulting file is the model ready to be used.


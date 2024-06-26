# Synthesis-Project-1

## Overview
The aim of this project is to develop an Artificial Intelligence model capable of detecting anomalies in Apache web server logs.

## Table of Contents
1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Usage](#usage)

## Installation
### Clone the repository
```
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
Once you have the repositorry on your local, download the log files avaliable in the Campus Virtual and safe them in the data folder of this project.

## Project Structure
This project is divided in three folders, each of them containing different kinds of files.

**data** folder is used to safe the files used for the data preprocessing, mainly compossed by .csv files.

**models** folder is used to safe the models developed to solve the challenge, apart form other models used for create embeddings on specific features of the data.

**src** folder is used to safe the code files, used to create to preprocess the data, develop and train the model. The files resulting from the execution of these files are saved on the proir folders.

## Usage
To reproduce the results obtained for each model, there would be need to ejecute the following files, all of them from the src folder.

First of all, we need to pass the log files to csv ones, to be able to work with them. For this we need to ejecute the file `nombre`. The resulting .cvs files will be saved on the data folder.

### Autoencoders
To preproces the data to be used for the autoencoders, we need to execute two different files. 

The first one is `data_cleaning_autoencoder.ipynb`. This file is the one used to clean the data and it will be needed to be executed once to create a combined csv of all the .csv logs but the last file and clean the resulting file, and then, there would be needed to execut the last cell, but changing the *file* variable definition from 'file = f"../data/{files[0]}.csv"' to 'file = f"../data/{files[1]}.csv"', which would clean the .csv used as test. The result of these executions will be two different .csv files called `sitges_access_clean_whole_set_but_last.csv` and `sitges_access_clean_last.csv`. Both files will be saved on the data folder.

The second one is `data_preprocessing_autoencoder.ipynb`. This file is used to preprocess the data once is cleaned. We will execute this file twice, the first one as it is and the second one changing the *file* variable definition from 'file = f"../data/{files[0]}.csv"' to 'file = f"../data/{files[1]}.csv"'. The result of these executions will be two different .csv files called `sitges_access_prepared_whole_set_but_last.csv` and `sitges_access_prepared_last.csv`. Both files will be saved on the data folder.

Once we have the data preprocessed, the models can be trained.

#### Normal Autoencoder
To train the normal autoencoder model, we will need to execute the file called `Normal_Autoencoder.ipynb`. The result of the execution will be a file called `normalAutoencoder.pt` that will be saved in the models folder.

#### LSTM Autoencoder
To train the LSTM autoencoder model, we will need to execute the file called `LSTM_Autoencoder.ipynb`. The result of the execution will be a file called `LSTMAutoencoder.pt` that will be saved in the models folder.


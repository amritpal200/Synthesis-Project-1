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

## Project Structure
This project is divided in three folders, each of them containing different kinds of files.

**data** folder is used to safe the files used for the data preprocessing, mainly compossed by .csv files.

**models** folder is used to safe the models developed to solve the challenge, apart form other models used for create embeddings on specific features of the data.

**src** folder is used to safe the code files, used to create to preprocess the data, develop and train the model. The files resulting from the execution of these files are saved on the proir folders.

## Usage
To reproduce the results obtained for each model, there would be need to ejecute the following files, all of them from the src folder.

First of all, we need to pass the log files to csv ones, to be able to work with them. For this we need to ejecute the file `nombre`. The resulting .cvs files will be saved on the data folder.

### Autoencoders
To preproces the data to be used for the autoencoders, we need to execute two different files. The first one is `data_cleaning_autoencoder.ipynb`


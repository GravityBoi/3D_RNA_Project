# Stanford RNA 3D Folding

This repository contains our work as part of the Stanford RNA 3D Folding Kaggle competition.

Authors: Max Neuwinger, Sarah Verreault, Andreas Hiropedi

## Folder structure

The repository contains the following folders:

- ```Analysis code```: this contains the notebooks we used to obtain insights about aspects of the data we were working with, as well as information about model confidence scores and model selection as a motivation for our meta learner
- ```DRfold2```: this contains all the notebooks we used to test the DRfold2 model (https://github.com/leeyang/DRfold2)
- ```Protenix```: this contains all the notebooks we used to test the Protenix model (https://github.com/bytedance/Protenix)
- ```RhoFold+```: this contains all the notebooks we used to test the RhoFold+ model (https://github.com/ml4bio/RhoFold)
- ```RibonanzaNet2```: this contains all the notebooks we used to test the RibonanzaNet2 model (https://www.kaggle.com/models/shujun717/ribonanzanet2)
- ```Ensembles```: this contains all our code used for our experiments with different model ensemble strategies
- ```Meta-learner```: this contains all our code used for our experiments with the meta learner
- ```Custom_datasets``: this contains our custom datasets that we created following the unforeseen permanent closing of submissions from the Kaggle competition (followin the May 29 deadline)
  

## Pre-requisites

1) To be able to copy this code locally and execute it, you will first need to ensure that you have Python (https://www.python.org ), git (https://github.com/git-guides/install-git) and Anaconda (https://www.anaconda.com/docs/getting-started/anaconda/install) already installed, as well as a code editor (Pycharm or Visual Studio Code (VScode) will do).

2) Once Python and git are set up, clone this repo using the following command:

```sh
git clone https://github.com/AndreasHiropedi/3D_RNA_Folding.git
```

3) After cloning the repository, change directory so that you are in the repository directory (```3D_RNA_Folding```).

4) Once there, you will need to install the following packages, as these are commonly used across all notebooks (note: these do not include any model specific dependencies, we list more specific instructions for how to run our experiment code for each model those under the corresponding model's section):

```sh
pip install pandas
pip install numpy
pip install matplotlib
pip install torch torchvision torchaudio
pip install pyyaml
pip install tqdm
pip install optuna
pip install scipy
```

5) Since all notebooks also use the USalign tool for scoring, you will additionally need to run the following command to install this locally (note: once this is installed locally, you will need to modify all the USalign directory paths in the code to point to where usalign is installed on your device, and will also need to comment out the code that copies USalign into the ```/kaggle/working``` directory):

```sh
conda install -c bioconda usalign
```

## Data

The data we used for our experiments was primarily provided by Kaggle (although we also created our custom datasets, which are stored under ```Data/Custom_datasets``) and can be access using the following link:

https://www.kaggle.com/competitions/stanford-rna-3d-folding/data

The data on this link includes the training, validation and test datasets (although the hidden test dataset that was used for scoring for the competition leaderboard has not been made publicly available yet), as well as data for Multiple Sequence Alignment (MSA).

## Analysis code



## RhoFold+



## RibonanzaNet2



## DRfold2



## Protenix



## Ensembles



## Meta-learner



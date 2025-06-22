# Stanford RNA 3D Folding

This repository contains our work as part of the Stanford RNA 3D Folding Kaggle competition.

Authors: Max Neuwinger, Sarah Verreault, Andreas Hiropedi

## Folder structure

The repository contains the following folders:

- ```Analysis_code``` - this contains the notebooks we used to obtain insights about aspects of the data we were working with, as well as information about model confidence scores and model selection as a motivation for our meta learner
- ```DRfold2``` - this contains all the notebooks we used to test the DRfold2 model (https://github.com/leeyang/DRfold2)
- ```Protenix``` - this contains all the notebooks we used to test the Protenix model (https://github.com/bytedance/Protenix)
- ```RhoFold+``` - this contains all the notebooks we used to test the RhoFold+ model (https://github.com/ml4bio/RhoFold)
- ```RibonanzaNet2``` - this contains all the notebooks we used to test the RibonanzaNet2 model (https://www.kaggle.com/models/shujun717/ribonanzanet2)
- ```Ensembles``` - this contains all our code used for our experiments with different model ensemble strategies
- ```Meta_learner``` - this contains all our code used for our experiments with the meta learner
- ```Custom_datasets``` - this contains our custom datasets that we created following the unforeseen permanent closing of submissions from the Kaggle competition (followin the May 29 deadline)
  

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

5) Since all notebooks also use the **USalign** tool for scoring, you will additionally need to run the following command to install this locally (note: once this is installed locally, you will need to modify all the USalign directory paths in the code to point to where usalign is installed on your device, and will also need to comment out the code that copies USalign into the ```/kaggle/working``` directory):

```sh
conda install -c bioconda usalign
```

## Data

The data we used for our experiments was primarily provided by Kaggle (although we also created our custom datasets, which are stored under ```Data/Custom_datasets``) and can be access using the following link:

https://www.kaggle.com/competitions/stanford-rna-3d-folding/data

The data on this link includes the training, validation and test datasets (although the hidden test dataset that was used for scoring for the competition leaderboard has not been made publicly available yet), as well as data for Multiple Sequence Alignment (MSA).

## RhoFold+

### Additional dependencies

The RhoFold+ notebooks require the following additional dependencies:

- https://www.kaggle.com/code/shosukesuzuki/requirements
- https://www.kaggle.com/datasets/andreashiropedi/rhofold

### Notebooks

- **rhofold+.ipynb** - code for our RhoFold+ baseline experiment

### Instructions

1. Install all dependencies listed in the **Pre-requisites** section

2. Once those are installed, you will also need to install all the additional dependencies listed in this section using the provided links

3. Once everything is installed, you will need to adjust some of the paths in the code to point to the correct directory, e.g., ```/kaggle/input/model-ckpt/RhoFold/pretrained/rhofold_pretrained_params.pt``` will need to be adjusted to the location of the ```RhoFold``` folder on your device (once that dependency is installed). Additionally, you will also need to adjust the paths for the datasets to point to the ```Custom_datasets``` folder in this repository.

4. After adjusting the paths, assuming everything is correctly installed, you can simply run the notebook as a Jupyter notebook.

## RibonanzaNet2

### Additional dependencies

The RibonanzaNet2 notebooks require the following additional dependencies:

- https://www.kaggle.com/datasets/shujun717/ribonanzanet2-ddpm-v2
- https://www.kaggle.com/datasets/shujun717/rnet3d-ddpm
- https://www.kaggle.com/models/shujun717/ribonanzanet2

### Notebooks

- **ribonanzanet2-ddpm-inference.ipynb** - code for our RibonanzaNet2 baseline experiment
- **ribonanzanet2-ddpm-inference-with-msa.ipynb** - code that adds MSA as an input feature to RibonanzaNet2
- **ribonanzanet2-ddpm-inference-with-confidence.ipynb** - code that computes confidence scores for the predictions made by RibonanzaNet2 using the pLDDT (predicted Local Distance Difference Test) metric

### Instructions


1. Install all dependencies listed in the **Pre-requisites** section

2. Once those are installed, you will also need to install all the additional dependencies listed in this section using the provided links

3. Once everything is installed, you will need to adjust some of the paths in the code to point to the correct directory, e.g., ```/kaggle/input/ribonanzanet2/pytorch/alpha/1``` will need to be adjusted to the location of the ```ribonanzanet2``` folder on your device (once that dependency is installed). Additionally, you will also need to adjust the paths for the datasets to point to the ```Custom_datasets``` folder in this repository.

4. After adjusting the paths, assuming everything is correctly installed, you can simply run the notebook as a Jupyter notebook.

## DRfold2

### Additional dependencies

The DRfold2 notebooks require the following additional dependencies:

- https://www.kaggle.com/datasets/andreashiropedi/drfold2
- https://www.kaggle.com/datasets/ogurtsov/biopython

### Notebooks

- **drfold2-no-msa.ipynb** - code for our DRfold2 baseline experiment
- **drfold2-add-msa.ipynb** -  code that adds MSA as an input feature to DRfold2
- **drfold2-with-confidence.ipynb** - code that computes confidence scores for the predictions made by DRfold2 using the pLDDT (predicted Local Distance Difference Test) metric

### Instructions

1. Install all dependencies listed in the **Pre-requisites** section

2. Once those are installed, you will also need to install all the additional dependencies listed in this section using the provided links

3. Once everything is installed, you will need to adjust some of the paths in the code to point to the correct directory, e.g., ```/kaggle/input/drfold/DRfold2/DRfold2/*``` will need to be adjusted to the location of the ```DRfold2``` folder on your device (once that dependency is installed). Additionally, you will also need to adjust the paths for the datasets to point to the ```Custom_datasets``` folder in this repository.

4. After adjusting the paths, assuming everything is correctly installed, you can simply run the notebook as a Jupyter notebook.

## Protenix

### Additional dependencies

The Protenix notebooks require the following additional dependencies:

- https://www.kaggle.com/datasets/maxneuwinger/protenix-public
- https://www.kaggle.com/datasets/maxneuwinger/protenix-wheel
- https://www.kaggle.com/datasets/maxneuwinger/protenix-nomsa-75crop-3000steps-ema
- https://www.kaggle.com/datasets/maxneuwinger/7ksteps-protenix-nomsa-75cropping
- https://www.kaggle.com/datasets/maxneuwinger/1000steps-256-crop-size

### Notebooks

- **protenix-baseline.ipynb** - code for our Protenix baseline experiment
- **protenix-baseline-withmsa.ipynb**- code that adds MSA as an input feature to Protenix
- **protenix-baseline-with-confidence.ipynb** - code that computes confidence scores for the predictions made by Protenix using the pLDDT (predicted Local Distance Difference Test) metric
- **protenix-finetuned-7000steps-nomsa-75crop.ipynb** - code for our Protenix finetuning experiments without MSA
- **protenix-finetuned-3000steps-nomsa-75crop.ipynb** - code for our Protenix finetuning experiments without MSA
- **protenix-finetuned-crop256-withmsa-1ksteps.ipynb** - code for our Protenix finetuning experiments with added MSA as an input feature

### Instructions

1. Install all dependencies listed in the **Pre-requisites** section

2. Once those are installed, you will also need to install all the additional dependencies listed in this section using the provided links

3. Once everything is installed, you will need to adjust some of the paths in the code to point to the correct directory, e.g., ```/kaggle/input/protenix-public/af3-dev``` will need to be adjusted to the location of the ```protenix-public``` folder on your device (once that dependency is installed). Additionally, you will also need to adjust the paths for the datasets to point to the ```Custom_datasets``` folder in this repository.

4. After adjusting the paths, assuming everything is correctly installed, you can simply run the notebook as a Jupyter notebook.

## Analysis code

### Notebooks

- **dataframe-comparing-predictions-confidences.ipynb** - this code computes the average confidence that each model has in its predictions

### Instructions

1. Install all dependencies listed in the **Pre-requisites** section

2. 

## Ensembles

### Notebooks

- **validationpredictions.ipynb** - code that generates predictions for all of our different ensemble strategies and stores them in CSV files
- **ensembles-scoring.ipynb** - code that generates the TM-scores for the ensemble predictions for  all of our different ensemble strategies

### Instructions

1. Install all dependencies listed in the **Pre-requisites** section

2. 

## Meta-learner

### Notebooks

- 

### Instructions

1. Install all dependencies listed in the **Pre-requisites** section

2. 

# Stanford RNA 3D Folding

This repository contains our work as part of the Stanford RNA 3D Folding Kaggle competition.

Authors: Max Neuwinger, Sarah Verreault, Andreas Hiropedi

## Folder structure

The repository contains the following folders:

- ```Data_Analysis_code``` - this contains the notebooks we used to obtain insights about aspects of the data we were working with, as well as information about model confidence scores and model selection as a motivation for our meta learner
- ```ModelNotebooks/DRfold2``` - this contains all the notebooks we used to test the DRfold2 model (https://github.com/leeyang/DRfold2)
- ```ModelNotebooks/Protenix``` - this contains all the notebooks we used to test the Protenix model (https://github.com/bytedance/Protenix)
- ```ModelNotebooks/RhoFold+``` - this contains all the notebooks we used to test the RhoFold+ model (https://github.com/ml4bio/RhoFold)
- ```ModelNotebooks/RibonanzaNet2``` - this contains all the notebooks we used to test the RibonanzaNet2 model (https://www.kaggle.com/models/shujun717/ribonanzanet2)
- ```SimpleEnsembles``` - this contains all our code used for our experiments with the different simple/heuristic model ensemble strategies
- ```Datasets``` - contains the csvs of the Kaggle data and our own custom versions with temporal splits and quality filtering
- ```Datasets_Scripts``` - contains the python scripts that we used to create our custom dataset splits and prepare the Kaggle MSA data for the Protenix format
- ```Helper_Scripts``` - contains small helper scripts responsible for evalauting multiple submissions and creating the finetuning plots from the raw Training Output
- ```MetaModel``` - this contains all our code used for our experiments with the meta model
- ```PredictionData_ForMetaModel``` - contains the predicitons of our different models used for the training and testing of the Meta Model
- ```Protenix_Finetuned_CustomTestSet_Results``` - contains the results of the different Protenix finetuned weights for our custom test set
- ```Protenix_Finetuning_WholeCode``` - contains the whole code we used to finetune and test Protenix
  

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

The data we used for our experiments was primarily provided by Kaggle (although we also created our custom datasets) and can be access using the following link:

https://www.kaggle.com/competitions/stanford-rna-3d-folding/data

The data on this link includes the training, validation and test datasets (although the hidden test dataset that was used for scoring for the competition leaderboard has not been made publicly available yet), as well as data for Multiple Sequence Alignment (MSA).

## Reproducing ProteinX Fine-Tuning Results

This section outlines the steps required to set up the environment and reproduce our ProteinX fine-tuning experiments.

### 1. Environment Setup

To set up the environment, please follow these steps:

1.  Install the required dependencies using the provided `setup.py` script. It is recommended to do this in an "editable" mode.
    ```bash
    pip install -e .
    ```
2.  Install the `pandas` library.
    ```bash
    conda install pandas
    ```

### 2. Summary of Code Modifications

We made several key modifications to the original ProteinX codebase to support our experiments. The main changes are located in the following files:

*   `configs/configs_data.py`: We added our custom `kaggle_rna3d` dataset configuration, which specifies paths to the competition data and parameters for our data loader.
*   `protenix/data/kaggle_rna_dataset.py`: This is a new file containing our custom `Dataset` class. It handles loading the Kaggle RNA data, performing data integrity checks, and interfacing with the MSA pipeline.
*   `protenix/data/msa_featurizer.py`: We adapted this file to include custom RNA MSA featurization logic, allowing the model to process the provided RNA alignments.
*   `protenix/data/json_to_feature.py`: This file was slightly edited to ensure the atom permutation list was compatible with the RNA data and competition input format.
*   `runner/train.py`: The training script was directly modified to use our custom `KaggleRNADataset` class as the data loader for the fine-tuning process.

### 3. Fine-Tuning

The fine-tuning experiments can be launched via a bash script, following the standard procedure outlined in the official ProteinX documentation.

1.  **Download Pretrained Weights:** Start from the official `model_v0.2.0.pt` pretrained weights provided by the Protenix authors.

2.  **Run Training Script:** Use a command similar to the one below, ensuring you point to the pretrained weights and specify our `kaggle_rna3d` dataset.

    ```bash
    torchrun --nproc_per_node=<num_gpus> runner/train.py \
        --run_name <your_run_name> \
        --base_dir /path/to/save/runs \
        --load_checkpoint_path /path/to/model_v0.2.0.pt \
        --data.train_sets kaggle_rna3d \
        --data.test_sets kaggle_rna3d \
        # ... other custom hyperparameters as specified in our report ...
    ```

    **Note on Ablation Study:** To run the ablation experiment (fine-tuning *without* MSAs), simply set the `enable_rna_msa` flag in the `configs/configs_data.py` file to `False` before launching the training script.

### 4. Inference

Once you have fine-tuned model checkpoints, you can use the provided notebooks to generate predictions:

*   `kaggle_comp_inference.ipynb`: Use this notebook to generate predictions with a model that was fine-tuned **without** MSA data.
*   `kaggle_comp_inference_MSA.ipynb`: Use this notebook to generate predictions with a model that was fine-tuned **with** MSA data.

These notebooks are configured to generate submission files for the Kaggle platform and can also be used to evaluate the models locally on our custom test set.

## Reproducing Meta-Model Ensemble Results

This section outlines the steps to train our meta-learning model and use it to generate an ensembled submission file. The process involves generating a feature set from the predictions of multiple base models, training a neural network to predict structure quality, and then using this network to select the best candidates for the final submission.

### 1. Dependencies

Ensure you have the following core Python libraries installed. You can install them using `pip` or `conda`.

```
pandas
numpy
scikit-learn
torch
joblib
tqdm
matplotlib
seaborn
lightgbm  # For the LightGBM version of the notebook
```

### 2. Workflow Overview

The process is divided into two main stages, each handled by a dedicated Jupyter notebook:

1.  **Training the Meta-Model:** Using predictions on the *training set* to train the quality assessment model.
2.  **Inference:** Using the trained model to score predictions on the *test set* and create a final submission file.

### 3. Step-by-Step Instructions

#### Step 1: Prepare the Input Data

Before running the notebooks, you must first generate and organize the predictions from the three base models: **ProteinX**, **DrFold2**, and **RibonanzaNet2**.

1.  Run each base model on the **training sequences** to generate their respective `submission.csv`, `confidence.csv`, and (if applicable) `ranking_scores.csv` files.
2.  Run each base model on the **test sequences** (our custom 94-sample set) to generate a separate set of prediction files.
3.  Place all these prediction files in organized directories.
4.  Update the file paths in the configuration cells at the top of both notebooks to point to your data files.

#### Step 2: Train the Meta-Model

Run the `01_Meta_Learner_Training.ipynb` notebook. This notebook will:

1.  Load the predictions and ground truth labels for the **training set**.
2.  Perform extensive **feature engineering**, calculating 28 distinct features (including biophysical energies, structural properties, and ensemble consensus metrics) for every candidate structure.
3.  Train the neural network-based meta-model.
4.  Save the two essential outputs:
    *   `meta_learner_model.pth`: The trained PyTorch model weights.
    *   `meta_learner_scaler.pkl`: The `StandardScaler` object fitted on the training data.

#### Step 3: Generate the Final Ensemble Submission

Run the `02_Meta_Learner_Inference.ipynb` notebook. This notebook will:

1.  Load the predictions for the **test set**.
2.  Perform the same feature engineering steps as the training notebook.
3.  Load the `meta_learner_model.pth` and `meta_learner_scaler.pkl` files from Step 2.
4.  Use the trained model to predict a TM-score for every candidate structure in the test set.
5.  Select the top 5 candidates with the highest predicted scores for each target RNA.
6.  Assemble the coordinate data from these selected candidates into a final `ensembled_submission.csv` file.

This `ensembled_submission.csv` is the final output of our meta-learning ensemble approach. An additional notebook is also provided, which follows a similar workflow but uses a LightGBM model instead of a neural network.

## RhoFold+

### Additional dependencies

The RhoFold+ notebooks require the following additional dependencies:

- https://www.kaggle.com/code/shosukesuzuki/requirements
- https://www.kaggle.com/datasets/andreashiropedi/rhofold

### Notebooks

- ```rhofold.ipynb``` - code for our RhoFold+ baseline experiment

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

- ```ribonanzanet2-ddpm-inference.ipynb``` - code for our RibonanzaNet2 baseline experiment
- ```ribonanzanet2-ddpm-inference-with-msa.ipynb``` - code that adds MSA as an input feature to RibonanzaNet2
- ```ribonanzanet2-ddpm-inference-with-confidence.ipynb``` - code that computes confidence scores for the predictions made by RibonanzaNet2 using the pLDDT (predicted Local Distance Difference Test) metric

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

- ```drfold2-no-msa.ipynb``` - code for our DRfold2 baseline experiment
- ```drfold2-add-msa.ipynb``` -  code that adds MSA as an input feature to DRfold2
- ```drfold2-with-confidence.ipynb``` - code that computes confidence scores for the predictions made by DRfold2 using the pLDDT (predicted Local Distance Difference Test) metric

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

- ```protenix-baseline.ipynb``` - code for our Protenix baseline experiment
- ```protenix-baseline-withmsa.ipynb``` - code that adds MSA as an input feature to Protenix
- ```protenix-baseline-with-confidence.ipynb``` - code that computes confidence scores for the predictions made by Protenix using the pLDDT (predicted Local Distance Difference Test) metric
- ```protenix-finetuned-7000steps-nomsa-75crop.ipynb``` - code for our Protenix finetuning experiments without MSA
- ```protenix-finetuned-3000steps-nomsa-75crop.ipynb``` - code for our Protenix finetuning experiments without MSA
- ```protenix-finetuned-crop256-withmsa-1ksteps.ipynb``` - code for our Protenix finetuning experiments with added MSA as an input feature

### Instructions

1. Install all dependencies listed in the **Pre-requisites** section

2. Once those are installed, you will also need to install all the additional dependencies listed in this section using the provided links

3. Once everything is installed, you will need to adjust some of the paths in the code to point to the correct directory, e.g., ```/kaggle/input/protenix-public/af3-dev``` will need to be adjusted to the location of the ```protenix-public``` folder on your device (once that dependency is installed). Additionally, you will also need to adjust the paths for the datasets to point to the ```Custom_datasets``` folder in this repository.

4. After adjusting the paths, assuming everything is correctly installed, you can simply run the notebook as a Jupyter notebook.

## Analysis code

### Notebooks

- ```dataframe-comparing-predictions-confidences.ipynb``` - this code computes the average confidence that each model has in its predictions

### Instructions

1. Install all dependencies listed in the **Pre-requisites** section

2. Once all dependencies are installed, you will need to adjust the paths for the validation labels dataset to point to the ```Custom_datasets``` folder in this repository.

3. Additionally, to perform the analysis in the ```dataframe-comparing-predictions-confidences.ipynb``` notebook, you will need to first generate the predictions with confidence scores for each of the three models (RibonanzaNet2, Protenix and DRfold2). Therefore, you will need to follow the setup instructions for each of the models and run the following three notebooks and save their outputs as CSV files:

- ```protenix-baseline-with-confidence.ipynb```
- ```drfold2-with-confidence.ipynb```
- ```ribonanzanet2-ddpm-inference-with-confidence.ipynb```

Note that, after running these notebooks, you will also need to adjust the paths that point to the CSV files for the predictions (e.g., ```/kaggle/input/predictions/drfold2_submission_with_confidence.csv``` will need to be modified to your local directory where this file is stored).

## Simple Ensembles

### Notebooks

- ```validationpredictions.ipynb``` - code that generates predictions for all of our different ensemble strategies and stores them in CSV files
- ```ensembles-scoring.ipynb``` - code that generates the TM-scores for the ensemble predictions for  all of our different ensemble strategies

### Instructions

1. Install all dependencies listed in the **Pre-requisites** section

2. Once all dependencies are installed, you will need to adjust the paths for the validation labels dataset to point to the ```Custom_datasets``` folder in this repository.

3. Additionally, to be able to successfully run both notebooks, you will need to first generate the predictions with confidence scores for each of the three models (RibonanzaNet2, Protenix and DRfold2). Therefore, you will need to follow the setup instructions for each of the models and run the following three notebooks and save their outputs as CSV files:

- ```protenix-baseline-with-confidence.ipynb```
- ```drfold2-with-confidence.ipynb```
- ```ribonanzanet2-ddpm-inference-with-confidence.ipynb```

Note that, after running these notebooks, you will also need to adjust the paths that point to the CSV files for the predictions (e.g., ```/kaggle/input/predictions/drfold2_submission_with_confidence.csv``` will need to be modified to your local directory where this file is stored).

4. Once you have generated the CSV output files and adjusted the paths, first run the ```validationpredictions.ipynb``` notebook, as this contains the ensemble implementation code. 

5. After running this notebook, you will need to again save the outputs as CSV files and adjust the paths in the ```ensembles-scoring.ipynb``` notebook before running the code. (e.g., ```/kaggle/input/naive-set/naive.csv``` will need to be modified to your local directory where this file is stored). 

6. Following this second directory path adjustment, you can safely execute the code in the ```ensembles-scoring.ipynb``` notebook to obtain TM-score values for all the ensemble strategies.

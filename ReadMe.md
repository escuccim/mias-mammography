# EXTS Capstone Project

Eric Scuccimarra (skooch@gmail.com)

2018-05-04

------

This project was far too complex and wide-ranging to fit into a single notebook. The files included in this repository are listed below:

### Overview

- Report.md - describes the steps taken during this project as well as summarizes the results

### Exploratory Data Analysis

- Wisconsin (UCI) EDA.ipynb - exploratory data analysis on the Wisconsin Breast Cancer data from the UCI Machine Learning Repository
- SVM.ipynb, kNN.ipynb, Decision Trees.ipynb, Multilayer neural networks.ipynb - standard machine learning techniques applied to the Wisconsin Breast Cancer data
- UCI Results.ipynb - results of the above notebooks consolidated
- MIAS Exploratory Data Analysis.ipynb - exploratory data analysis of the MIAS data and images

### DDSM Data Preprocessing

- overview_of_image_processing.md - overview of the challenges posed by and steps taken to turn the CBIS-DDSM and DDSM data into usable images
- /Decompressing-For-LJPEG-image/ - code from GitHub used to convert the DDSM LJPEG files into PNGs
- crop_cbis_images_x.ipynb - code used to create extract the ROIs from the CBIS-DDSM data to create dataset x
- crop_normal_images_x.ipynb - code used to create training data from the normal, DDSM images for dataset x
- crops_mias_images_x.ipynb - code used to create a supplementary test dataset from the MIAS data for dataset x
- review_images_x.ipynb - review random sampling of training images for dataset x to identify potential problems
- write_to_tfrecords_x.ipynb - combines the images from the CBIS-DDSM and DDSM datasets and writes tfrecords files used for training
- mammo_utils.py - various functions shared across notebooks

### DDSM ConvNet Training 

- training_utils.py - various functions used in creating and training convnets
- candidate_1.0.0.x.py - python scripts used to create and train various candidate models. Only the models referenced in the report are included here.
- vgg_16.3.py - a customized version of VGG evaluated
- inception_v4.05.py - a model based on Inception, but significantly scaled down
- inception_utils.py - functions used to create our Inception clone

### DDSM Training Results

- /logs/ - TensorBoard logs for selected training runs of selected models
- model_notes.xlsx - notes on the various runs of training various models, including hyperparameters and results
- ddsm_results.csv - consolidated results for selected training runs
- ddsm_results.ipynb - the csv file imported into a notebook and sorted
- convnet_training_metrics.ipynb - training and validation metrics for selected training runs. The metrics used to generate this notebook are saved as .npy files in /data/results/
- convnet_1.0.0.35b.ipynb - code used to create and train our best model in a notebook. Includes evaluation of the trained model on the test and MIAS datasets. Note that the model checkpoints are too large to upload to GitHub so must be downloaded. URLs are provided below.

### Additional Resources

- The training, validation and test datasets used in this work are available for download on Amazon S3. To download them use training_utils.download_data(what=x) where x is the dataset you wish to download. Each dataset is split into 5 zip files containing the tfrecords and 4 npy files containing the test and validation data and images. Be aware that the unzipped size of the data is several gigabytes.
- The following pre-trained models are also available:
  - Model 1.0.0.29 trained for multi-class classification on dataset 8: https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/model_s1.0.0.29l.8.2.zip
  - Model 1.0.0.35b.98 trained for binary classification on dataset 9: <https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/model_s1.0.0.35b.98.ckpt.zip> 
  - Model 1.0.0.35b.96 trained for binary classification on dataset 9: <https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/model_s1.0.0.35b.96.bu30.zip> 
- If there are any files which are missing from this repository they should be included in one of the two following repositories:
  - https://github.com/escuccim/mias-mammography - contains the notebooks and code run locally for EDA, data pre-processing and analysis of results
  - https://github.com/escuccim/mammography-models - was used to sync training code between my laptop and the Google GPU instances where the models were trained.


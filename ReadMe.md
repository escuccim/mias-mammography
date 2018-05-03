# ConvNets for Detection of Abnormalities in Mammograms
Eric Scuccimarra (skooch@gmail.com)

## Introduction
Breast cancer is the second most common cancer in women worldwide. About 1 in 8 U.S. women (about 12.4%) will develop invasive breast cancer over the course of her lifetime. The five year survival rates for stage 0 or stage 1 breast cancers are close to 100%, but the rates go down dramatically for later stages: 93% for stage II, 72% for stage III and 22% for stage IV. Human recall for identifying lesions is estimated to be between 0.75 and 0.92 [1], which means that as many as 25% of abnormalities may go undetected. 

The DDSM is a well-known dataset of normal and abnormal scans, and one of the few publicly available datasets of mammography imaging. Unfortunately, the size of the dataset is relatively small. To increase the amount of training data we extract the Regions of Interest (ROI) from each image, perform data augmentation and then train ConvNets on the augmented data. The ConvNets were trained to predict both whether a scan was normal or abnormal, and to predict whether abnormalities were calcifications or masses and benign or malignant.

## Related Work
There exists a great deal of research into applying deep learning to medical diagnosis, but the lack of available training data is a limiting factor, and thus there is not much research into applying ConvNets to mammography. [1, 4] use ConvNets to classify pre-detected breast masses by pathology and type, but do not attempt to detect masses from scans. [2,3] detect abnormalities using combinations of region-based CNNs and random forests. 

## Datasets
The DDSM [6] is a database of 2,620 scanned film mammography studies. It contains normal, benign, and malignant cases with verified pathology information. The DDSM is saved as Lossless JPEGs, an archaic format which has not been maintained in about 20 years. The CBIS-DDSM [8] collection includes a subset of the DDSM data selected and curated by a trained mammographer. The CBIS-DDSM images have been pre-processed and saved as Dicom images, and thus are better quality than the DDSM images, but this dataset only contains scans with abnormalities. In order to create a dataset which can be used to predict the presence of abnormalities, the ROIs were extracted from the CBIS-DDSM dataset and combined with normal images taken from the DDSM dataset.

The MIAS dataset is a very small set of mammography images, consisting of 330 scans of all classes. The scans are standardized to a size of 1024x1024 pixels. The size of the dataset made this unusable for training, but it was used for exploratory data analysis and as a supplementary test data set.

Data from the University of California Irvine Machine Learning Repository [5] was also used for exploratory data analysis to gain insight into the characteristics of abnormalities.

## Methods

### Data Preprocessing and Augmentation
In order to create a training dataset of adequate size images from the CBIS-DDSM dataset were combined with images from the DDSM dataset. As the raw images are not of a standard size we extracted 299x299 tiles from each raw scan. For the CBIS-DDSM images the provided masks were used to extract the Region of Interest (ROI). For the DDSM images tiles were created from each image and used as long as they met certain basic criteria.

Data augmentation was used to increase the size of the datasets, with multiple datasets created with different techniques used to extract the ROI and different amounts of data augmentation.

#### Training Datasets
Multiple datasets were created using different techniques and amounts of data augmentation. The datasets ranged in size from 27,000 training images to 62,000 training images. 

1. Dataset 6 consisted of 62,764 images. This dataset was created to be as large as possible, and each ROI is extracted multiple times in multiple ways using both ROI extraction methods described below. Each ROI was extracted with fixed context, with padding, at its original size, and if the ROI was larger than our target image it was also extracted as overlapping tiles. 
2. Dataset 8 consisted of 40,559 images. This dataset used the extraction method 1 described below to provide greater context for each ROI. This dataset was created for the purpose of classifying the ROIs by their type and pathology.
3. Dataset 9 consisted of 43,739 images. The previous datasets had used zoomed images of the ROIs, which was problematic as it required the ROI to be pre-identified and isolated. This dataset was created using extraction method 2 described below.

### ROI Extraction Methods for CBIS-DDSM Images
The CBIS-DDSM scans were of relatively large size, with a mean height of 5295 pixels and a mean width of 3131 pixels. Masks highlighting the ROIs were provided. The masks were used to extract the ROI by creating a square around the white areas of the mask. 

Using the mask, the ROIs were extracted at 598x598 and then sized down to 299x299. To increase the size of the training data, each ROI was extracted multiple times using the methodologies described below. The size and variety of the data was also increased by randomly horizontally flipping each tile, randomly vertically flipping each tile, randomly rotating each tile, and by randomly positioning each ROI within the tile.

#### ROI Extraction Method 1
The analysis of the UCI data indicated that the edges of an abnormality were important as to determining its pathology and type, and this was confirmed by a radiologist. Levy et al [1] also report that the inclusion of context was an important contributor to the accuracy of the classification.

The ROIs had a mean size of 450 pixels and a standard deviation of 396. In the interest of representing each ROI as well as possible, each ROI was extracted in multiple ways:

1.	The ROI was extracted at 598x598 at its original size.
2.	The ROI was zoomed to 598x598, with padding to provide context.
3.	If the ROI had the size of one dimension more than 1.5 times the other dimension it was extracted as two tiles centered in the center of each half of the ROI along it's largest dimension.

#### ROI Extraction Method 2
The previous datasets relied on having the ROIs pre-identified and the ROI was then zoomed and cropped. While this provided very good images of the ROIs, the fact that each ROI was zoomed in or out made the data created too artificial to be applicable to using raw scans as input. This dataset left each ROI at its natural size for the purpose of training a model which could recognize abnormalities in un-processed scans.

1. If the ROI was smaller than a 598x598 tile it was extracted with 20% padding on either side. 
2. If the ROI was larger than a 598x598 tile it was extracted with 5% padding.
3. Each ROI was then randomly cropped three times using random flipping and rotation. 
4. The resulting images were then sized down to 299x299.

#### Normal Images
The normal scans from the DDSM dataset did not have ROIs so were processed differently. As these images had not been pre-processed they contained artifacts such as white borders and white patches of pixels used to cover up identifying personal information. To remove the borders each image was cropped by 7% on each side. 

To keep the normal images as similar to the CBIS-DDSM images, different pre-processing was done for each dataset created. For datasets 6 and 8, the DDSM images were sized down by a random factor between 1.8 and 3.2, then segmented into 299x299 tiles with a variable stride between 150 and 200. Each tile was then randomly rotated and flipped.
 
For dataset 9, each DDSM image was cut into 598x598 tiles with the size unchanged, and then the tiles were sized down to 299x299.
 
To avoid the inclusion of images which contained overlay text, white borders or artifacts, or were mostly background, each tile was then added to the dataset only if it met upper and lower thresholds on mean and variance. The thresholds were determined through random sampling of tiles and tuning of the thresholds to eliminate images which did not contain usable data. 

### MIAS Images
Supplementary test datasets were also created from the MIAS images. As these images are from a completely different distribution than the DDSM images, we felt they would provide a good assessment of how well the models would generalize. The MIAS images were a uniform size of 1024x1024. The images were increased in size by 2.58 so that their height was the same as half the mean height of the DDSM images. The ROIs were then extract using the same methods used for the CBIS-DDSM images except the ROIs were extracted directly at 299x299 to avoid losing information by resizing the images up and down.

#### Data Balance
In reality, only about 10% of mammograms are abnormal, in order to maximize recall our datasets were weighted more heavily towards abnormal scans, with a final distribution of 83% normal and 17% abnormal. As each ROI was extracted to multiple images, in order to prevent different images of the same ROI appearing in both the training and test data, the existing divisions of the CBIS-DDSM data were maintained. The test data was divided evenly between test and validation data with no shuffling to avoid overlap.

The normal images had no overlap, so were shuffled and divided among the training, test and validation data. The final divisions were 80% training, 10% test and 10% validation. We would have liked to have larger test and validation datasets, but we found it easier to use the existing train/test divisions in the CBIS-DDSM data.

### Labels
In the CBIS-DDSM dataset the scans are grouped into the following categories:
1.	Normal
2.	Benign Calcification
3.	Malignant Calcification
4.	Benign Mass
5.	Malignant Mass

Although we did experiment with classifying images directly into all five categories, as previous work [1] has already dealt with classifying pre-identified abnormalities, we focused on classifying images as normal or abnormal.

### ConvNet Architecture
Our first thought was to train existing ConvNets, such as VGG or Inception, on our datasets. These networks were designed and trained on ImageNet data, which has a much greater number of classes than our datasets do and the images have a great deal more variance than do medical scans. For this reason we felt that models with large numbers of parameters might easily overfit our data. A lack of computational resources also made training VGG or Inception on our datasets impractical. For these reasons we designed our own architectures specifically for this work.

We started with a simple model based on VGG, consisting of stacked 3x3 convolutional layers alternating with max pools followed by three fully connected layers. Our model had fewer convolutional layers with less filters than VGG, and smaller fully connected layers. We also added batch normalization [15] after every layer. This architecture was then evaluated and adjusted iteratively, with each iteration making one and only one change and then being evaluated. We also evaluated techniques including Inception-style branches [16, 17, 18] and residual connections [19]. 

To compensate for the unbalanced nature of the dataset a weighted cross-entropy function was used, weighting positive examples higher than negative ones. The weight was a hyperparameter for which we tested values ranging from 1 to 4.

The best performing architecture will be detailed below.

### Training
At first, initial evaluation of models was done using Dataset 5 due to its relatively small size. Each model was trained through between 30 and 50 epochs and accuracy, accuracy, precision, recall and f1 score were used to evaluate performance. 

Due to the fact that Dataset 5 was created without proper separation of the training and test datasets, it is likely that variations of the same images were including in both the training and validation datasets, making the validation dataset essentially useless. Once we realized this we stopped using Dataset 5 and started evaluating with Datasets 8 and 9.

As Dataset 9 best represented our goals for this work, for the final phases we trained and evaluated models exclusively on this dataset.

### Online Data Augmentation

We had built in optional online data augmentation to our graphs, but enabling it slowed down training significantly as it ran on the CPU rather than the GPU and thus became a bottleneck. We found code online [22] which enabled the GPU to handle the data augmentation and eliminated the bottleneck. Enabling this applied random horizontal and vertical flips to the images, which improved performance and allowed the models to be trained for longer by reducing overfitting. As enabling this did slow down the training speed typically we only used it for the last few epochs of training.  

### Transfer Learning
We had considered using transfer learning from VGG or Inception, but felt that the features of the ImageNet data were different enough from those of radiological scans that it made more sense to learn the features from scratch on this dataset. To evaluate the usefulness of transfer learning from pre-trained networks, VGG-19 and Inception v3, the features were extracted from our datasets using pre-trained versions of these models. The final layers of the networks were then retrained to classify to our classes while the convolutional weights were frozen. Our hypothesis that the features extracted by these networks would not be applicable to classifying medical scans seemed to be confirmed by the results of this experiment, which were significantly below the accuracy of the most-frequent baseline.

Transfer learning did, however, proof very useful in training our own networks, where initializing the convolutional weights to weights pre-trained on our datasets greatly sped up the training process. However, this method was not entirely reliable. When trained past 15-25 epochs or so, most of our models began to overfit the training data, and continuing to train a pre-trained model usually led to substantial drops in validation accuracy unless the convolutional weights were frozen. Our approach to handling this was to train a model on multi-class classification for 15-20 epochs, under the assumption that this would train the network to extract the most relevant features, and then re-training the network for binary classification for 5-10 epochs. The re-training was done both with and without the convolutional layers frozen.   

## Results
### Architecture
We evaluated a large number of models on our datasets, including customized versions of VGG-16 and Inception v4. The 1.0.0.x family of custom models performed the best.

* Model 1.0.0.29 had nine convolutional layers and three fully connected layers. The convolutional layers used the philosophy of VGG, with 3x3 convolutions stacked and alternated with max pools.
* Model 1.0.0.35 was identical to 1.0.0.29 but with the input data scaled to be between 0 and 1.
* Model 1.0.0.46 also had the input data centered to the mean.

In models 1.0.0.35 and 1.0.0.45 we also added optional online contrast adjustment and online data augmentation.

<img src="model_s1.0.0.29.png" alt="Model 1.0.0.29" align="right" style="max-width: 50%;">

A variety of other models were evaluated which included things like Inception-style branches and residual connections, but while these models learned the training data they did not generalize well to the validation or test datasets. We suspect that the size of these models may have led to overfitting of the training data.

A slightly modified version of VGG-16 was also trained as a benchmark. A full version of VGG-16 required more memory than we had available, so we made the following alterations to the architecture:

1. The architecture was altered to accept 299x299 images as input
2. The stride of the first convolutional layer was changed to 2
3. The first fully connected layer had 1024 units instead of 4096, and the second fully connected layer had 512 units.
4. Batch normalization was included after every layer

These changes brought the memory requirements down to acceptable levels and doubled the training speed. While changing the stride of convolutional layer 1 decreased the training accuracy, we felt that it might allow the model to generalize a bit better.

Finally we attempted to train a customized version of Inception v4 on our datasets. Our version had the number of filters in each convolutional layer cut in half and we replaced the global average pooling with a flattening and then one fully-connected layer. The time required to train this model made it impractical to work on further, and the results we obtained did not merit any further exploration.

### Performance
Different models performed differently on different datasets with different classification methods. Unsurprisingly, all models on all datasets performed better at binary classification than on multi-class classification. The complete results for all of our models evaluated are in the Excel spreadsheet "model notes.xlsx." 

Table 1 below shows the accuracy and recall on the test dataset for selected models trained for binary classification. The most-frequent baseline accuracy for the datasets was .83. We should note that, with the exception of 1.0.0.29, the models that achieve perfect recall have accuracies which indicate they are predicting everything as positive, and the models that have very low recall are predicting everything as negative.

|Model          |Classification |Dataset    |Epochs |Accuracy    |Recall      |Initialization |
|---------------|---------------|-----------|-------|------------|------------|---------------|
|1.0.0.46b      |         Binary|          6|20     |.1810       |1.0         |Scratch        |
|1.0.0.29n      |         Binary|          8|30     |.9930       |1.0         |Scratch        |
|1.0.0.35b      |         Binary|          8|20     |.9678       |.8732       |Scratch        |
|1.0.0.46b      |         Binary|          8|10     |.9896       |.9776       |1.0.0.46l.6    |
|1.0.0.35b      |         Binary|          9|20     |.9346       |.8998       |Scratch        |
|1.0.0.46b      |         Binary|          9|30     |.8370       |.0392       |Scratch        |
|VGG-16.03.04b.8|         Binary|          8|10     |.8747       |.2951       |VGG-16.03.04l6 |
|VGG-16.03.04b.9|         Binary|          9|30     |.8881       |.3589       |Scratch        |
|inception_v4.05b.9|      Binary|          9|20     |.1828       |1.0         |Scratch        |
<small>\* only fully connected layers re-trained</small>              
<div style="text-align:center;"><i>Table 1: Binary Performance on Test Set</i></div><br>

Table 2 below shows the accuracy and recall of the test dataset for selected training runs for multi-class classification. We again see that, with the exception of 1.0.0.29n, all of the models tend to predict everything as either positive or negative, resulting in a recall close to 1 or close to 0 and accuracy close to the baseline.

|Model          |Classification |Dataset    |Epochs |Accuracy    |Recall      |Initialization |
|---------------|---------------|-----------|-------|------------|------------|---------------|
|1.0.0.29n      |    Multi-class|          6|40     |.9142       |.9353       |Scratch        |  
|1.0.0.46l.8    |    Multi-class|          8|20     |.1139       |1.0         |Scratch        |
|1.0.0.46l.6    |    Multi-class|          6|20     |.8187       |0           |Scratch        |
|VGG-16.03.04l.6|    Multi-class|          6|20     |.8333       |.0288       |Scratch        | 
<div style="text-align:center;"><i>Table 2: Multi-class Performance on Test Set</i></div><br>

<p>Figure 1 shows the training metrics for model 1.0.0.29 trained on dataset 8 for binary classification.

<img src="1.0.0.29_results.png" alt="Binary Accuracy and Recall of Model 1.0.0.29 on Dataset 8" align="center" style="height: 200px;"><br>
<i>Figure 1 - Binary Accuracy and Recall for Model 1.0.0.29 on Dataset 8</i>

Figure 2 shows the training metrics for model 1.0.0.45 trained on dataset 9 for binary classification. The validation performance seems tied to the training performance through epoch 15 after which both it and the recall drop to the baseline as the model learns to predict everything as negative.

<img src="1.0.0.45b.9.1_results.png" alt="Binary Accuracy and Recall of Model 1.0.0.45 on Dataset 9" align="center"><br>
<i>Figure 2 - Binary Accuracy and Recall for Model 1.0.0.45 on Dataset 9</i>

Figure 3 shows the training metrics for model 1.0.0.35 trained on dataset 9 for binary classification. The validation accuracy tracks the training accuracy much more closely than did 1.0.0.45 although the recall remains volatile.

<img src="1.0.0.35b.9_results.png" alt="Binary Accuracy and Recall of Model 1.0.0.35 on Dataset 9" align="center"><br>
<i>Figure 3 - Binary Accuracy and Recall for Model 1.0.0.35 on Dataset 9</i> 

While model 1.0.0.29 achieved the best results, we were unable to replicate this performance when retraining the model on the same dataset. We feel that model 1.0.0.35 was the best performing model in achieving a combination of accurracy and recall that we were able to duplicate. Model 1.0.0.35 also achieved good performance on the MIAS datasets, which indicates that the model can generalize to unrelated scans. Table 3 below shows the accuracy and recall of selected models on MIAS dataset 9, which should track the ability of the models to generalize to unrelated, unaugmented scans.

|Model          |Training Dataset   |MIAS Accuracy    |MIAS Recall      |
|---------------|-------------------|-----------------|-----------------|
|1.0.0.35b.96   |9                  |.7156            |.8905            |
|vgg_16.3.04b.9 |9                  |.9314            |.6517            |
|1.0.0.28.2b.9  |9                  |.9165            |.5342            |
|1.0.0.46b.8.4  |8                  |.2746            |.9811            |

<div style="text-align:center;"><i>Table 3: Performance on MIAS Dataset</i></div><br>

### Effect of Cross Entropy Weight
As mentioned above, a weighted cross entropy was used to improve recall and counter the unbalanced nature of our dataset. Increasing the weight improved recall at the expense of precision. Without the weighted cross entropy our models tended to end up with a precision of 1 and a recall of 0, predicting everything in the validation set as negative.

The weighted cross entropy also helped with the issue of overfitting the training data. With our unbalanced dataset it was easier to minimize the unweighted cross entropy by predicting everything as normal than to learn to identify the abnormal examples. By weighting abnormal examples higher the model was forced to learn to recognize them.    

### Effect of Input Data Scaling
We had attempted to scale and center the input data when creating the datasets but the size of the dataset made this impossible. As a result our first models, including 1.0.0.29 took raw pixel data, between 0 and 255 as input. Models 1.0.0.4x were the same architecture as 1.0.0.29 but with the data centered online by subtracting the pre-calculated mean. This improved the training results but the validation results became much more volatile and seemingly unrelated to the training results. 

To investigate this we removed the online data centering and scaling and retrained the model. We were surprised to see the validation performance become more stable so we then tested various permutations of centering and scaling the input data in our graph. 

We found that while normalizing the input data made the models train faster and improved training accuracy, it seemed to have a negative impact on the validation and test datasets. The optimal combination was scaling the data to between 0 and 1 while not centering it.  

We do not understand how or why centering the input data caused this behavior, but we suspect it may have effected the training data differently than the test and validation data, possibly due to how the graph was constructed in TensorFlow. Another possibility is that centering and scaling the input data caused the models to learn faster and thus learn to overfit the training data faster.  

### Effect of Contrast Enhancement
We found that increasing the contrast of the scans could make subtle features more prominent and allowed the contrast to be adjusted via a command line parameter to the scripts. We evaluated several combinations of settings for adjusting the contrast and found that the best results were achieved by using a contrast setting of 1.1 for training and 1.5 to 1.8 at test time. We found that increasing the contrast at test time could improve the recall by up to 10%, but at the cost of lowering the precision.   

### Decision Thresholds
These results were obtained using a probability threshold of 0.50. Recall or precision could be drastically increased, at the cost of the other metric, by adjusting the threshold. We used a pr curve to evaluate the effect of altering the threshold, and altering the threshold from 0.05 to 0.95 allowed us to achieve either precision or recall of close to 1.0. 

This could prove very useful to radiologists, allowing them to screen out scans which have high probabilities of being either normal or abnormal and focus on more ambiguous scans. 


## Conclusion
While we have been able to achieve higher than expected accuracy on both classifying to normal/abnormal as well as classifying the type and pathology of abnormalities, training dataset 6 and 8 were constructed in an artificial fashion which may not generalize to raw scans. Dataset 9 was constructed specifically to avoid this problem, and while the results for models trained on this dataset were not as good as for models trained on dataset 8, they were still better than we had expected.  

The relative volatility of the validation accuracy and recall also are a cause for concern as to whether the models are learning features that will generalize to other datasets, if such datasets were available. However, models trained on Dataset 9 performed relatively well on the MIAS data, which is a very good indication that the models are learning useful features and can generalize.  
 
Our other big concern was that the models seemed unstable, in that training the same model on the same dataset would not always produce the same results. We found the increasing the cross entropy weighting tended to stabilize this to a degree, as did using online data augmentation.

Despite these problems with our results, we feel that, as a proof of concept, we have demonstrated the Convolutional Neural Networks can be trained to recognize abnormalities in mammograms, with recall over 90%, substantially above human performance. 

The life and death nature of breast cancer makes putting a system like this into practice difficult. However the fact that adjusting the thresholds makes it possible to achieve either very high precision or very high recall offers the possibility of using such systems to aid radiologists rather than replacing them. Outputting probabilities rather than predictions would provide radiologists with additional information to aid them in reading scans. 

To this end, future work would include creating a system which takes an entire scan as input and analyses it for abnormalities. Algorithms such as a sliding window, YOLO or attention models could also indicate what areas of the scans should be looked at closely. Unfortunately, the lack of available training data seems to be the bottleneck for developing these ideas further.

## Supplementary Files

### GitHub Repositories
Two personal GitHub repositories were used for this work:

1. https://github.com/escuccim/mias-mammography - contained the Jupyter notebooks and code run locally.
2. https://github.com/escuccim/mammography-models - was used to sync code between my laptop and the Google Cloud instances where the models were trained 

If there is something missing from this repository, it may be in located in one of those. Below in a non-exhaustive list of scripts and notebooks which contain relevant code or were used for this work.

### Notebooks
- MIAS Exploratory Data Analysis.ipynb - exploratory data analysis of the MIAS data
- Wisconsin (UCI) EDA.ipynb - exploratory data analysis of the UCI data
- SVM.ipynb, kNN.ipynb, Decision Trees.ipynb, Multilayer neural networks.ipynb - application of standard machine learning techniques to the UCI data
- UCI Results.ipynb - results of above evaluations
- crop_cbis_images_x.ipynb - various methods used to extract ROIs from the CBIS-DDSM dataset
- crop_normal_images_x.ipynb - corresponding code to create images from normal DDSM images
- crop_mias_images_x.ipynb - corresponding code the create images from the MIAS dataset
- review_images_x.ipynb - to review the images created in the above scripts and identify any problems
- write_to_tfrecords_x.ipynb - create tfrecords files from dataset x
- convnet_1.0.0.29l.ipynb - code used to train model 1.0.0.29 in a notebook, with results and evaluation on test data
- convnet_training_metrics.ipynb - training and validation metrics for selected training runs
- ddsm_results.ipynb - accuracy and recall metrics for selected training runs

### Python Scripts
- mammo_utils.py - functions used in pre-processing data
- training_utils.py - functions used for creating and training models
- inception_utils.py - functions used to create and train our Inception clone
- candidate_x.x.x.x - various candidate models trained
- vgg_16.x.py - code used to create and train our VGG-lite model
- inception_v4.x.py - code used to create and train our Inception-lite model

### Markdown Files
- ReadMe.md - what you are reading right now
- overview_of_image_processing.md - an overview of the steps taken to create and pre-process the image data

### Other
- data/results - metrics generated during training saved as .npy files
- the tfrecords used to train the models are available from download from Amazon S3. The URLs are in training_utils.py.
- Decompressing-For-LJPEG-image - code used to convert the DDSM images from LJPEG to usable images
- model_notes.xlsx - notes kept during training of models, only includes results relevant to this overview

### Training Logs
The TensorBoard training logs are also provided for selected models in the /logs/ directory. The logs include scalar metrics taken every 50 steps, image summaries of the kernels taken every epoch, and pr curves used to evaluate the effect of the decision threshold.

### Models
The following pre-trained models are available for download. Each zip file contains the checkpoint for the model:

- model_s1.0.0.29l.8.2 - trained on dataset 8 for binary classification - https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/model_s1.0.0.29l.8.2.zip
- model_s1.0.0.35b.9 - trained on dataset 9 for binary classification - https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/model_s1.0.0.35b.9.zip


## References
[1]	D. Levy, A. Jain, Breast Mass Classification from Mammograms using Deep Convolutional Neural Networks, arXiv:1612.00542v1, 2016

[2]	N. Dhungel, G. Carneiro, and A. P. Bradley. Automated mass detection in mammograms using cascaded deep learning and random forests. In Digital Image Computing: Techniques and Applications (DICTA), 2015 International Conference on, pages 1–8. IEEE, 2015.

[3]	N.Dhungel, G.Carneiro, and A.P.Bradley. Deep learning and structured prediction for the segmentation of mass in mammograms. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 605–612. Springer International Publishing, 2015.

[4]	J.Arevalo, F.A.González, R.Ramos-Pollán,J.L.Oliveira,andM.A.G.Lopez. Representation learning for mammography mass lesion classiﬁcation with convolutional neural networks. Computer methods and programs in biomedicine, 127:248–257, 2016.

[5]	Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

[6]	The Digital Database for Screening Mammography, Michael Heath, Kevin Bowyer, Daniel Kopans, Richard Moore and W. Philip Kegelmeyer, in Proceedings of the Fifth International Workshop on Digital Mammography, M.J. Yaffe, ed., 212-218, Medical Physics Publishing, 2001. ISBN 1-930524-00-5.

[7]	Current status of the Digital Database for Screening Mammography, Michael Heath, Kevin Bowyer, Daniel Kopans, W. Philip Kegelmeyer, Richard Moore, Kyong Chang, and S. Munish Kumaran, in Digital Mammography, 457-460, Kluwer Academic Publishers, 1998; Proceedings of the Fourth International Workshop on Digital Mammography.

[8]	Rebecca Sawyer Lee, Francisco Gimenez, Assaf Hoogi , Daniel Rubin  (2016). Curated Breast Imaging Subset of DDSM. The Cancer Imaging Archive.

[9]	Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057.

[10]	O. L. Mangasarian and W. H. Wolberg: "Cancer diagnosis via linear programming", SIAM News, Volume 23, Number 5, September 1990, pp 1 & 18.

[11]	William H. Wolberg and O.L. Mangasarian: "Multisurface method of pattern separation for medical diagnosis applied to breast cytology", Proceedings of the National Academy of Sciences, U.S.A., Volume 87, December 1990, pp 9193-9196.

[12]	O. L. Mangasarian, R. Setiono, and W.H. Wolberg: "Pattern recognition via linear programming: Theory and application to medical diagnosis", in: "Large-scale numerical optimization", Thomas F. Coleman and YuyingLi, editors, SIAM Publications, Philadelphia 1990, pp 22-30.

[13]	K. P. Bennett & O. L. Mangasarian: "Robust linear programming discrimination of two linearly inseparable sets", Optimization Methods and Software 1, 1992, 23-34 (Gordon & Breach Science Publishers).

[14]	K. Simonyan, A. Zisserman, Very Deep Convolutional Networks for Large-Scale Image Recognition, arXiv:1409.1556, 2014

[15]	S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In Proceedings of The 32nd International Conference on Machine Learning, pages 448–456, 2015

[16]	C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1–9, 2015.

[17]	C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna. Rethinking the inception architecture for computer vision. arXiv preprint arXiv:1512.00567, 2015.

[18]	C. Szegedy, S. Ioffe, V. Vanhoucke, Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning, arXiv:1602.07261v2, 2016

[19]	K. He, X. Zhang, S. Ren, J. Sun, Deep Residual Learning for Image Recognition, arXiv:1512.03385, 2015

[20]  J. Redmon, S. Divvala, R. Girshick, A. Farhadi, You Only Look One: Unified, Real-Time Object Detection, arXiv:1506.02640, 2015

[21] R. Girshick, J. Donahue, T. Darrell, J. Malik, Rich feature hierarchies for accurate object detection and semantic segmentation, arXiv:1311.2524, 2013

[22] S. Arkhangelskiy, Data Augmentation on GPU in TensorFlow, https://becominghuman.ai/data-augmentation-on-gpu-in-tensorflow-13d14ecf2b19 
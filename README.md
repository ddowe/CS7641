---
layout: base
title: Final Report
---

# **Final Report**

## **Identifying Contrails to Reduce Global Warming**

![contrails_with_labels](./images/contrails_with_labels_2.PNG)

## **Table of Contents**

1. [Introduction & Background](#1-introduction--background)
2. [Problem Definition](#2-problem-definition)
3. [Data Collection](#3-data-collection) 
4. [Methods](#4-methods) 
    - [Data Pre-Processing](#41-data-pre-processing-color-scheme-normalization-band-selection)
    - [Dimensionality Reduction and Dataset Creation](#42-dimensionality-reduction-and-dataset-creation)
    - [Unsupervised Learning with GMM](#43-unsupervised-learning-with-gmm)
    - [Supervised Learning - Ensemble Classifier](#44-supervised-learning---ensemble-classifier)
    - [Supervised Learning - Binary Classification Neural Network](#45-supervised-learning---binary-classification-neural-network)
    - [Hyperparameter Tuning - Binary Classification Neural Network](#46-hyperparameter-tuning---binary-classification-neural-network)
    - [Supervised Learning - U-Net CNN](#47-supervised-learning---u-net-cnn)
5. [Results and Discussion](#5-results-and-discussion)
    - [Data Pre-Processing Results](#51-data-pre-processing-results)
    - [Dimensionality Reduction and Dataset Creation Results](#52-dimensionality-reduction-and-dataset-creation-results)
    - [Unsupervised Learning with GMM Results](#53-unsupervised-learning-with-gmm-results)
    - [Supervised Learning - Ensemble Classifier Results](#54-supervised-learning---ensemble-classifier-results)
    - [Supervised Learning - Binary Classification Neural Network Results](#55-supervised-learning---binary-classification-results)
    - [Hyperparameter Tuning - Binary Classification Neural Network Results](#56-supervised-learning---hyperparameter-tuned-binary-classification-results)
    - [Supervised Learning - U-Net CNN Results](#57-supervised-learning---u-net-cnn-results)
    - [Metrics Used to Evaluate Model Performance](#58-metrics-used-to-evaluate-model-performance)
    - [Visualization of Dataset and Results](#59-visualization-of-dataset-and-results)
6. [Comparative Analysis](#6-comparative-analysis-of-contrail-detection-models)
7. [Conclusion](#7-conclusion)
8. [Contribution Table](#8-contribution-table)
9. [References](#9-references)


## **1. Introduction & Background**

Contrails, resulting from the condensation and crystallization of water vapor emitted by aircraft engines in the upper atmosphere, have been identified as a significant driver of global climate change. Extensive research has been conducted to develop contrail detection algorithms using satellite imagery, specifically the Advanced Baseline Imager (ABI) on the GOES-16 satellite. Geraedts et al. introduced a CNN algorithm trained on manually labeled GOES-16 ABI images, achieving state-of-the-art performance in linear contrail detection [1]. Another study incorporated crucial temporal (time-based) context in the classification process, improving contrail detection accuracy and emphasizing the importance of monitoring contrail evolution over time [2].

To help improve the accuracy of existing contrail detection models, Google Research has deployed a [code competition on Kaggle](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/overview) to incentivize the ML community to develop more enhanced solutions. This competition is the inspiration of our project topic.

## **2. Problem Definition**

To address the problem, our study aims to employ machine learning techniques to validate and improve contrail detection models that rely on GOES-16 ABI images. The primary objective is to develop dependable tools for the aviation industry that effectively detect/predict contrail formation and mitigate their impact on climate change.

## **3. Data Collection**
[Our dataset](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/data), provided by the Kaggle competition organizers, consists of ~450GB of GOES-16 ABI images. All images in the dataset (both the data and labels) have a size of $$256\times 256$$ pixels. There are $$N=22,385$$ contrail datapoints in total. Each individual contrail datapoint consists of 9 different views of the contrail, corresponding to different "bands" (infrared wavelength ranges) of the same contrail image. For each of the 9 bands, there are 8 temporal images of the contrail taken at 10-minute intervals, where the 5th image is considered the primary time of interest. Thus, for each contrail datapoint, we have a total of $$(9\ bands)\times(8\ time\ steps)=72$$ images to potentially use for training, incorporating temporal and visual variation into the project challenge. Ground truth consists of a single binary image whose pixels are white when they contain a contrail and black when no contrail exists. To minimize contrail misclassifications, key labeling criteria must be followed, considering factors such as the size of the contrail, the length-to-width ratio, evolution over time, and visibility across multiple images.

## **4. Methods**

### **4.1 Data Pre-Processing: Color Scheme, Normalization, Band Selection**
First, let us analyze two examples from our training dataset that contain contrails. The figure below demonstrates band 11 (the fourth band of bands 08-16) and the 5th image of the temporal images stored in each band.

![ExampleDataset](./images/ExampleDataset.png)

Upon observation, we can see that the contrails shown in the human pixel masks images are not clearly visible by just looking at the original images from band 11. Thus, as part of our Data Preprocessing, we implemented the following:

- Normalized our dataset.
- Combined a few bands together to represent 1 band for each contrail
- Modified the color scheme of our entire dataset.

We took the band images of ash plumes, represented by bands 11, 14, and 15, and converted them into color images. We extracted only the 5th frame from each band image to obtain a mask frame and computed differences between the bands, normalized the differences, and stacked them into a new color scheme array. The resulting color values were clipped to the range of [0, 1]. In order to determine which bands and the bounds we used for the ash color scheme conversion, we followed the Ash RGB recipe found on page 7 in reference [3]. This color scheme was originally developed for viewing volcanic ash in the atmosphere but is also useful for viewing thin cirrus, including contrails. In this color scheme, contrails appear in the image as dark blue.


![AshRGB](./images/AshRGB.png) [3]

This Helps with:

- Emphasizing ice clouds as darker colors
- Converts it to ash color scheme which removes unnecessary information in the image
- Creating a streamlined dataset with RGB images for a simpler appearance

In summary, we start with 9 bands and 8 temporal images for each datapoint but we combine and manipulate 3 of the band images (out of 9) and use only the 5th temporal image to generate color representations of ash plumes. We stack them into a new color scheme array and store our new normalized dataset with the changed color scheme as $$(256\times 256\times 3)$$.

### **4.2 Dimensionality Reduction and Dataset Creation**
With the colorized RGB images showing the contrails more clearly, we used PCA to reduce the dimensions to a much more manageable size. We flattened each $$(256\times 256\times 3)$$ image into a $$(1\times 196608)$$ array and performed PCA using the Scikit-learn Python library. The number of principal components after PCA was $$D=2000$$.

Following the dimensionality reduction, we split the data into training/validation/test sets using an 80%/10%/10% split, respectively. The final size of the respective contrail datasets were:
* Training:  $$(17908 \times 2000)$$
* Validation: $$(2238 \times 2000)$$
* Test: $$(2239 \times 2000)$$

### **4.3 Unsupervised Learning with GMM**
Contrails are identified visually by their distinctive shape and color in an image. Thus, as part of our unsupervised learning we performed Gaussian Mixture Model (GMM) clustering on the contrail images to cluster pixels by color and uncover the contrails. We take the RGB colorized images of the data and fit a GMM model to them with $$K=10$$ and $$max\ iters=10$$.

### **4.4 Supervised Learning - Ensemble Classifier**
For our first step in supervised learning, we decided to create an ensemble classifier because of its simplicity, its potential to improve classification accuracy, and because it leverages the collective strengths of individual classifiers. We trained five traditional machine learning models to classify whether a given image contained a contrail or not. The majority vote of the combined models (the "ensemble") is chosen as the prediction for a given test image. To create this type of binary classification, we changed the labels for the training data (the image pixel masks) to binary labels, such that '1' means the associated image contains a contrail and '0' means the image has no contrail. The five models we chose and their associated hyperparameters are:

1. Random Forest
   - Max Depth = 40
   - Number of trees = 80
   - Criterion = Entropy
2. K-Nearest Neighbors
   - K = 2
3. Logistic Regression
   - Penalty = L2
   - C = 0.00001
4. Multi-Layer Perceptron
   - 2 Layers
   - 25 neurons per layer
   - ReLu activation used
5. Gaussian Naïve Bayes

To determine hyperparameters during training, we used grid search to select hyperparameters and cross validation to evaluate them. For example, during training of the KNN model, we performed a sweep of $$K$$ values and selected $$K=2$$ because it had the highest accuracy on the validation set, as shown below.

![KNN Plot](./images/KNN_Plot.png)

### **4.5 Supervised Learning - Binary Classification Neural Network**

For our next step in supervised learning, we leveraged a simple neural network binary classification scheme using the PyTorch framework to create a deep learning model. The training data consists of 17908 samples of 256x256 pixel images that were reduced to 2000 features using principal component analysis (PCA). Labels for the training data were changed to binary labels with '1' meaning that the associated image contained a contrail and '0' was a data point with no contrail. Similarly, the validation set contained 2238 samples. A standard scalar function from the sklearn library was applied to the training and validation data to normalize the values before sending it to the neural network. This function subtracts the mean from the column and divides each column value by the standard deviation of the column [4].  

This simple neural network contains two linear transformation layers. A sigmoid function was applied to the output of the model that returns a value between 0 and 1 for binary classification. For the hyperparameters, we used a 0.01 learning rate for 15 epochs. Consideration for tuning the different hyperparameters along with the number of layers and neurons may achieve better performance for future work. The neural network can be visualized in the figure below:
![Neural Network](./images/NN_visualized.png)

Our loss function utilizes the PyTorch class BCELoss, which is a standard Binary Cross Entropy (BCE) loss function commonly used for binary classification. BCE, also known as logarithmic loss or log loss, is a loss metric used for tracking incorrect label classification for data. In other words, this loss function penalizes the neural network when the model deviates from classifying correct labels according to probability. The calculation for this loss function is as follows:

$$-\frac{1}{N} \sum_{i=1}^{N} y_i log(P(y_i)) + (1-y_i)  log(1-P(y_i))$$
[5]

Gradients derived from the loss function are leveraged to adjust the weights in each iteration of the model training. This is for minimizing loss and increasing the accuracy of the model [4]. 

### **4.6 Hyperparameter Tuning - Binary Classification Neural Network**

To improve the performance of our deep learning model, we performed several updates and adjustments to the hyperparameters and architecture. Here's a summary of the changes we made:

1. Adjusted Learning Rate: We changed the learning rate to 0.00005. A smaller learning rate can help the model converge more gradually, potentially leading to better generalization and avoiding overshooting the optimal solution.
    
2. Optimizer Change: We switched the optimizer to Adam. Adam is a popular optimization algorithm that adapts the learning rate based on the past gradients, providing faster convergence and better handling of sparse gradients.
    
3. Increased Depth and Width of Layers: We increased the depth and width of the neural network layers. By adding more layers and/or increasing the number of neurons in each layer, the model can learn more complex patterns and representations from the data.
    
4. Added Dropout Layers: Dropout layers were introduced into the model. Dropout is a regularization technique that randomly drops out a fraction of neurons during training, preventing the model from relying too much on any particular set of neurons and reducing overfitting.
    
The hyperparameter tuning approach allowed us to fine-tune the model's performance and generalize better on unseen data. The combination of adjusted learning rate, Adam optimizer, increased model capacity, and dropout layers led to a more robust and reliable binary classification model, resulting in slightly improved F1 scores, accuracy, and reduced overfitting for the training set. 

### **4.7 Supervised Learning - U-Net CNN**

While the previous models - which perform binary classification (i.e., detecting which images contain contrails) - are useful, it would be even more desirable to be able to detect *where* in the image the contrails are. Often it's difficult to distinguish between clouds and contrails, and automatic detection of the contrails within the image data would be advantageous. Thus, to improve upon our previous methods, we employed a U-Net deep learning architecture in PyTorch for contrail segmentation. U-Net is a deep learning architecture that has proven to be highly effective for image segmentation tasks. The name "U-Net" comes from the U-shaped architecture of the network, characterized by an encoder-decoder structure with skip connections. The U-Net architecture is particularly useful for image segmentation tasks because of its ability to accurately localize and delineate objects within images. It excels in scenarios where precise pixel-level segmentation is required, making it well-suited for detecting and segmenting complex patterns, such as contrails.

![U-Net](./images/UNET.png)
*Example U-Net Architecture*

Input images are 3-channels and 256x256 pixels. For the training set, data augmentation is applied, including horizontal flipping and random resized cropping with a scale factor between 0.75 and 1.0. 

We implemented two models trained using the following hyperparameters:

U-Net with EfficientNet-B3 encoder:
- Number of epochs: 30
- Batch size: 32
- Learning rate: 1e-3
- The Adam optimizer is used for training the model. A cosine learning rate scheduler with warm-up is employed to adjust the learning rate during training.

For training, we used the EfficientNet-B3 encoder with pretrained ImageNet weights. EfficientNet-B3 is a highly efficient and powerful convolutional neural network that has shown superior performance on various image recognition tasks.

U-Net with ResNeSt26d encoder:
- Number of epochs: 20
- Batch size: 16
- Learning rate: 8e-4
- The RMSProp optimizer is used for training the model

The ResNeSt26d encoder is based on the ResNeSt26d model (takes an input image of size 256x256 and processes it through a series of convolutional layers, batch normalization, and ReLU activation functions)

#### Neural Network Architecture:

| Layer (type)        | Output Shape      | Param #    |
|---------------------|-------------------|------------|
| Conv2d-1            | [-1, 32, 128, 128]| 864        |
| BatchNorm2d-2       | [-1, 32, 128, 128]| 64         |
| ReLU-3              | [-1, 32, 128, 128]| 0          |
| Conv2d-4            | [-1, 32, 128, 128]| 9,216      |
| BatchNorm2d-5       | [-1, 32, 128, 128]| 64         |
| ReLU-6              | [-1, 32, 128, 128]| 0          |
| Conv2d-7            | [-1, 64, 128, 128]| 18,432     |
| BatchNorm2d-8       | [-1, 64, 128, 128]| 128        |
| ReLU-9              | [-1, 64, 128, 128]| 0          |
| MaxPool2d-10        | [-1, 64, 64, 64]  | 0          |
| ...                 | ...               | ...        |
| RadixSoftmax-177    | [-1, 1024]        | 0          |
| SplitAttn-178       | [-1, 512, 16, 16] | 0          |
| Identity-179        | [-1, 512, 16, 16] | 0          |
| Identity-180        | [-1, 512, 16, 16] | 0          |
| Identity-181        | [-1, 512, 16, 16] | 0          |
| AvgPool2d-182       | [-1, 512, 8, 8]   | 0          |
| Conv2d-183          | [-1, 2048, 8, 8]  | 1,048,576  |
| BatchNorm2d-184     | [-1, 2048, 8, 8]  | 4,096      |
| ReLU-185            | [-1, 2048, 8, 8]  | 0          |
| ...                 | ...               | ...        |
| Activation-245      | [-1, 1, 256, 256] | 0          |

**Total params: 24M**

- The neural network consists of convolutional layers, batch normalization, ReLU activation functions, max-pooling layers, average pooling layers, and attention mechanisms.
- The network is designed for image segmentation and outputs a mask of size [1, 256, 256].


During training, the model is evaluated using the Dice coefficient, which measures the similarity between the predicted contrail masks and the ground truth masks. The Dice coefficient ranges from 0 to 1, where 1 indicates perfect agreement between the predicted and ground truth masks. The Dice coefficient is preferred for image segmentation because it comprehensively evaluates the model's performance by considering both precision and recall. It quantifies the overlap between predicted and ground truth masks, rewarding accurate segmentations and penalizing false positives. The Dice coefficient was calculated as follows:

$$\huge \frac{2*|X∩Y|}{|X|+|Y|}$$


![DICE](./images/DICE.png)

## **5. Results and Discussion**

### **5.1 Data Pre-Processing Results**
To better visualize our dataset, let's examine an example that displays all the bands for only the 5th temporal image. The figure below showcases the 9 bands (from 08-16), presenting different views of the contrail.

![ContailsBands](./images/ContailsBands.png)

After combining the band images to generate color representations of ash plumes, computing differences, normalizing them, and stacking them into a new color scheme array, we can see the figure below how the normalized new color scheme image shows the contrails much more clearly than our original dataset without any data preprocessing.

![OriginalvsChangedvsMask](./images/OriginalvsChangedvsMask.png)

### **5.2 Dimensionality Reduction and Dataset Creation Results**
By reducing the dimensions of each image from 196608 to a mere 2000, we were able to retain 96.75% of the variance while reducing the size of the dataset by a factor of 98.3x. An example of a reconstructed contrail image from the reduced components is shown below:

![ReconstructedImage](./images/ReconstructedImage.png)

After splitting the data into training/validation/test sets using an 80%/10%/10% split, and before moving on to train our model, we tried to take a look at each of our dataset making sure there are enough contrails vs no contrails datapoints in each set, for example, below is a bar chart we created to compare our training/validation/test sets with vs without contrails.

![ContrailsVSnoContrails](./images/ContrailsVSnoContrails.png)

### **5.3 Unsupervised Learning with GMM Results**
We found that GMM clustering produced, on average, a very good representation of shapes within the dataset images and could often identify contrails if they are large enough. However, small or fragmented contrails might be missed by the GMM model due to our value of $$K=10$$. After evaluating a subset of the GMM-clustered data, we found that >50% of the contrail regions were accurately clustered by the model. Some example clustering results are shown below:

![GMM_IMAGES](./images/GMM_CLUSTERING_GRAPHS.png)

### **5.4 Supervised Learning - Ensemble Classifier Results**
For our simple ensemble classifier, we tracked the test set accuracy for each individual model and compared it to the ensemble accuracy. KNN and MLP classifiers perform surprisingly well, with test set accuracies >66%. However, the Gaussian Naive Bayes classifier performs rather poorly in comparison, presumably because the strong assumptions (conditional independence and underlying Gaussian data distribution) of Gaussian Naive Bayes are inaccurate for our chosen application. Nevertheless, results show that the ensemble model slightly outperforms the individual classifiers, with a test set accuracy of 70.1%. 

![Ensemble_Accuracy](./images/ensemble_acc.png)

### **5.5 Supervised Learning - Binary Classification Results**

For our binary classification deep learning model, we used our training dataset to track our loss and accuracy for binary classification. To prevent overfitting, we used our validation dataset to track the loss and accuracy against the training dataset. 

![Loss 15 Epochs](./images/loss_15.png)

![Accuracy 15 Epochs](./images/accuracy_15.png)

By using an early stopping method, we conclude that the optimal stopping point for this neural network is at 7 epochs. This is the point where the training and validation loss functions start to diverge from each other, and it is a good indication that the model is starting to overfit [5]. Consequently, at this epoch our model achieves an accuracy of around 62% for the validation set.

![F1 Score 15 Epochs](./images/f1_15.png)

A high precision score tells us that the model is likely to have correct positive classification  (images positively classified as contrails are correct), and high recall tells us that the model is good at capturing all the positive classifications in our model (may contain true negative classifications) [8]. Although precision and recall are important metrics, F1 score gives a more cohesive metric score by combining these into a single metric. The F1 score is a harmonic mean between precision and recall, with a high F1 score alluding to a high precision and recall, a low score meaning that precision and recall are low, and medium score meaning that either precision or recall is low and the other is high [8]. Our model across 15 epochs show an above medium F1 score peaking around the 9th epoch with a score of 0.593.

We also calculated the Receiver Operating Characteristic (ROC) curve using the BinaryROC function from the torchmetrics Python library. The ROC is a performance metric for binary classification models that plots the true positive rate (TPR) vs. the false positive rate (FPR) [9].

$$ TPR = \frac{TP}{TP+FN} $$
$$ FPR = \frac{FP}{FP+TN} $$

The area under the curve (AUC) of an ROC metric tells us the performance of our classification thresholds. Consequently, a higher curve away from the 0 diagonal can be interpreted as a probability that our neural network model is better at classifying images with contrails and images without contrails in our dataset. 

![Binary ROC](./images/binaryroc_15.png)

### **5.6 Supervised Learning - Hyperparameter Tuned Binary Classification Results**

Through hyperparameter tuning, the model was trained for more epochs (iterations over the training data), which might have allowed the model to learn more complex patterns and representations. As a result, the validation accuracy improved slightly compared to previous attempts.

The validation accuracy, measured at epoch 70, reached a value of 64%. This means that, when evaluating the model's performance on the validation dataset, approximately 64% of the samples were correctly classified.

Loss             |  Accuracy
:-------------------------:|:-------------------------:
![Loss 2](./images/LOSS_v2.png)  |  ![Acc 2](./images/ACC_v2.png)

ROC             |  F1
:-------------------------:|:-------------------------:
![Loss 2](./images/ROC_v2.png)  |  ![Acc 2](./images/F1_v2.png)

### **5.7 Supervised Learning - U-Net CNN Results**
#### U-Net CNN with EfficientNet-B3 encoder

The U-Net model was trained for 30 epochs, and the training progress was monitored based on the training loss, validation loss, validation Dice coefficient, and learning rate. The model demonstrated a steady improvement in performance over the epochs. Initially, during the first epoch, the training loss was 0.034, and the validation loss was 0.0357. The validation Dice coefficient was 0.4774, indicating a moderate level of accuracy. As the training progressed, the model's performance improved, achieving a training loss of 0.0229 and a validation loss of 0.0263 during the final epoch. The validation Dice coefficient reached a value of 0.6142, indicating a high level of accuracy in contrail detection. Overall, the U-Net model demonstrated promising results in accurately identifying contrails in images, showcasing the effectiveness of the chosen architecture and training process.

Losses             |  Dice Coefficient
:-------------------------:|:-------------------------:
![Loss 3](./images/DICELOSS.png)  |  ![Dice](./images/DICECOFF.png)

#### U-Net CNN with ResNeSt26d encoder
For the Resnet encoder the dice coeffecient in the traning data went upto 0.668 for 20 epochs and went upto 0.5683 for the validation data.

![DiceResnet](./images/DiceResnet.png)

### **5.8 Metrics Used to Evaluate Model Performance**
- Accuracy: Fraction of predictions that the model evaluates correctly to the total number of predictions. $$\frac{\text{Correct Predictions}}{\text{Total Predictions}}$$
- Precision: Ratio of the number of true positive predictions to the total number of positive predictions. It signifies the number of correctly labelled positive predictions. 
- Recall: Ratio of the number of true positive predictions to the sum of true positive and false negative predictions. It signifies the extent of positive predictions that were incorrectly labelled as negative.
- F1-score: It is the harmonic mean of the precision and recall. Since it combines both metrics into a single metric, it is an ideal choice to evaluate the performance of our models. The formula is given by: $$\frac{2}{precision^{-1} +  recall^{-1}}$$

### **5.9 Visualization of Dataset and Results**

![Breaking down an image with contrails](./images/data_vis.PNG)

The figure below shows 4 examples from the validation dataset of the predicted masks from our U-Net model with Resnet Encoder compared with the Ground Truth Contrail Mask.

![Res0](./images/Res0.png)

![Res1](./images/Res1.png)

![Res2](./images/Res2.png)

![Res3](./images/Res3.png)

## **6. Comparative Analysis of Contrail Detection Models**

As the demand for efficient and accurate contrail detection in satellite images continues to grow, this project report presents a comprehensive comparative analysis of four distinct models: Unsupervised Gaussian Mixture Model Clustering, Ensemble Classifier, Binary Classification Deep Learning Neural Network, and U-Net Convolutional Neural Network. The primary objective of this study is to identify the most effective approach for achieving robust and precise contrail detection across various environmental conditions. Through meticulous evaluation of each model's specific results, strengths, and weaknesses, we aim to provide valuable insights that will contribute to significant advancements in contrail detection techniques.

| Model                                   | Specifics of Results                                                                                 | Strengths                                                                                            | Weaknesses                                                                                            |
|-----------------------------------------|------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| Unsupervised GMM Clustering             | - Produced a good representation of shapes in dataset images.                                      | - Unsupervised learning, provides insights into natural groupings.                                  | - Sensitivity to hyperparameters, Gaussian assumption not ideal for complex shapes.                 |
|                                         | - Successfully identified prominent and well-defined contrails.                                    | - Able to capture overall shapes and structures in the images.                                     | - Struggles with detecting small or fragmented contrails.                                           |
|                                         | - Suitable for identifying well-defined and prominent contrails.                                   |                                                                                                      | - Requires careful parameter selection to achieve optimal results.                                   |
| Ensemble Classifier             | - Produced a high classification accuracy (~70.1%) for the detection of contrails in images. | - Simplicity and flexibility, utilizing well-known traditional classification models which have stood the test of time.   |   - Low interpretability due to the many different models used in the ensemble.      |
|                                 | - Used 5 models: Random Forest, KNN, GNB, LogReg, MLP | - Robust to noise and overfitting.   |   - Poorly-suited models used in the ensemble can drag down the overall accuracy.      |
| Binary Classification Deep Learning Model | - Achieved an accuracy of ~62% for the validation set.                                             | - Learns complex features, efficient for binary classification tasks.                               | - Requires a large labeled dataset, prone to overfitting beyond a certain number of epochs.         |
|                                         | - Peaked F1 score at 0.593, indicating a balanced trade-off between precision and recall.         |                                                                                                      |                                                                                                       |
|                                         | - ROC curve and AUC indicated good performance in classifying images with contrails and without contrails. |                                                                                                      |                                                                                                       |
| U-Net CNN                               | - Validation Dice coefficient reached 0.6142, indicating high accuracy in identifying and segmenting contrails. | - Designed for image segmentation tasks.                                                           | - Requires a large annotated dataset, computationally expensive.                                      |
|                                         | - Consistently improved in performance during training, reducing both training and validation loss.  | - Captures detailed spatial features.                                                               | - Requires image augmentation prior to training to eliminate overfitting.                                                                                                      |
|                                         | - Effectively captured fine details and localized objects, leading to accurate contrail detection.  |                                                                                                      |                                                                                                       |
                                                                                                                                                        


Overall, each model has its strengths and weaknesses in contrail detection. The choice of the best model depends on the specific requirements of the application and the nature of the dataset. For our application, unsupervised methods such as GMM seem to be rather poor at anything other than visualization of the data. Both the ensemble classifier and initial deep learning model are fairly successful at detecting which images contain contrails, but are unable to isolate specific contrail locations within a given image. The U-Net CNN model, on the other hand, showed promising results in accurately identifying and segmenting contrails, making it a strong candidate for further development and real-world implementation.

## **7. Conclusion**

In our neural network, we built multiple classification models to classify contrail images. Although our experiments show significant results, our performance metrics show that there is room for improvement, particularly in our neural network learning models. In order to reduce overfitting, minimize loss, and improve the accuracy of our supervised models, we discussed several techniques to improve performance. Some of the techniques we discussed to reduce overfitting and loss while maintaining or improving our accuracy are tuning hyperparameters, data augmentation, dropout strategies, and regularization. 

1. Hyperparameter Tuning:
	Hyperparameters (e.g. epoch, batch size, neural network layers) play a crucial role in enhancing the performance of a neural network model to obtain optimal results. We discussed using various hyperparameter optimization techniques like grid search or random search in order to improve the learnability of our model, which may lead to improved accuracy, speed, and better overall performance [7].
	
1. Data Augmentation:
	Data augmentation is an excellent method to increase the size and enhance the diversity of our dataset. We plan to use various general image transformations (flipping, rotation, scaling, noise, etc.) to increase the number of sample images. We also discussed segmenting the images to 10x10 pixel sub images. A larger dataset is a way to assist in alleviating overfitting problems.

1. Dropout Strategies:
	Another common strategy for overfitting issues is to use a dropout strategy. Dropout ignores randomly selected neurons in the neural network, which changes the neural network to learn from the data in a different way [7].

1. Regularization Techniques:
	Applying regularization, like the L1 or L2 regularization method, will help reduce the complexity of out model by adding a penalty to our loss function. Consequently, our loss function will penalize the larger coefficients and limit the variance of our model. Regularization can improve the performance of our training and help limit overfitting.

The original models (namely, the ensemble classifier and NN-based binary classifier) could only predict whether an image had contrails or not. Our new models (namely, the U-Net CNNs) can now predict the location of contrails in images to a very high degree of accuracy. Given the large size of the dataset and the need for pixel-based image classification, our U-Net model performs very well for a wide variety of images in the test set, and predicts the location of even very small contrails. It proves that CNN based models are powerful to train large datasets, provided we process the dataset correctly. Moreover, it is evident that for this particular problem, supervised learning is the way to go. Unsupervised learning models cannot determine the presence of small contrails and there is no good way to train the model to separate out contrails from a colour image. With the ground truth masks provided, supervised learning models can be trained to be a very high accuracy. The key takeaway here is that not all types of models can be applied to every problem.  


## **8. Contribution Table**

Alphabetized by Last Name:

|Team Member |Contribution |
| --------------- | --------------- | 
| Michael Cho | Binary Classification Neural Network, U-Net Efficient Encoder Model, Server Infrastructure, Video Presentation | 
| Danielle Dowe | Comparative Analysis |
| Aditya Iyer | Video Presentation |
| Mark Lee | Dimensionality Reduction, GMM Clustering, Ensemble Classifier, Video Presentation |
| Laith Shamieh | Data Pre-Processing, Data Visualization, U-Net ResNet Encoder Model, Video Presentation |


## **9. References**
[1] J. Y. H. Ng, K. McCloskey, J. Cui, V. R. Meijer, E. Brand, A. Sarna, N. Goyal, C. Van Arsdale, and S. Geraedts, "OpenContrails: Benchmarking Contrail Detection on GOES-16 ABI," arXiv:2304.02122 [cs.CV], 2023. 

[2] N. Nipu, C. Floricel, N. Naghashzadeh, R. Paoli, and G. E. Marai, "Visual Analysis and Detection of Contrails in Aircraft Engine Simulations," arXiv:2208.02321 [cs.HC], 2022. 

[3] https://eumetrain.org/sites/default/files/2020-05/RGB_recipes.pdf

[4] A. Prasad, “PyTorch For Deep Learning — Binary Classification ( Logistic Regression ),” Analytics Vidhya, Sep. 13, 2020. https://medium.com/analytics-vidhya/pytorch-for-deep-learning-binary-classification-logistic-regression-382abd97fb43 (accessed Jul. 06, 2023).

[5] “Binary Cross Entropy: Where To Use Log Loss In Model Monitoring,” Arize AI. https://arize.com/blog-course/binary-cross-entropy-log-loss/ (accessed Jul. 06, 2023).

[6] V. R. Meijer, L. Kulik, S. D. Eastham, F. Allroggen, R. L. Speth, S. Karaman, and S. R. H. Barrett, "Contrail coverage over the United States before and during the COVID-19 pandemic," *Environmental Research Letters*, vol. 17, no. 3, p. 034039, March 2022. doi: 10.1088/1748-9326/ac26f0

[7] S. Zivkovic, “#018 PyTorch - Popular techniques to prevent the Overfitting in a Neural Networks,” Master Data Science, Nov. 08, 2021. https://datahacker.rs/018-pytorch-popular-techniques-to-prevent-the-overfitting-in-a-neural-networks/ (accessed Jul. 06, 2023).

[8] J. Korstanje, “The F1 score,” Medium, Aug. 31, 2021. https://towardsdatascience.com/the-f1-score-bec2bbc38aa6 (accessed Jul. 07, 2023).

[9] S. Narkhede, “Understanding AUC - ROC Curve,” Medium, Jun. 15, 2021. https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5 (accessed Jul. 07, 2023).

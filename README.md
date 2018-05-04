# Semi-Supervised Urban Sound Classification
(NYU CUSP) Machine Learning 4 Cities - Final Project

Authors: Ben Steers (bs3639@nyu.edu), Nina Nurrahmawati (nn1221@nyu.edu), Yu Chen (yc3300@nyu.edu)

### Abstract

Automatic urban sound classification is a growing interest in the field of urban informatics as modern processing power allows algorithms to better extract information from audio streams. One of the obstacles preventing further progress on audio classification is the scarcity of real-world labeled data, which is crucial when using supervised classification. This paper looks into the performance of audio classification where only a partially labeled dataset is available. Two feature representations are considered for the audio data: 128-dimensional embeddings from VGGish, a pre-trained convolutional neural network trained on the Youtube AudioSet dataset, and mel-frequency cepstrum coefficients (MFCCs), a representation of the timbral quality of an audio signal. The performance of a Random forest classifier is tested on each input over a varying mix of labeled and unlabeled data. Results indicate that semi-supervised learning improved the test set accuracy over the supervised case using the same amount of labeled data with both the VGGish features and the MFCCs for proportions of the dataset for less than or equal to 50% labeled data.

The dataset can be found [here](https://serv.cusp.nyu.edu/projects/urbansounddataset/urbansound8k.html).

The input features that were used:
* VGGish embeddings. Get the model [here](https://github.com/tensorflow/models/tree/master/research/audioset).
* MFCCs. Librosa documentation can be found [here](https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html).


## Introduction

Following the increasing availability of multimedia data streams and the growing utility of environmental audio analysis, sound event classification is raising more awareness and attracting more attention in the field. It is evident that by generating an efficient audio analysis, sound event classification will be beneficial in serving people and purposes in needs; in surveillance, such as automatically detecting and locating the source of gunshots; in the enforcement of noise ordinances, like excessive construction noise; in enhancing speeches and/or music and in detecting and categorizing urban interests in real time. 

One of the challenges is the disparity between large amount of obtainable audio data and insufficient labeled data. The purpose of this project is to examine the applicability of sound classification using a partially labeled dataset. 

Many past sound classification are recognized on either supervised learning (SL) or unsupervised learning. Their drawbacks were evident as SL has high necessity of a significant amount of labeled data for training activities, and the effort on labeling discoveries from unsupervised data is laborious. This is where semi-supervised learning (SSL) has been proposed as an alternative to improve classification accuracy with limited data.  

Previous papers have done SSL for the purposes of audio classification. Zhang and Schuller (2012) performed semi-supervised classification on a dataset of broad sound classes ranging from People, to Nature, to Office. Low level descriptors of the audio were used to train the classifier using Expectation Maximization (EM), including features like loudness, inharmonicity, energy bands, several spectral moments, etc. A statistically significant improvement was seen in the performance of the model when applying the semi-supervised approach, as opposed to training only on the subset that was labeled. Similarly but targeting on a more specific audio dataset, Diment, Heittola and Virtanen (2013) conducted SSL using Gaussian Mixture Model (GMM) and EM algorithm for musical instrument recognition. In order to use GMMs for classification, the number of components were initialized as the centers for each class and were retrained periodically to prevent the components from diverging. Their evaluation saw significant increase over the supervised case, using only a small proportion of labeled data.

Google have introduced a recently released model built on their open source machine learning framework, Tensorflow. The model is called VGGish, which is a convolutional network trained on millions of audio clips. The model is named after Google’s image processing CNN architecture, VGG, which this algorithm was based upon (Simonyan and Zisserman, 2015). The model treats audio classification as an image processing task, using windows over the Mel spectrograms as input “images”. The model that is released on Github is only half of the network, consisting of only the convolutional layers, while dropping all of the fully connected layers that were used for the training of the network (Ellis, 2017). The output of the model is a 128-dimensional vector representing the learned feature representation of the audio that was used to feed into the fully-connected classification network to learn to classify on the AudioSet dataset, meaning that any information that was relevant in classifying AudioSet has been extracted and is contained in the embedding.

Mel-frequency cepstral coefficients are a common audio feature representation, especially in the field of speech recognition, and give a representation of the timbre of a sound. They are calculated by first converting the audio into a Mel-frequency spectrogram using a short-term Fourier transform (STFT). A Fourier transform is then taken along the frequency axis of the spectrogram which results in the values for MFCCs (Tiwari, 2010). They capture the periodic harmonic relationships of an audio signal. They were popular before the advent of deep learning and using learned feature representations instead of expert-defined feature representations, like what MFCCs are. They are still frequently used as a baseline to compare more modern algorithms against (Salamon, Bello, Farnsworth, & Kelling, 2017).

In this paper, two feature extraction methods are applied to the UrbanSound8k audio dataset - 128-dimensional VGGish embeddings and Mel-frequency cepstral coefficients (MFCCs). The processed dataset (with audio embeddings) will be trained using a Random Forest classifier using EM. Results will be compared based on performances from two feature extractors. We then investigate the likelihood of unlabeled audio data being accurately concluded from partially labeled audio inputs. 


## Data

Audio data for this project is collected from UrbanSound8k, released by NYU CUSP. The data was sourced from field recordings uploaded to www.freesound.org. The dataset contains 8,732 labeled sound excerpts of ten different urban sound classes: Air conditioner; Car horn; Children playing; Dog bark; Drilling; Engine idling; Gunshots; Jackhammer; Siren; and Street music (Salamon, Jacoby, & Bello, 2014). 

The audio files are stored in 24 bit format (representing the resolution used for the discretization of the audio amplitude). The files were pre-processed by converting them from 24 bit PCM (pulse code modulation) into 16 bit PCM because the wave file reader from Scipy package only able to read 16 or 32 bit PCM data. 

For deriving the VGGish embeddings, the default model parameters were used, as defined in the Github repository (Ellis, 2017). A STFT window length of 0.025 seconds was used, with a hop size of 0.01 seconds. The CNN windowing length was 0.96 seconds long, with an equal hop size. Audio clips shorter than the VGGish window size (0.96 seconds) were dropped. The reduced dataset size used for this paper is 8,289 clips. The rest of the model parameters can be seen in the Github repository. Once calculated, Principal Component Analysis (PCA) is used to transform the embeddings to decorrelate features.

The MFCCs were calculated using Librosa, a python library for audio analysis. The spectrogram was calculated with a window length of approximately 0.1 second and a hop size of approximately 0.25 seconds. The number of coefficients to return was set to 20.

Because the audio files vary in length, so do the VGGish embeddings and the MFCC vectors. In order to convert the feature representations into a fixed length vector, the features need to be aggregated along the time dimension. Initially, the modeling was attempted using the mean, along with a combination of the first three moments concatenated into one long vector, however the model performance improved when changing to using only the mean. Before feeding the features to the model, they were normalized between -1 and 1 using the statistical properties of the training set.


## Methods

This project aims to measure the performance level of SSL applied to the audio data. We used the pre-trained VGGish model from CNN Architectures for Large-Scale Audio Classification to get the embedding from our data and Mel-frequency cepstral coefficients (MFCCs) as two separate inputs for the training data. 

The modeling is performed at a varying proportion of labeled data and unlabeled data. The percentage of labeled data is incremented from 1% to 100% (with greater resolution between 1% and 10%) of the entire dataset to see how the model accuracy increases on the testing set. For each proportion of labeled data, the model was trained 30 times to get a distribution over accuracies.

On each iteration, the data was randomly split into stratified training and testing sets using a test size of 0.33. Then the input data was scaled between -1 and 1 using the training set distribution. According to the proportion of labeled data, a random selection of labels were removed from the training data. A model was then fit once with both the labeled and the unlabeled data and once with only the labeled data for both VGGish and MFCC features to show a comparison between the semi-supervised method and the equivalent supervised method (same amount of labeled data). 

Expectation maximization was used to train the semi-supervised model. First, the model is trained on only the labeled data. Then the model is used to predict the probability that the unlabeled data is in each class. This is referred to as the Expectation step. Those probabilities are then used as weights of each class for retraining the model. This is called the Maximization step. The Expectation and Maximization steps are then repeated until the stopping criteria is met. The EM algorithm was written using the sklearn machine learning model interface, and is therefore compatible with any of the sklearn models that support probability prediction and weighted model fitting. For this experiment, a random forest classifier was used. The models were each trained 30 times at each proportion of labeled data: 0.01, 0.03, 0.05, 0.08, 0.1, 0.3, 0.5, 0.7, and 1, where 1 represents an entirely labeled dataset. 


## Results

Classification was performed on 4 different scenarios: supervised and semi-supervised learning for both VGGish and MFCCs, as is seen in Figure 1. As was expected, the performance of the VGGish features was higher than the MFCCs. This is because the MFCCs are hand-engineered features crafted by field experts, which take an opinionated view of what information is important for classification. The assumption is that, basic harmonic structure is sufficient for distinguishing different audio signals. VGGish embeddings, however, are not created under strong assumptions. The feature representation is learned in the process of training the model because those features are efficient at representing the training set manifold that the original network was trained on, enough to assign proper labels to a wide variety of classes. Therefore, it only makes sense that the learned feature representation would perform better. 

Comparing between the supervised and semi-supervised cases in Figure 1, it can be seen that the semi-supervised cases performed better than their supervised counterparts in both feature sets. The performance differential is greater between each case in the lower proportions, whereas in the upper proportion of labeled data, the performances start to plateau and it appears that the models are converging on an optimal configuration. It can also be seen that at higher proportions of labeled data, the supervised and semi-supervised model performance is converging. This is because the training data used is becoming more and more similar, as the number of unlabeled data decreases along the x axis. 

![OOS Accuracy Curves](test-accuracy-ci.png)

Figure 1. VGG-ish and MFCC performance on semi-supervised and supervised learning within two standard deviation confidence interval.

To test if the distribution of accuracies between the supervised and semi-supervised cases were different, a Kolmogorov-Smirnov (K-S) test was performed between each set of distributions. The K-S statistic showed a statistically significant difference in both feature sets for proportions of labeled data at 50% and below. Above 50%, training on the mix of unlabeled and labeled dataset becomes less critical. This may just be because the accuracy is starting to plateau even in the supervised case, so adding more data isn’t going to improve performance much.

![K-S Test](ks-test.png)

Figure 2. The K-S statistic between the semi-supervised and supervised accuracy curves for both VGGish and MFCCs, plotted against the KS test critical value. The green portions are locations where the distributions are significantly different and the red portions are where they are not.

To visualize how the classification performance fairs between each class, confusion matrices are plotted (see Figures 3 & 4). They show the number of times that each class was predicted to be a specific class. Perfect classification would look like an identity matrix, while random choice would be a matrix of equally distributed values. The confusion matrices are shown for a subset of label proportions that were deemed to be representative of the model states, judging by the accuracy curves. The confusion matrices are shown for VGGish features in the semi-supervised and supervised cases. Looking at the classification of car horn samples in the supervised case, it can be seen that the predicted labels are pretty evenly split between car horn and street music, and to a lesser extent, siren. However when we look at the semi-supervised case, we can see that the unlabeled data actually reinforced the street music label and reduced the occurrence of the true label. This effect decreases as more labels are added, though with 30% labeled data, car horn is still classed as street music higher in the semi-supervised case than in the supervised case. This may be because there are relatively few samples available for car horn in the dataset compared to the other classes. In the sparse label case where only 5% of the data was labeled, a reduction in the mis-classified (off-diagonal) samples can be seen in the semi-supervised case.

![VGGish Semi-Supervised Confusion Matrix](conf_matrix_vggish_0.05,0.3,0.7.png)

![VGGish Supervised Confusion Matrix](conf_matrix_vggish_sup_0.05,0.3,0.7.png)

Figure 3 & 4. The confusion matrices for models trained on VGGish embeddings using semi-supervised and supervised learning, (respectively along rows). The classification performance is shown for the labeled data proportions: 0.05, 0.3, and 0.7.

In order to get a better understanding of the features used, a dimensionality technique was used to reduce the high dimensional feature space into two dimensions so that it could be plotted. For this task, t-distributed stochastic neighbor embedding (t-SNE) is a popular and powerful technique for reducing a dataset’s dimensionality while preserving local neighborhood relationships. The embeddings for both VGGish and MFCCs can be seen in Figure 5. Looking at the grouping of points in each plot, VGGish has clear, large clusters of samples with the same class, whereas this clustering is not seen in MFCCs. This shows that the feature representation of VGGish is more semantically meaningful and contains more information that characterizes the classes.  

![t-SNE Visualization](tsne_results.png)

Figure 5. The t-SNE embeddings of the VGGish and MFCC features, with colors corresponding to the class labels.

## Conclusions

In this project, the application of semi-supervised learning to audio classification was explored. Expectation Maximization was used to train a random forest classifier on both VGGish embeddings and MFCCs extracted from audio samples of the UrbanSound8k dataset. The models were then compared to their baseline performance before Expectation Maximization. It was found that for lower proportions of labeled data in both feature sets, there was a statistically significant difference between the semi-supervised and supervised cases, based on the distributions of their respective test set accuracies. 
One issue with the dataset is that there is an uneven number of samples per class. Car horn and gun shot have less than 300 samples compare to other classes, which have more than 800 samples each. Despite of having the lowest number of samples, gunshot is still managed to have the highest proportion for true positive value in semi-supervised method with 5% labeled data. However, as we mentioned earlier, the car horn is often misclassified as the street music. It is showing that whether samples size affects the prediction is still inconclusive.
As a suggestion for further improvement, this experiment could be tested with a variety of dataset sizes to see how performance is not only related to the proportion of labeled data, but to the overall quantity of labeled data. Testing across various models would also provide some interesting insights, as training convergence is also dependent on the model. Additionally, different aggregations could be tested for the input features along the temporal dimension. While mean was found to be the best performing feature tested, there are most likely more useful representations that retain more of the temporal information.

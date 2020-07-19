# Heart-Anomaly-Detection
The objective of this project is to implement deep learning methods for detecting heart anomalies by classifying them into four different categories using heart audio recordings.

The solution concept is based on (a) representing data by a two dimensional time-frequency heatmap using Mel-Frequency Cepstral Coefficient method (Fourier transform, followed by applying so called Mel filters, and then further transforming the data by using log transformation and finally applying the discrete cosine transform), and (b) using different neural networks for classification, including convolutional neural networks, recurrent neural networks, and their combinations. \

The implementation is based on using TensorFlow Keras library. The code is well structured and commented.


The dataset is from a machine learning challenge, originating from two sources – general public audio recordings using an iPhone app, and clinical trials in hospitals. Two datasets are used, one containing 176 audio files in wav format, and another containing 656 audio files in wav format. The former is a labeled data with four categories (normal heart and four (potentially) anomalous heart conditions). The latter is a labelled data with three categories (normal and two anomalous heart conditions).

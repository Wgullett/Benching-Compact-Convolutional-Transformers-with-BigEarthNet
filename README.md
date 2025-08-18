# CCT for BigEarthNet Classification
This repository contains the code and results for a project focused on classifying satellite imagery from the BigEarthNet dataset using a Compact Convolutional Transformer (CCT). The project was conducted as part of a machine learning research effort to explore the effectiveness of hybrid vision architectures on multi-label satellite image classification.

## Project Overview
The goal of this project was to implement and evaluate a Compact Convolutional Transformer (CCT) model, a hybrid architecture combining the strengths of Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs). The model was trained to classify satellite images from the BigEarthNet dataset, which consists of multi-spectral image patches from the Sentinel-1 and Sentinel-2 satellites.

## Dataset
The BigEarthNet dataset was used for this project. The original 43 land-cover classes were mapped to 19 more general classes, and one-hot encoding was applied to prepare the labels for training. The dataset was split into training, validation, and testing sets to ensure robust model evaluation.

The data preprocessing pipeline was implemented to handle the multi-label nature of the task, ensuring that the model could predict multiple land-cover classes for each image patch.

## Model Architecture
The core of this project is the Compact Convolutional Transformer (CCT) model. This architecture features a convolutional front-end that acts as a tokenizer, extracting local features from the image and generating a sequence of tokens. This sequence is then fed into a standard Transformer encoder, which captures global dependencies between the tokens.

The CCT model's key components include:

Tokenizer: A convolutional block that transforms the input image into a sequence of feature tokens.

Position Embedding: A learned embedding that provides positional information to the tokens, which is crucial for the Transformer to understand spatial relationships.

Transformer Encoder: Composed of several Transformer layers, each with a multi-head self-attention mechanism and a feed-forward network.

Sequence Pooling: A final pooling layer that aggregates the output of the Transformer to produce a single feature vector for classification.

## Experiments and Results
Two main experiments were conducted to evaluate the CCT model. Both were trained on Google Colab with a single GPU, using the AdamW optimizer and a BinaryCrossentropy loss function with label smoothing.

## Experiment 1: Baseline Model
The initial experiment used a smaller CCT configuration. The model was trained for 10 epochs.

Model Configuration:

Number of layers: 2

Number of heads: 4

Projection dimension: 64

Performance Metrics:

Accuracy: 0.7719

Micro F1-score: 0.7674

## Experiment 2: Deeper Model
The second experiment aimed to improve performance by increasing the model's capacity. The number of layers and heads were increased, and the model was trained for 50 epochs.

Model Configuration:

Number of layers: 4

Number of heads: 8

Projection dimension: 128

Performance Metrics:

Accuracy: 0.7788

Micro F1-score: 0.7656

## Results Analysis
The deeper model from Experiment 2 achieved a slightly higher accuracy, suggesting that the increased capacity helped it learn a more complex mapping from images to labels. However, its micro F1-score was slightly lower, indicating a small drop in the harmonic mean of precision and recall. This could be a sign of overfitting, even with regularization techniques, and suggests a need for further hyperparameter tuning.

## Setup and Usage
To run this project, you will need a Google Colab environment with a GPU. You must also have access to the BigEarthNet dataset on Hugging Face.

Clone the Repository:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

Install Dependencies:
The main dependencies are tensorflow, keras, huggingface-hub, and datasets.

pip install tensorflow keras huggingface-hub datasets

Run the Notebook:
Open the notebook in Google Colab and execute the cells. You will be prompted to log in to your Hugging Face account with a write token to access the dataset.

Future Work
Advanced Data Augmentation: Implement more advanced data augmentation techniques like Mixup or CutMix to improve generalization.

Hyperparameter Tuning: Systematically tune hyperparameters such as the learning rate, weight decay, and label smoothing factor to find the optimal configuration.

Regularization: Explore other regularization methods like Dropout or Stochastic Depth to prevent overfitting, particularly with larger models.

Larger Model Architectures: Experiment with even deeper CCT models or other state-of-the-art vision architectures.

Longer Training: Train the model for more epochs with a carefully managed learning rate schedule to see if performance improves.

License
This project is licensed under the MIT License.

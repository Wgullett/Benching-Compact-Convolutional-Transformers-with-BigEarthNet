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

Categorical Accuracy: 50.97%

Micro F1-score: 24.77%

Micro Precision score: 23.62%

Micro Recall score: 26.07%

## Experiment 2: Deeper Model
The second experiment aimed to improve performance by increasing the model's capacity. The number of layers and heads were increased, and the model was trained for 50 epochs.

Model Configuration:

Number of layers: 4

Number of heads: 8

Projection dimension: 128

Performance Metrics:

Accuracy: 51.52%

Micro F1-score: 25.95%

Micro Recall score: 23.1%

Mirco Precision score: 24.44%

## Results Analysis

The deeper model from Experiment 2 achieved a slightly higher accuracy, suggesting that the increased capacity helped it learn a more complex mapping from images to labels. However, its micro F1-score was slightly lower, indicating a small drop in the harmonic mean of precision and recall. This could be a sign of overfitting, even with regularization techniques, and suggests a need for further hyperparameter tuning. Additionally, the poor f1-score, percision, and recall were due to the high class imbalance, and the issue of "rare classes".

<img width="1200" height="1000" alt="image" src="https://github.com/user-attachments/assets/f8138f81-ecff-4f15-91d4-0b73e86d08ae" />

As seen in the figure above the large amount of variation makes it difficult for the model to recogonize the labels which are less likely to occur during training. Unfortunately, due to this being a benchmarking experiment, adjusting the dataset to "fix" the class imbalances would give this model an unfair advantage when compared to other models which were benchmarked following the same methodologies. 

To further confirm this, the precision of each individual label was calculated to determine if the miniority classes had lower precision rates than the majority ones. Confirming that the large class imbalance is respondsible for the differenece in Traning Accuracy, and Test Accuracy 

Urban fabric: 0.14180618975139522,
Industrial or commercial units: 0.019430051813471502,
Arable land: 0.37272177593504363,
Permanent crops: 0.05540114878636279,
Pastures: 0.19320486815415822,
Complex cultivation patterns: 0.20480880648899188,
Land principally occupied by agriculture, with significant areas of natural vegetation: 0.25513207406579486,
Agro-forestry areas: 0.051168123835459364,
Broad-leaved forest: 0.27075573502280087,
Coniferous forest: 0.3150844640686627,
Mixed forest: 0.33682954826802314,
Natural grassland and sparsely vegetated areas: 0.015978695073235686,
Moors, heathland and sclerophyllous vegetation: 0.03383458646616541,
Transitional woodland, shrub: 0.2896164867995854,
Beaches, dunes, sands: 0.004081632653061225,
Inland wetlands: 0.03715472989489123,
Coastal wetlands: 0.003952569169960474,
Inland waters: 0.1278070233890851,
Marine waters: 0.14451808582341047}

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

License
This project is licensed under the MIT License.

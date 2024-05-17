# Multi-task-fcn

## Overview

Multi-task-fcn is a collaborative project with Professor Dario Oliveira from FGV (Getúlio Vargas Foundation), aimed at addressing the crucial challenge of semantic segmentation for identifying tree species within vast and densely forested areas. The project's primary objective is to create a scalable model for monitoring deforestation and, more specifically, selective deforestation, which often remains imperceptible to the human eye.

## Project Status

The project is currently focused on semantic segmentation of tree crowns within the 
Municipality of Curitibano, Santa Catarina, Brazil. We are utilizing high-quality 
images obtained from remote sensing to conduct initial tests of our model. 
These images capture a static point in time, providing a foundation for our research.

## Future Goals

Our next major goal is to extend our efforts into the realm of time series analysis. 
We aim to implement our segmentation model on a series of lower-quality remote sensing 
images. This expansion will enable us to track changes in the forest canopy over time, 
enhancing our ability to monitor deforestation and selective deforestation more effectively.

## Problem Statement

Deforestation poses a significant threat to our environment, biodiversity, and the global 
ecosystem. However, identifying and tracking deforestation, especially when it involves 
selectively removing specific tree species, is a complex and resource-intensive task. 
Traditional monitoring methods are often unable to capture subtle changes in dense forest 
regions, making it imperative to develop advanced computer vision and machine learning 
solutions.

## Project Goals

Our project, Multi-task-fcn, aims to:

- Develop a robust semantic segmentation model capable of identifying tree species within vast and dense forest areas.
- Enable precise monitoring of deforestation, including selective deforestation, which is difficult to detect visually.
- Create a scalable solution that can be applied to a variety of geographical locations and forest types.

## How It Works

We are utilizing state-of-the-art techniques in deep learning, particularly the Fully Convolutional Network (FCN) architecture, 
to achieve precise semantic segmentation of tree species. Our model leverages multi-task learning to improve overall accuracy
and to detect changes that may indicate deforestation.

## First Results


<figure>
  <img src="https://github.com/luizfernando608/multi-task-fcn/blob/main/views/mapa_avaliacao_qualitativa.png?raw=true" alt="my alt text"/>
  <figcaption>Model Evaluation on image subset</figcaption>
</figure>


<figure>
  <img src="https://github.com/luizfernando608/multi-task-fcn/blob/main/views/versao_balanceada.png?raw=true" alt="my alt text"/>
  <figcaption>Evolution of annotated pool </figcaption>
</figure>

## User Guide
1. Install the required packages.
  ```bash
  pip install -r requirements.txt
  ```
  Install PyTorch with the following command:
  ```bash
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```
  Install cuda and cudnn if you want to use GPU.

2. Set the local paths in args.yaml file.
You should change these paths to your local paths.
  - ``ortho_image``: .tif image with remote sensing data.
  - ``train_segmentation_path``: .tif image with the ground truth label of the training set. Here, we have the contour of the tree crowns and the tree species.
  - ``test_segmentation_path``: .tif image with the ground truth label of the test set.
  - ``mask_path``: .tif image with the mask indicating the region of interest.
  - ``data_path``: The folder path to save the outputs from the model.

3. Set the some parameters
- ``nb_class``: The number of classes in the segmentation label.
- ``size_crops``: The size of the crops to be used in the training. Adjust this parameter according to the available memory.
- ``batch_size``: The batch size for the training.
- ``num_iter``: The number cycles of active learning.
- ``num_epochs``: The number of epochs for the training.
- ``num_samples``: The number of samples used to train in each epoch.

4. Run the main.py file.

## Input Data Format
### Ortho Image
- The ortho image is a .tif file with remote sensing data. Example:
<figure>
  <img src="https://github.com/Green-AI-Lab-GAIA/multi-task-fcn/blob/main/views/orthoimage_example.png?raw=true"/>
  <figcaption>Ortho Image Example</figcaption>
</figure>

### Ground Truth Segmentation Label
- The ground truth segmentation label is a .tif file with the contour of the tree crowns and the tree species. This data should be divided into two sets: training and test. The polygons should be labeled with the tree species ID, from 1 to n.
The background should be labeled with 0. The image should have only one channel.
Example of the training set:
<figure>
  <img src="https://github.com/Green-AI-Lab-GAIA/multi-task-fcn/blob/main/views/train_example.png?raw=true"/>
  <figcaption>Ground Truth Segmentation Label Example</figcaption>
</figure>

### Mask
The mask is one channel .tif file with the region of interest. The mask should be a binary image, where 1 indicates the region of interest and 0 indicates the background.
Example:
<figure>
  <img src="https://github.com/Green-AI-Lab-GAIA/multi-task-fcn/blob/main/views/mask_example.png?raw=true"/>
  <figcaption>Mask Example</figcaption>
</figure>

## Contact

If you have any questions or want to get in touch with us, please reach out to:

- Professor Dario Oliveira: dario.oliveira@fgv.br
- Researcher Luiz Luz: luiz.fernando.luz608@gmail.com

We welcome contributions, suggestions, and collaborations to further advance our mission of 
monitoring and mitigating deforestation. Together, we can make a positive impact on our 
environment and help protect our planet's forests.

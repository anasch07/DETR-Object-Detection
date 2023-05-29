# Object Detection using DETR

Our solution is an end-to-end object detection model using Transformers.Images are passed to ResNet backbone to extract a feature vector of size 256 which is further fed to the transformer encoder that yields positional embeddings, positional embeddings along with the features are fed to the transformer-decoder to predict ROI coordinates using Hungarian loss. Technologies used: Python, PyTorch.
Let’s start first  with the architecture !


## DETR Architecture Overview
![Alt text](detr-architecture.png?raw=true "Detr Architecture")

The DETR (DEtection TRansformer) architecture combines CNNs and transformers for end-to-end object detection. Here's a brief overview:

- **CNN Backbone**: Processes the input image to extract visual features.
- **Positional Encoding**: Injects positional information into the model.
- **Transformer Encoder**: Captures global contextual information and spatial relationships.
- **Transformer Decoder**: Attends to object features and generates bounding box predictions and class labels.
- **Object Queries**: Learnable queries used to attend to different objects in the image.
- **Loss Function**: Compares predictions with ground truth annotations during training.
- **Post-Processing**: Refines object localization and provides the final detected objects and labels.

DETR offers advantages such as handling variable object numbers and eliminating manual anchor box design. It has demonstrated competitive performance in object detection tasks, paving the way for future research and development.

For more details, refer to the original [DETR paper](https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers) by Meta AI.
## Usage

To run the project, follow these steps:

1. Donwload the ipynb file
2. Upload it to Google Colab
3. Run the cells in order
4. Voila ! Please Wait the training process to finish
5. You can now test the model on random images from our dataset


## Dataset
## Loss Function in DETR 

The loss function in DETR consists of two main steps: calculating the best match of predictions with respect to given ground truths using a graph technique with a cost function, and defining a loss to penalize the class and box predictions.

To achieve the best match, DETR uses an optimal bipartite matching function. This matching function finds the best predicted box for each ground truth by solving an assignment problem with the lowest cost. The matching cost considers both the class prediction and the similarity between predicted and ground truth boxes. The Hungarian algorithm is used to efficiently compute this matching cost.

After obtaining the matched pairs, the loss function is computed. It involves a negative log likelihood between all permutations of predictions and ground truth to penalize extra and incorrect boxes and classifications. The loss is down-weighted when the ground truth class label is empty (no object) to account for class imbalance.

In addition to the classification loss, DETR uses a box loss function that combines the L1 loss and the Generalized IOU loss. This loss helps predict the box directly without relying on anchor references or scaling issues. Both losses are normalized by the number of objects inside the batch.

#### Please Check the folder DETR-LOSS-BENCHMARKING for the visualization of the loss function


---


## Results and Discussion

The trained DETR model achieves promising results in detecting buses and trucks in the Open Images dataset. The model shows comparable performance to other state-of-the-art object detection methods while offering the advantages of eliminating manually designed anchor boxes and handling a variable number of objects.

However, there are some limitations and areas for improvement. The current implementation may have performance limitations due to hardware constraints. Additionally, the model's accuracy can be further improved by fine-tuning hyperparameters and exploring data augmentation techniques.

## Future Work

Given more time and resources, several avenues can be explored for further improvement and research:

- Incorporating additional object classes: Extend the model to detect and classify a wider range of objects beyond buses and trucks.

- Transfer learning: Investigate the potential of transfer learning by using pre-trained models on large-scale datasets such as ImageNet.

- Model optimization: Explore model compression techniques to reduce the model size and inference time without sacrificing performance.

- Real-time object detection: Develop a real-time object detection system by optimizing the model and leveraging hardware acceleration techniques.

## Authors

This project was developed by Anas Chaibi and Sofiene Azzabi as part of their coursework in the Department of Computer Science and Mathematics Engineering at INSAT, University of Carthage, Tunisia. For any questions or inquiries, please contact us at anas.chaibi@insat.ucar.tn and sofiene.azzabi@insat.ucar.tn.



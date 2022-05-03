# INT2-Group-33
Repo containing the model & code for INT 2 Open Assessment Group 33. A model for image classification using supervised learning on the CIFAR-10 dataset.

## Base Model
Our base model is inspired by the VGG architecture. It consists of 3-4 VGG blocks with duplicting weights using a kernel size of 3.

The base model achieved results of ~60% validation accuracy but suffered from massive overfitting issues.

## Initial Regularisation
To address the initial overfitting we added Dropout to each layer in the model. Small values initially of around 0.1-0.2 and saw an increase in 5-10% in validation accuracy but a reduction in epoch accuracy.

## Batch Normalisation
With the addition of batch normalisation to improve over accuracy via standardising the outputs from each layer we increased overall accuracy but overfitting became an issue again.

In order to offset the induced overfitting caused by adding Batch Normalisation to the model we used a technique of increasing dropout through the layers of the model. 

## Data Augmentation
Data Augmentation was another method of regularisation used to aid the model to learn general features as opposed to overfitting to the training dataset.

## Combination
The combination of all these techniques has led us to the final model in this repo.

## Conclusion
In conclusion our final validation accuracy stands at ~90%. The final model was trained over 800 epochs for approximately 2.5 hours using a GPU with Cuda capabilities.

---

**Full results table & pictures of logs to come soon.**
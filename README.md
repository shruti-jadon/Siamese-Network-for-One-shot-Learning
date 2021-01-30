# Improving Siamese Networks for One Shot Learning using Kernel Based Activation functions
Humans learn new things with a very small set of examples — e.g. a child can generalize the concept of a “Dog” from a single picture but a machine learning system needs a lot of examples to learn its features. In particular, when presented with stimuli, people seem to be able to understand new concepts quickly and then recognize variations on these concepts in future percepts. Machine learning as a field has been highly successful at a variety of tasks such as classification, web search, image and speech recognition. Often times however, these models do not do very well in the regime of low data. ‘

This is the primary motivation behind One Shot Learning; to train a model with fewer examples but generalize to unfamiliar categories without extensive retraining.

## Citation
If you find our code useful, please consider citing our work using the bibtex:
```
@incollection{jadon2021improving,
  title={Improving Siamese Networks for One-Shot Learning Using Kernel-Based Activation Functions},
  author={Jadon, Shruti and Srinivasan, Aditya Arcot},
  booktitle={Data Management, Analytics and Innovation},
  pages={353--367},
  year={2021},
  publisher={Springer}
}
```

One-shot learning cane be used for object categorization problem in computer vision. Whereas most machine learning based object categorization algorithms require training on hundreds or thousands of images and very large datasets, one-shot learning aims to learn information about object categories from one, or only a few, training images.

How to Run it:
```
Clone this repository
go to directory
python Model/train_mnist.py 
python Model/test_contrastive.py
```
The output of train_mnist will be saved as pickle, you can modify number of epochs.
```
To test the embeddings obtained score:
python Model/intracluster_score.py
```

Since One Shot Learning focuses on models which have a nonparametric approach of evaluation, we came across Kafnets(kernel based non-parametric activation functions) that have shown initial promise in this domain of training neural networks using different forms of activation functions; so as to increase non-linearity, therefore decreasing the number of layers, and increasing the accuracy in a lot of cases. This paper(https://arxiv.org/abs/1707.04035) has proposed two activation functions KAF and KAF2D, and focuses on their nature of continuity and differentiability. We have taken help of implementations of these activation functions and compared their effectiveness against traditional ones when used in the context of One Shot learning.

## MNIST Embeddings in 2D Space
![alt text](https://github.com/shruti-jadon/Siamese-Network-for-One-shot-Learning/blob/master/Images/embeddings_Combined.jpg)
## MNIST LOSS Results
![alt text](https://github.com/shruti-jadon/Siamese-Network-for-One-shot-Learning/blob/master/Images/MNISTLoss.jpg)

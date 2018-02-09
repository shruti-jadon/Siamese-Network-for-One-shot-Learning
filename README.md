# OneShotLearning
Humans learn new things with a very small set of examples — e.g. a child can generalize the concept of a “Dog” from a single picture but a machine learning system needs a lot of examples to learn its features. In particular, when presented with stimuli, people seem to be able to understand new concepts quickly and then recognize variations on these concepts in future percepts. Machine learning as a field has been highly successful at a variety of tasks such as classification, web search, image and speech recognition. Often times however, these models do not do very well in the regime of low data. ‘

This is the primary motivation behind One Shot Learning; to train a model with fewer examples but generalize to unfamiliar categories without extensive retraining.

One-shot learning cane be used for object categorization problem in computer vision. Whereas most machine learning based object categorization algorithms require training on hundreds or thousands of images and very large datasets, one-shot learning aims to learn information about object categories from one, or only a few, training images.

One way of addressing problems in One Shot learning is to develop specific features relevant to the domain of the problem; features that possess discriminative properties particular to a given target task. However, the problem with this approach is the lack of generalization that comes along with making assumptions about the structure of the input data. In this project, we make use of an approach similar to while simultaneously evaluating different activation functions[KAFNETS] that may be better suited to this task. The overall strategy we apply is two fold; train a discriminative deep learning model on a collection of data with similar/dissimilar pairs. Then, using the learned feature mappings, we can evaluate new categories.

Since One Shot Learning focuses on models which have a nonparametric approach of evaluation, we came across Kafnets(kernel based non-parametric activation functions) that have shown initial promise in this domain of training neural networks using different forms of activation functions; so as to increase non-linearity, therefore decreasing the number of layers, and increasing the accuracy in a lot of cases. This paper(https://arxiv.org/abs/1707.04035) has proposed two activation functions KAF and KAF2D, and focuses on their nature of continuity and differentiability. We have taken help of implementations of these activation functions and compared their effectiveness against traditional ones when used in the context of One Shot learning.

## MNIST Embeddings in 2D Space
![alt text](https://github.com/shruti-jadon/OneShotLearning/blob/master/embeddings_Combined.jpg)
## MNIST LOSS Results
![alt text](https://github.com/shruti-jadon/OneShotLearning/blob/master/MNISTLoss.jpg)

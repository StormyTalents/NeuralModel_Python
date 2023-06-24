### Note: recently included support for recurrent neural networks as well.

# FeedForwardModel
A library that enables incredibly **simple creation of feed forward neural models** in TensorFlow.

Simply specify the input shape, output shape, and a list of components (or layers) to make up the network.
The library then figures out the dimensions of each component, initializes each component, and builds the model automatically.
Most importantly, the model is able to infer the sizes of intermediate layers automatically, requiring you to only specify the essentials.

This enables you to make quick changes to your entire model in just a few lines.

## Examples

**fc_test.py**: An example implementation of a simple multi-class classifier.

The model used in this example:
~~~~
...
model = FeedForwardModel(2, 4, [FullyConnectedComponent(8),
                                ActivationComponent(tf.nn.sigmoid),
                                DropoutComponent(0.99),
                                FullyConnectedComponent()])
x, out = model.build()
...
~~~~

---
**cnn_test.py**: An example implementation of a CNN. Uses the MNIST dataset.

The model used in this example:
~~~~
...
model = FeedForwardModel([28, 28], 10, [Conv2DComponent(5, num_kernels=20, stride=2),
                                        ActivationComponent(tf.nn.relu),
                                        Conv2DComponent(4, num_kernels=10),
                                        ActivationComponent(tf.nn.relu),
                                        Conv2DComponent(3, num_kernels=5),
                                        ActivationComponent(tf.nn.relu),
                                        DropoutComponent(0.98),
                                        FullyConnectedComponent()])
x, out = model.build()
...
~~~~

## Not convinced yet?
### VGG16 for ImageNet
Here is an implementation of the VGG16 model for the ImageNet classification task -- classifying images of size 224x224x3 into 1000 categories:
~~~
model = FeedForwardModel([224, 224, 3], 1000, [Conv2DComponent(3, num_kernels=64),
                                               ActivationComponent(tf.nn.relu),
                                               Conv2DComponent(3, num_kernels=64),
                                               ActivationComponent(tf.nn.relu),
                                               MaxPool2DComponent(2),
                                               Conv2DComponent(3, num_kernels=128),
                                               ActivationComponent(tf.nn.relu),
                                               Conv2DComponent(3, num_kernels=128),
                                               ActivationComponent(tf.nn.relu),
                                               MaxPool2DComponent(2),
                                               Conv2DComponent(3, num_kernels=256),
                                               ActivationComponent(tf.nn.relu),
                                               Conv2DComponent(3, num_kernels=256),
                                               ActivationComponent(tf.nn.relu),
                                               MaxPool2DComponent(2),
                                               Conv2DComponent(3, num_kernels=512),
                                               ActivationComponent(tf.nn.relu),
                                               Conv2DComponent(3, num_kernels=512),
                                               ActivationComponent(tf.nn.relu),
                                               MaxPool2DComponent(2),
                                               Conv2DComponent(3, num_kernels=512),
                                               ActivationComponent(tf.nn.relu),
                                               Conv2DComponent(3, num_kernels=512),
                                               ActivationComponent(tf.nn.relu),
                                               MaxPool2DComponent(2),
                                               FullyConnectedComponent(4096),
                                               ActivationComponent(tf.nn.relu),
                                               FullyConnectedComponent(4096),
                                               ActivationComponent(tf.nn.relu),
                                               FullyConnectedComponent(1000),
                                               ActivationComponent(tf.nn.relu),
                                               FullyConnectedComponent()])
x, out = model.build()
~~~

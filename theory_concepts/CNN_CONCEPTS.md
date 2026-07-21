# Convolutional Neural Networks (CNNs or ConvNets)

Convolutional Neural Networks (CNNs or ConvNets) are specialized neural architectures that are predominantly used for several computer vision tasks, such as image classification and object recognition. These neural networks harness the power of Linear Algebra, specifically through convolution operations, to identify patterns within images.

## Convolutional neural networks have three main kinds of layers, which are:

* Convolutional layer
* Pooling layer
* Fully-connected layer

---

## Convolutional Layer

The convolutional layer is the most important layer of a CNN; responsible for dealing with the major computations. The convolutional layer includes input data, a filter, and a feature map.

**The filter** — which is also referred to as **kernel** — is a two-dimensional array of weights, and is typically a 3×3 matrix.

It is applied to a specific area of the image, and a dot product is computed between the input pixels and the weights in the filter. Subsequently, the filter shifts by a stride, and this whole process is repeated until the kernel slides through the entire image, resulting in an output array.

The resulting output array is also known as a feature map, activation map, or convolved feature.

### Number of Filters

This parameter is responsible for defining the depth of the output. If we have three distinct filters, we have three different feature maps, creating a depth of three.

### Stride

This is the distance, or number of pixels, that the filter moves over the input matrix.

### Zero-padding

This parameter is usually used when the filters do not fit the input image. This sets all elements outside the input matrix to zero, producing a larger or equally sized output. 

There are three different kinds of padding:-

1. **Valid padding**: Also known as no padding. In this specific case, the last convolution is dropped if the dimensions do not align.

2. **Same padding**: This padding ensures that the output layer has the exact same size as the input layer.

3. **Full padding**: This kind of padding increases the size of the output by adding zeros to the borders of the input matrix.

---

## Pooling Layer

The pooling layer is responsible for reducing the dimensionality of the input. It also slides a filter across the entire input — without any weights — to populate the output array. 

We have two main types of pooling:

1. **Max Pooling**: As the filter slides through the input, it selects the pixel with the highest value for the output array.

2. **Average Pooling**: The value selected for the output is obtained by computing the average within the receptive field.

The pooling layer serves the purpose of reducing complexity, improving efficiency, and limiting the risk of overfitting.

---

## Fully-connected Layer

This is the layer responsible for performing the task classification based on the features extracted during the previous layers. While both convolutional and pooling layers tend to use ReLU functions, fully-connected layers use the Softmax activation function for classification, producing a probability from 0 to 1.

*Due to its power in image recognition tasks, CNNs have been highly effective in many fields related to Computer Vision.*

*Computer Vision is a field of AI that enables computers to extract information from digital images, videos, and other visual inputs. Some common applications of computer vision today can be seen across several industries, including the following*

---


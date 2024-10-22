# Description

This program uses TensorFlow and Keras to classify handwritten digits from the MNIST dataset.
It is a sequential model, as it has single input pathed to single output. The architecture of the model is as follows:

### Input Layer
The input images are 28x28 grayscale pictures. They're represented as 2D arrays, but this layer flattens them into 1D vectors.

### Hidden Layer 1
This layer has 256 neurons and uses the rectifies linear unit (ReLU) activation function: f(x) = max(0, f(x)).

### Hidden Layer 2
This layer uses the same ReLU activation function as the first hidden layer but now has 128 neurons.

### Output Layer
This layer has 10 neurons, one for each potential digit (0-9). It uses softmax to convert the raw output scores into probabilities; the neuron with the highest probability represents the predicted digit. 

### Compilation

The model uses a Stochastic Gradient Descent (SGD) optimizer to minimize the loss function. The loss function is Space Categorical Crossentropy (SCC). The formula for SCC is: Loss = - (1/N) * Σ log( p_yi ) <br>
Where N is the total number of samples, yi is the true label for the ith sample, and p_yi is the predicited probability of the true class for the ith sample.

# How to run

### Retrain Model
To retrain the model, you run the mnist_model_builder file with the argument --retrain.<br>
For example: *python .\mnist_model_builder.py --retrain*

### Identify a Certain Digit
If you want to give it an individual digit to identify, you run the mnist_input_identifier file with the file path as an argument.<br>
For example: *python mnist_input_identifier.py ./my_digits/nine.png*
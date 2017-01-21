# NeuralNetwork
##Overview
This is the code for this video by Siraj Raval on Youtube. This is a simple single layer feedforward neural network (perceptron). We use binary digits as our inputs and expect binary digits as our outputs. We'll use backpropagation via gradient descent to train our network and make our prediction as accurate as possible.
##Dependencies
Numpy (Numpy for if I needed to format data)
##Challenge
The challenge for this project was to create a 3 layer feedforward neural network using only numpy as my dependency. By doing this, I learned exactly how backpropagation works and developed an intuitive understanding of neural networks, which will be useful for more the more complex nets I build in the future. I used a small binary dataset, which I defined programmatically.
##Graphic visualization of my code (neural network)
![alt tag](https://github.com/jamipuchi/NeuralNetwork/blob/master/Neural%20networks%20.png?raw=true)
##Results
The only numbers that were important were the first two ones. If they were different, the result should be 0 and if they were equal 1. Here is the table (the bold ones were used for the training set)
| Input  |||Output|
| ------------- |:-------------:| -----:|-----:|
|**0**|**0**|**0**|**0**|
|**0**|**0**|**1**|**0**|
|**0**|**1**|**0**|**1**|
|**0**|**1**|**1**|**1**|
|1|0|0|0|
|1|0|1|0|
|1|1|0|1|
|1|1|1|1|

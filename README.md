# Convolutional Neural Networks Labs

The objective of this lab is to give you hands-on experience designing a convolutional neural network (CNN).
These instructions will provide you with a basic CNN architecture, and provide you with default training
settings. This sample model has a mediocre performance. Your job is to use your knowledge about
model capacity, under/over fitting, and deep network architectures to improve the performance, i.e.,
reduce the testing error.   

## Setup your development environment
The code for this lab is in Python and uses the TensorFlow library. These instructions will teach you how
to edit and run the code in VS-Code, a commonly used coding editor with debugger, on your computer. 
You may use any editor and debugger that you like (e.g., Juptyer notebooks, Google Colab, Kagle, etc).
However, the instructions will assume you are using vs-code, and the instructor will be unable to provide
help for different editors. 

### Install Python 
[Download](https://www.python.org/downloads/) Python and follow the installation instructions for your OS

### Install VS-Code
[Download](https://code.visualstudio.com/Download) VS-Code and follow the installation instructions for your OS.

### Install the Python Extension in VS-Code
1. Open VS-Code 
2. Click on the Extension Icon or use the command `Ctrl+Shift+X`
3. Type "python" in the search box
4. Click on the extension called `Python` published by Microsoft
5. Click Install

### Additional Resources
If you need more details on how to install Python and the Python extension, please read these 
[guide](https://code.visualstudio.com/docs/python/python-tutorial)


## Getting the code 
After you have installed Python and the Python extension, download the code, save it to a new
folder, and then open the folder in VS-Code.

1. Click on the green button with the `Code` label on the top right of this page.
2. Click on `Download Zip`

Alternatively, you can use `git`. You can also copy and paste manually from the individual files. 


### Open the folder in VS-Code
1. Open VS-code
2. Click on the Explorer icon or press `Ctrl+Shift+E`
3. Click on `Open Folder`
4. Select and open the `cnn-lab` folder. 
5. You should see the lab files displayed on the file explorer column. 

## Code test drive
Before we start tinkering with the code, let's take a look at the various files, understand
what they do, and make sure they work.  

### The `data-exploration.py` file
This file downloads the CIFAR10 dataset and saves a sample of 25 images. The CIFAR10 dataset is
commonly used to measure the performance of a CNN.  As with any data science project, we start by doing
some data exploration. In this case, we are just going to take a look at a few images. 

To run the file, 
1. Open the `data-exploration.py` file. 
2. Press `Ctrl+Shift+D` to open the debug icon and then click `Run and debug`.
2. Alternatively, you can press `Ctrl+F5` to directly run the file.   

To open the sample images, 
1. Take a look at line 44. 
2. Open the file with the name specified in line 44. 

### The `build-cnn-model.py` file
This file builds a CNN model and saves it to a file. This is an important file as all the
design choices about the architecture of the model are coded here. 

Let's run the file and see the model that it creates. 
1. Run the file by pressing `Ctrl+F5`
2. Open the file with the name specified in line 70

#### Questions
1. Describe the architecture of the CNN. Provide details about the layers' size, type, and number. Also
   provide details about the activation functions used. 
2. Build a two-column table. The first row should have the lines of code that define the layers. The
   second column should have a screen shot of the corresponding layer in the architecture figure.
3. Draw the input and output tensors and label its sizes for each layer. 

### The `training.py` file
Now that we have defined the architecture of the model, we are ready to train it. The training algorithm
and parameter choices are coded in this file. The file also runs trains the model and then saves it to a
file. 

Lets run the file to train the model. 
1. Press `Ctrl+F5`

#### Questions
1. Which variables in the code are used as the training dataset by the function `fit`?
2. Which variables in the code are used as the testing dataset by the function `fit`?
3. Is accuracy a good performance measure? Why?
4. Is categorical cross-entropy the appropriate loss function?  Why?

### The `analysis.py` file
Now that we have a trained model, we can observe the training error, testing error, and accuracy. 

Run the file to generate the plots
1. Press `Ctrl+F5`
2. Open the accuracy plot (file name specified in line 22)
3. Open the loss plot (file name in line 35)

#### Questions
1. What is difference between validation and testing in this code? 
2. What is the difference between training/testing loss and training/testing accuracy?
3. We are using the categorical cross-entropy (CCE) to measure the loss. 
   Would a larger CCE or smaller CCE result in a lower testing error?  
4. What is the accuracy of the model?
5. Has the training/testing accuracy converged in the accuracy plot?
6. Has the training/testing loss converged  in the loss plot?
7. Did the difference between the training and testing loss/accuracy decreased or     
   increased across epochs? If it increased and decreased in the plot specify in which
   epochs it increased/decreased? 
8. Give the potential reasons that can explain the low accuracy of the model (anything 
   below 90% is considered low)

## Updating the code to improve performance. 
Now that you have a trained model, lets see if we can improve it. We can change the training parameters
and the model itself. 

### Updating the Training Parameters
1. Based on your answers to Q.5 in the `analysis.py` file [section](#the-analysispy-file), 
   make changes to the training parameters to improve the testing accuracy and loss. 
2. Describe your changes. 
3. Run the `training.py` and `analysis.py` files with your new parameters. 
4. Did the testing performance improve? Explain why or why not. 

### CNN Model updates
1. Based on your answers to Q.5 in the `analysis.py` file [section](#the-analysispy-file), make changes to the
   CNN architecture. 
2. Describe your changes. 
3. Run the `training.py` and `analysis.py` files with your new parameters. 
4. Did the testing performance improve? Explain why or why not. 

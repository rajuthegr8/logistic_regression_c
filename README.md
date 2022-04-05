# Logistic regression in C

## Assignment 

Write a classification code to use gradient descent method to learn optimal parameters as a
classifier for a gender prediction problem. As an example, consider the task of predicting someone’s
gender (Male/Female) based on their Weight and Height. So the output y = 0 (female) or 1(male);
the input is a 2D data (weight, height). You are provided two datasets with one for training, and the
other for testing. If you open each ".txt" file, you will see all the samples are lined up row by row.
Each sample contains three columns: Height, Weight, and Male.

● Height in inches

● Weight in pounds

● Male: 1 means that the measurement corresponds to a male person, and 0 means that the
measurement corresponds to a female person.

You need to "C/C++" (other languages are not acceptable) to write a program to achieve the
classification using Logistic Regression. You have the option to either turn in one piece of code
(having both training and testing tasks) or two pieces of code (with one for training and the other for
testing). Please follow the steps to implement the logistic function, cost function, partial derivative,
and gradient descent by yourself instead of using any extra library or APIs. For some basic
calculation units, such as log(), e^x, you can use the maths library of C.

## Setup

```bash
g++ train.cpp -o train
./train
```

The model will be saved as model.txt and all training and test accuracies will be printed on the screen

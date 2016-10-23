# Smooth_Random_Trees
A Differentially-Private Random Decision Forest using Smooth Sensitivity

Based on the algorithm proposed in:
S. Fletcher and M. Z. Islam. Differentially Private Random Decision Forests using Smooth Sensitivity. Expert Systems with Applications (Submitted), 2016
Which can be found at:
https://arxiv.org/abs/1606.03572 

The algorithm requires:
- training and testing data
- a list of the categorical (i.e. discrete) attributes
- the number of trees to build
- the total privacy budget 

The algorithm outputs:
- a differentially-private classification model
- six class variables that describe the model and its performance on the testing data

Please cite the above paper if you use my code :)

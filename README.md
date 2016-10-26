# Smooth_Random_Trees
A Differentially-Private Random Decision Forest using Smooth Sensitivity

Based on the algorithm proposed in:

S. Fletcher and M. Z. Islam. Differentially Private Random Decision Forests using Smooth Sensitivity. *axXiv preprint*, 2016, https://arxiv.org/abs/1606.03572 

The algorithm requires:
- training and testing data
- a list of the categorical (i.e. discrete) attributes
- the number of trees to build
- the total privacy budget 

The algorithm outputs:
- a differentially-private classification model
- six class variables that describe the model and its performance on the testing data

You can redistribute them and/or modify them under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. The main requirement of which is that you cite the above paper. Thanks!

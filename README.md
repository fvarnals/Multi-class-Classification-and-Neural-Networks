# Machine Learning - Multi-class Classification and Neural Networks

#### <em>Recognise hand-written digits using one-vs-all logistic regression and neural networks.

Training set used was a dataset of 5000 20x20 pixel grayscale images of hand-written digits (source: MNIST).<br>

[ex3data1.mat](https://github.com/fvarnals/Multi-class-Classification-and-Neural-Networks/blob/master/ex3data1.mat) contains:<br>
<code>X</code> - 5000x400 Matrix , where each row is a 400 dimensional vector representing a single training example, created by 'unrolling' the 20x20 grid of pixels for each digit image.<br>
<code>y</code> - 5000 dimensional vector that contains labels for the training set.

<strong>Steps to train the classifier and implement recognition of digit images:</strong><br>

1) <strong>[lrCostFunction.m](https://github.com/fvarnals/Multi-class-Classification-and-Neural-Networks/blob/master/lrCostFunction.m) -  Logistic regression cost function</strong><br>
<code>function [J, grad] = lrCostFunction(theta, X, y, lambda)</code> 
- Compute the cost (error) of classifications, using theta as the parameter for regularized logistic regression.<br>
- Use the cost to compute the gradient of the cost w.r.t. to the parameters, ensuring that implementation is vectorized.<br>

2) <strong>[oneVsAll.m](https://github.com/fvarnals/Multi-class-Classification-and-Neural-Networks/blob/master/oneVsAll.m) - Train a one-vs-all multi-class classifier</strong><br>
<code>function [all_theta] = oneVsAll(X, y, num_labels, lambda)</code>
- Train multiple logistic regression classifiers using <code>lrCostFunction</code> and <code>fmincg</code> to find optimal values of Theta that minimise cost/error of classification predictions.
- Return all the classifiers in a matrix <code>all_theta</code>, where the i-th row corresponds to the classifier for label i (number labels 0-9)<br>

3) <strong>[predictOneVsAll.m](https://github.com/fvarnals/Multi-class-Classification-and-Neural-Networks/blob/master/predictOneVsAll.m) - Predict using a one-vs-all multi-class classifier</strong><br>
<code>function p = predictOneVsAll(all_theta, X)</code>
- Use trained Theta values <code>all_theta</code> and Matrix<code>X</code> in combination with <code>sigmoid</code> function to return a vector <code>p</code> of predictions for each example. 
- Values of <code>p</code> vary from 1:K where p = the identified label.</em><br>
  
4) <em><strong>predict.m</em></strong>- Neural network prediction function<br>
<code>function p = predict(Theta1, Theta2, X)</code>
<em>Predict the label of an input given a trained neural network; outputs the predicted label of X given the trained weights of a neural network (Theta1, Theta2)</em>

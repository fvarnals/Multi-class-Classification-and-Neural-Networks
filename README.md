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
- Use trained Theta values <code>all_theta</code> and matrix <code>X</code> in combination with the <code>sigmoid</code> function to return a 5000x10 <code>output</code> matrix, where each row represents a sample digit from the test set (1:m), and each column represents a number label from 1:K (here 1:10).
- The values in each row of the <code>output</code> matrix represent the probability of that sample digit belonging to each number label.
- Therefore, for each row, we find the column with the highest value in order to give our predicted number label <code>p</code>, using: <code>[v p] = max(output, [], 2)</code>. Since Octave is not 0 indexed, column 10 represents '9'.

4) <strong>[predict.m](https://github.com/fvarnals/Multi-class-Classification-and-Neural-Networks/blob/master/predict.m) - Neural network prediction function</strong><br>
<code>function p = predict(Theta1, Theta2, X)</code>
- Predict the label of each input image using a trained neural network, comprising of an <strong>input layer</strong>, one <strong>hidden layer</strong>, and an <strong>output layer</strong>, with the <code>sigmoid</code> activation function applied to the outputs of each layer.
- We use a vectorised implementation to increase processing speed.
- Input layer values are the <strong>rows</strong> of matrix <code>X</code>, with a <strong>column</strong> of 1's added as the first column to account for the bias term.
- The <code>hidden_layer_inputs</code> are calculated as the dot product of <code>X</code> and <code>Theta1</code>.
- We then apply the <code>sigmoid</code> activation function to the <code>hidden_layer_inputs</code> to calculate the <code>hidden_layer_outputs</code>.
- The <code>output_layer_inputs</code> are calculated as the dot product of <code>hidden_layer_outputs</code> and <code>Theta2</code>.
- We then apply the <code>sigmoid</code> activation function to the <code>output_layer_inputs</code> to calculate the <code>output_layer_outputs</code>.
- The maximum <strong>column</strong> value of the <code>output</code> layer outputs gives the predicted number label (from the <strong>column</strong> index, using:<br> <code>[v p] = max(output_outputs, [], 2)</code>

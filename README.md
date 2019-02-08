# Machine Learning - Multi-class Classification and Neural Networks

#### <em>Implement one-vs-all logistic regression and neural networks to recognize hand-written digits.<br>

Steps:
1) <em><strong>lrCostFunction.m</em></strong>- Logistic regression cost function<br>
<code>function [J, grad] = lrCostFunction(theta, X, y, lambda)</code>
<em>Compute the cost of using theta as the parameter for regularized logistic regression and the gradient of the cost w.r.t. to the parameters, ensuring that implementation is vectorized:</em><br>

2) <em><strong>oneVsAll.m</em></strong>- Train a one-vs-all multi-class classifier<br>
<code>function [all_theta] = oneVsAll(X, y, num_labels, lambda)</code>
<em>Train multiple logistic regression classifiers using <code>lrCostFunction</code> and <code>fmincg</code>; return all the classifiers in a matrix all_theta, where the i-th row corresponds to the classifier for label i (number labels 0-9):</em><br>

3) <em><strong>predictOneVsAll.m</em></strong>- Predict using a one-vs-all multi-class classifier<br>
<code>function p = predictOneVsAll(all_theta, X)</code>
<em> Return a vector <code>p</code> of predictions for each example in the matrix X. Values of <code>p</code> vary from 1:K where p = the identified label.</em><br>
  
4) <em><strong>predict.m</em></strong>- Neural network prediction function<br>
<code>function p = predict(Theta1, Theta2, X)</code>
<em>Predict the label of an input given a trained neural network; outputs the predicted label of X given the trained weights of a neural network (Theta1, Theta2)</em>

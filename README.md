# Machine Learning - Multi-class Classification and Neural Networks

#### <em>Implement one-vs-all logistic regression and neural networks to recognize hand-written digits.<br>

Steps:
1) <em><strong>lrCostFunction.m</em></strong>- Logistic regression cost function<br>
<em>Compute the cost of using theta as the parameter for regularized logistic regression and the gradient of the cost w.r.t. to the parameters, ensuring that implementation is vectorized:</em>
<code>function [J, grad] = lrCostFunction(theta, X, y, lambda)</code>

2) <em><strong>oneVsAll.m</em></strong>- Train a one-vs-all multi-class classifier<br>
<em>Train multiple logistic regression classifiers and return all the classifiers in a matrix all_theta, where the i-th row corresponds to the classifier for label i (number labels 0-9):</em>
<code>function [all_theta] = oneVsAll(X, y, num_labels, lambda)</code>

<em><strong>predictOneVsAll.m</em></strong>- Predict using a one-vs-all multi-class classifier<br>
<em><strong>predict.m</em></strong>- Neural network prediction function<br>


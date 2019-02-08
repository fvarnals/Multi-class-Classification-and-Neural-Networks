# Machine Learning - Multi-class Classification and Neural Networks

#### <em>Implement one-vs-all logistic regression and neural networks to recognize hand-written digits.<br>

Steps:
1) <em>Compute the Cost Function and gradient for logistic regression with regularisation:</em>

<code> 
function [J, grad] = lrCostFunction(theta, X, y, lambda)
m = length(y); % number of training examples

% Initialise cost (J) and gradient (grad)
J = 0;
grad = zeros(size(theta));

h = sigmoid(X*theta);

J_unreg = (1/m) * (-y'*log(h) - (1-y)'*log(1-h));
theta(1) = 0; %don't include bias term in regularisation
J = J_unreg  + ((lambda/(2*m))*(theta'*theta)); %compute cost with theta as parameter for regularised logistic regression 

grad = (1/m) * ((h-y)'*X); 
grad = grad + ((lambda/m).*theta'); %compute the gradient of the cost w.r.t. to the parameters.
grad = grad(:);
end
</code>

<strong>Files that I had to write code to:</strong><br>
<em><strong>lrCostFunction.m</em></strong>- Logistic regression cost function<br>
<em><strong>oneVsAll.m</em></strong>- Train a one-vs-all multi-class classifier<br>
<em><strong>predictOneVsAll.m</em></strong>- Predict using a one-vs-all multi-class classifier<br>
<em><strong>predict.m</em></strong>- Neural network prediction function<br>


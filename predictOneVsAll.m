function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels
%are in the range 1..K, where K = size(all_theta, 1).
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples)

m = size(X, 1); % to get right size for 1s to add to X (remember Sj + 1)
                % to include bias node x0
num_labels = size(all_theta, 1); % to get right size for num_labels ie rows

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);m % template for output

% Add ones to the X data matrix
X = [ones(m, 1) X]; % ones added to X dfor bias term as column on far left of X

output = sigmoid(X * all_theta'); % transpose all_theta as X = 5000x401
                                  % all_theta = 10x401

[v p] = max(output, [], 2); % '2' here denotes the max column for each row.
                          % Output is a 5000x10 matrix where each row denotes
                          % a sample/pixel array, and each column denotes
                          % a number label. Therefore we want the max value
                          % in any column for a given row to tell us which
                          % number we have.


end

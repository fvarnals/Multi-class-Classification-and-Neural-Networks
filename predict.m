function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

X = [ones(m,1) X]; % add 1s to X as first column for bias term

hidden_inputs = X * Theta1'; % X is 5000 x 401, Theta1 = 25x401

hidden_outputs = sigmoid(hidden_inputs); % sigmoid of resultant 5000x25

n = size(hidden_outputs, 1); % get n for size of 1s column needed next

hidden_outputs = [ones(n,1) hidden_outputs]; %add 1s column for bias

output_inputs = hidden_outputs * Theta2'; % h_o = 5000x26, Theta2 = 10x26

output_outputs = sigmoid(output_inputs);

[v p] = max(output_outputs, [], 2); % we have 5000x10 matrix so we want the
                                    % max column value for each row to give #

end

function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);
y = labels;
m = size(data, 2);

groundTruth = full(sparse(labels, 1:m, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

td = theta * data;
td = bsxfun(@minus, td, max(td));
temp = exp(td);
%{
fprintf('Size of data %d\n');
size(data)
fprintf('Size of labels %d\n');
size(y)
fprintf('Size of theta %d\n');
size(theta)
fprintf('Size of thetagrad %d\n');
size(thetagrad)
%}
denominator = sum(temp);
p = bsxfun(@rdivide, temp, denominator);

cost = td .* groundTruth;
cost = (-1/m) * (sum(cost(:)) - sum(log(denominator(:)),1)) + (lambda / 2) * sum(sum(theta .^2));
% cost = (-1/m) * sum(sum(log(p) * y)) + (lambda / 2) * sum(sum(theta .^2));

thetagrad = (-1/m) * (groundTruth - p) * data' + lambda * theta;

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end


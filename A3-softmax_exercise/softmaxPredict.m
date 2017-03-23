function [pred] = softmaxPredict(softmaxModel, data)

% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%
% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
% Unroll the parameters from theta
theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix

DEBUG = false; % Set DEBUG to true when debugging.
if DEBUG
  data = randn(8, 10000);
end

pred = zeros(1, size(data, 2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.
%{
fprintf('Size of data %d\n');
size(data)
fprintf('Size of pred %d\n');
size(pred)
fprintf('Size of theta %d\n');
size(theta)
%}
td = theta * data;
td = bsxfun(@minus, td, max(td));
temp = exp(td);

denominator = sum(temp);
p = bsxfun(@rdivide, temp, denominator);

[prob, pred] = max(p, [], 1);
%{
fprintf('First 5 of p %d\n');
p(:,1:5)
fprintf('First 30 of prediction %d\n');
pred(1:30)
%}
% ---------------------------------------------------------------------

end


function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);
W1 = stack{1}.w;
W2 = stack{2}.w;
b1 = stack{1}.b;
b2 = stack{2}.b;
%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

  o1 = sigmoid(bsxfun(@plus, W1*data, b1));

  o2 = sigmoid(bsxfun(@plus, W2*o1, b2));
  
  td = softmaxTheta * o2;
  td = bsxfun(@minus, td, max(td));
  temp = exp(td);
  denominator = sum(temp);
  p = bsxfun(@rdivide, temp, denominator);

[prob, pred] = max(p, [], 1);


% -----------------------------------------------------------

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

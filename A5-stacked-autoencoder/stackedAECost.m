function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);
W1 = stack{1}.w;
W1grad = zeros(size(W1));
W2 = stack{2}.w;
W2grad = zeros(size(W2));
b1 = stack{1}.b;
b2 = stack{2}.b;

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

  % Forward prop
  z1 = bsxfun(@plus, W1*data, b1);
  o1 = sigmoid(z1);

  z2 = bsxfun(@plus, W2*o1, b2);
  o2 = sigmoid(z2);
  
  td = softmaxTheta * o2;
  td = bsxfun(@minus, td, max(td));
  temp = exp(td);
  denominator = sum(temp);
  p = bsxfun(@rdivide, temp, denominator);

  diff = groundTruth - p;
  
% Back prop
  % Output Layer
  outderv = -diff;
  
  % Hidden L2 Layer
  hid2derv = (softmaxTheta' * outderv) .* o2 .* (1 - o2);
  W2grad = W2grad + hid2derv * o1';
  stackgrad{2}.b = 1/M * (stackgrad{2}.b + sum(hid2derv, 2));
  
  % Hidden L1 Layer
  hid1derv = (W2' * hid2derv) .* o1 .* (1 - o1);

  W1grad = W1grad + hid1derv * data';
  stackgrad{1}.b = 1/M * (stackgrad{1}.b + sum(hid1derv, 2));

  % Computing cost
  cost = td .* groundTruth;
  cost = (-1/M) * (sum(cost(:)) - sum(log(denominator(:)),1)) + (lambda / 2) * sum(sum(softmaxTheta .^2));
  stackgrad{1}.w = 1/M * W1grad;  %+ lambda * W1; No regularization for hidden terms
  stackgrad{2}.w = 1/M * W2grad;  %+ lambda * W2; No regularization for hidden terms
  
  softmaxThetaGrad = (-1/M) * (groundTruth - p) * o2' + lambda * softmaxTheta;

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

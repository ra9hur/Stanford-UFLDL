function [cost,grad] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.
% -------------------- YOUR CODE HERE --------------------                                    

  W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
  W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
  b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
  b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

  cost = 0;
  alpha = 1;  
  W1grad = zeros(size(W1)); 
  W2grad = zeros(size(W2));
  b1grad = zeros(size(b1)); 
  b2grad = zeros(size(b2));  

  M = size(data, 2);

  % Forward prop
  z1 = bsxfun(@plus, W1*data, b1);
  o1 = sigmoid(z1);               % hidden layer : sigmoid
  o2 = bsxfun(@plus, W2*o1, b2);
  
  % Applying Sparsity Parameters
  rhohat = zeros(hiddenSize, 1);  
  rhohat = (1 ./ M) * sum(o1,2);
  p = sparsityParam;
  q = rhohat;
  sp_first = p*log(p./q);
  sp_second = (1-p)*log((1-p)./(1-q));
  sparsity = sp_first + sp_second;
  sparsity_delta = (-p./q + (1-p)./(1-q));
  
  % Output Layer
  outderv = -(data - o2);       % derivative of linear encoder = -(t - y)

  % Hidden Layer
  hidderv = bsxfun(@plus, W2' * outderv, beta * sparsity_delta);
  hidderv = hidderv .* o1 .* (1 - o1);

% Gradient and Weight Regularization  
  W1grad = W1grad + hidderv * data';  
  W1grad = alpha * (1/M * W1grad + lambda * W1);

  W2grad = W2grad + outderv * o1';
  W2grad = alpha * (1/M * W2grad + lambda * W2);

  b1grad = 1/M * alpha * (b1grad + sum(hidderv, 2));
  b2grad = 1/M * alpha * (b2grad + sum(outderv, 2));
  
% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
  J_simple = 1/M * 0.5 * sum(sum((data - o2).^2));
  cost = cost + J_simple + sum(beta * sparsity);
  cost = cost + sum(lambda * 0.5 * (W1(:).^2)) + sum(lambda * 0.5 * (W2(:).^2));
  
%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

  grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function softp = softplus(x)
    ex = 1 + exp(x);
    softp = log(ex);
endfunction

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
endfunction

%{
  %o2 = max(0.01*o2,o2);                 % output layer : Rectilinear units - Leaky ReLU
  o2 = softplus(o2);          % https://en.wikipedia.org/wiki/Rectifier_(neural_networks)

  
  % Output Layer
   outderv = diff;       % derivative of linear encoder = -(t - y)
   outderv = sigmoid(outderv);    % https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
%{
  % Referring http://eric-yuan.me/cnn/
  if (o2 > 0)
    outderv = eye(size(outderv));
  else
    outderv = zeros(size(outderv));
  endif
%}  
  
%}
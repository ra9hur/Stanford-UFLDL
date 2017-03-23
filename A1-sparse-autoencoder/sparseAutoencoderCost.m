function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
alpha = 1;                  % Prev good - 1.0   0.7
momentum = 0;               % Prev good - 0.7
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

# Jsparse(W, b) = J(W, b) + weight decay + sparsity penalty

%  (a) Implement forward propagation in your neural network, and implement the 
%      squared error term of the cost function.  Implement backpropagation to 
%      compute the derivatives.   Then (using lambda=beta=0), run Gradient Checking 
%      to verify that the calculations corresponding to the squared error cost 
%      term are correct.

% data = 64 * 10000
% X = 64 * 1000
% size(data)
% X = data(:,1:10000);
% size(X)
M = size(data, 2);
% max = size(data, 2);
% M = 1000;

  % Forward prop
  #o1 = sigmoid(bsxfun(@plus, W1*data, b1));  ...... split to enhance performance
  z1 = bsxfun(@plus, W1*data, b1);
  o1 = sigmoid(z1);

  #o2 = sigmoid(bsxfun(@plus, W2*o1, b2));    ...... split to enhance performance
  #z_2 = W1 * data + repmat(b1, 1, m);
  z2 = bsxfun(@plus, W2*o1, b2);
  o2 = sigmoid(z2);

  diff = o2 - data;

  % Applying Sparsity Parameters
  rhohat = zeros(hiddenSize, 1);    % Initialize rhohat
  rhohat = (1 ./ M) * sum(o1,2);
  p = sparsityParam;
  q = rhohat;
  sp_first = p*log(p./q);
  sp_second = (1-p)*log((1-p)./(1-q));
  sparsity = sp_first + sp_second;
  sparsity_delta = (-p./q + (1-p)./(1-q));
  
  % Output Layer
  outderv = diff .* o2 .* (1 - o2);

  % Hidden Layer
  % hidderv = W2' * outderv + (repmat(beta * sparsity_delta, 1, M));
  hidderv = bsxfun(@plus, W2' * outderv, beta * sparsity_delta);
  hidderv = hidderv .* o1 .* (1 - o1);

% Gradient and Weight Regularization  
  W2grad = W2grad + outderv * o1';
% W2grad = 1/M * (W2grad + lambda * W2);
% W2delta = momentum .* W2delta - W2grad;
% W2 = W2 - alpha * W2delta;       % alpha as applicable in SGD
  W2grad = 1/M * (W2grad + lambda * W2);
  W2grad = alpha * W2grad;       

  W1grad = W1grad + hidderv * data';
  W1grad = 1/M * (W1grad + lambda * W1);
  W1grad = alpha * W1grad;       

  b2grad = b2grad + sum(outderv, 2);
  b2grad = 1/M * b2grad;
% b2delta = momentum .* b2delta - b2grad;
% b2 = b2 - alpha * b2delta;       % alpha as applicable in SGD
  b2grad = alpha * b2grad;

  b1grad = b1grad + sum(hidderv, 2);  
  b1grad = 1/M * b1grad;
  b1grad = alpha * b1grad;

  % Computing cost
  J_simple = 1/M * 0.5 * sum(sum(diff.^2));
  cost = cost + J_simple + sum(beta * sparsity);  
  cost = cost + sum(lambda * 0.5 * (W1(:).^2)) + sum(lambda * 0.5 * (W2(:).^2));   % Weight Regularization

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

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

%{
Generalize and then, optimize
- By first generalizing, we are approximating the curvature (contours) of the error surface
- Once we know the curvature, we can apply optimization methods to speed up
  % Refer assignment3
    gradient = model_to_theta(d_loss_by_d_model(model, training_batch, wd_coefficient));
    momentum_speed = momentum_speed * momentum_multiplier - gradient;
    theta = theta + momentum_speed * learning_rate;
- Apply SGD as the last iteration to fine-tune ?
optimization strategy
Decide on the mini-batch size

2. Momentum
3. weight decay co-efficient - Lambda
4. Vary # of hidden units

http://cs231n.github.io/neural-networks-3/#sgd
http://www.erogol.com/comparison-sgd-vs-momentum-vs-rmsprop-vs-momentumrmsprop/
# Learning rate - Try 0.002, 0.01, 0.05, 0.2, 1.0, 5.0, and 20.0
  # Vanilla update
  x += - learning_rate * dx

# Momentum update (as in Hinton lecture videos)
  v = mu * v - learning_rate * dx # integrate velocity
  x += v # integrate position

# Momentum as in NN assignments
  momentum_speed = momentum_speed * momentum_multiplier - gradient;
  theta = theta + momentum_speed * learning_rate;
  
# Common practice for momentum updates
  v_prev = v # back this up
  v = mu * v - learning_rate * dx # velocity update stays the same
  x += -mu * v_prev + (1 + mu) * v # position update changes form    
  
# Nesterov Momentum
  x_ahead = x + mu * v
  v = mu * v - learning_rate * dx_ahead       # evaluate dx_ahead (the gradient at x_ahead instead of at x)
  x += v

# Adagrad - adaptive learning rate method
  # Assume the gradient dx and parameter vector x
  cache += dx**2
  x += - learning_rate * dx / np.sqrt(cache + 1e-8)

# RMSprop
  # cache : mean square
  cache = decay_rate * cache + (1 - decay_rate) * dx**2
  x += - learning_rate * dx / np.sqrt(cache + 1e-8)

Thought:
1. Apply L-BFGS for initial weight training - auto encoders
2. Use SGD to fine tune weights
3. Try out all these options
%}
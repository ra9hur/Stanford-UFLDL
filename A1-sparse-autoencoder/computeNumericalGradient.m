function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros
#numgrad = zeros(size(theta));
numgrad = [];

%% ---------- YOUR CODE HERE --------------------------------------
% Instructions: 
% Implement numerical gradient checking, and return the result in numgrad.  
% (See Section 2.3 of the lecture notes.)
% You should write code so that numgrad(i) is (the numerical approximation to) the 
% partial derivative of J with respect to the i-th input argument, evaluated at theta.  
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
% respect to theta(i).
%                
% Hint: You will probably want to compute the elements of numgrad one at a time. 

EPSILON = 0.0001;
% numel(theta): returns # of elements in theta
for i = 1:numel(theta)
  thetaPlus = theta;
  thetaPlus(i) = thetaPlus(i) + EPSILON;
  thetaMinus = theta;
  thetaMinus(i) = thetaMinus(i) - EPSILON;
  numgrad(i,:) = (J(thetaPlus) - J(thetaMinus))/(2*EPSILON);
endfor

%{
eps = 1e-4;
n = numel(numgrad);
I = eye(n, n);
for i = 1:n
    eps_vec = I(:,i) * eps;
    numgrad(i) = (J(theta + eps_vec) - J(theta - eps_vec)) / (2 * eps);
end
%}
%% ---------------------------------------------------------------
endfunction
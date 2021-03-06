function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
sum1 = 0;
sum2 = zeros(size(theta));
for i = 1:m
    sum1 = sum1 + (-y(i) * log(sigmoid( X(i,:) * theta )) - (1 - y(i)) * log(1- sigmoid( X(i,:)* theta )));
    for j = 1:size(theta)
      sum2(j) = sum2(j) + (sigmoid(X(i,:) * theta) - y(i)) * X(i, j);
    end
end

sum3 = 0;
for i = 1 : size(theta, 1)
    if i == 1
      grad(i,1) = sum2(i,1)/m;
    else
      sum3 = sum3 + theta(i,1)^2;
      grad(i,1) = sum2(i,1)/m + (lambda/m) * theta(i,1);
    end
end
J = sum1/m + (lambda/(2*m)) * sum3;


% =============================================================

end

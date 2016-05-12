function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

sum1 = 0;
sum2 = 0;
a = theta(1);
b = theta(2);
for c = 1:m
    sum1 = sum1 + (a + b * X(c,2) - y(c));
end

theta(1) = theta(1) - (alpha/m) * sum1;

for d = 1:m
    sum2 = sum2 + (a + b * X(d,2) - y(d)) * X(d,2);
end

theta(2) = theta(2) - (alpha/m) * sum2;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end

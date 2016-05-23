function [ U, theta, V, out ] = PostProcess( D, U, V, S )

if(exist('S','var'))
    theta = diag(S);
else
    theta = randn(size(U, 2), 1);
end

[row, col, val] = find(D);

lb = -1e+9*ones(size(theta));
ub = +1e+9*ones(size(theta));

% max number of iterations
param.maxIter = 1000;    
% max number of calling the function
param.maxFnCall = 1000;  
% tolerance of constraint satisfaction
param.relCha = 1e+5;      
% final objective function accuracy parameter
param.tolPG = 1e-3;   
% stored gradients
param.m = 2;

grad = sparse(row, col, val, size(D,1), size(D,2));

callfunc = @(theta) bgfsPost( theta, row, col, val, U, V, grad );

[theta, obj, iter, numCall] = lbfgsb(theta,lb,ub,callfunc, [], [], param);

theta = diag(theta);

out.obj = obj;
out.iter = iter;
out.numCall = numCall;

fprintf('bfgs iter:%d , obj:%.4d \n', iter, obj);

end

%% ---------------------------------------------------------------
function [ f, g ] = bgfsPost( s, row, col, val, U, V, grad )

res = partXY((U*diag(s))', V', row, col, length(row));
res = res' - val;

% function value
f = (1/2)*sum(res.^2);

% gradient
if nargout > 1
    setSval(grad, res, length(res));
    
    g = zeros(size(s));
    for i = 1:length(g)
        g(i) = U(:,i)'*grad*V(:,i);
    end
end

end


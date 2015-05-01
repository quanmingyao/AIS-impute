function [U, S, V, output ] = SoftImpute( D, lambda, para )
% D: sparse observed matrix

if(isfield(para, 'maxR'))
    maxR = para.maxR;
else
    maxR = min(size(D));
end

if(isfield(para, 'decay'))
    decay = para.decay;
else
    decay = 0.8;
end

objstep = 1;

maxIter = para.maxIter;
tol = para.tol*objstep;

lambdaMax = topksvd(D, 1, 5);

[row, col, data] = find(D);
[m, n] = size(D);

% U = randn(size(D, 1), 1);
% V0 = randn(size(D, 2), 1);
% V1 = V0;
% S = 1;

R = randn(n, 1);
U = powerMethod( D, R, 5, 1e-6);
[~, ~, V0] = svd(U'*D, 'econ');
V1 = V0;

Z = sparse(row, col, data, m, n);
curR = 1;

clear D;

obj = zeros(maxIter, 1);
RMSE = zeros(maxIter, 1);
Time = zeros(maxIter, 1);
t = tic;
for i = 1:maxIter
    lambdai = abs(lambdaMax - lambda)*(decay^i) + lambda;
    
    % make up sparse term Z = U*V' +spa
    spa = partXY(U', V1', row, col, length(data));
    spa = data - spa';
    Z = setSval(Z, spa, length(spa));
    
    [ R ] = filterBase( V1, V0, 1e-5);
    R = R(:, 1:min(size(R,2), maxR));
    if(para.exact == 1)
        [U, S, V] = exactSVD(U, eye(size(U,2), size(V1,2)) ,V1, Z, curR);
        S = diag(S);
        nnzS = sum(S > lambdai);
        U = U(:, 1:nnzS);
        V = V(:, 1:nnzS);
        S = S(1:nnzS);
        S = S - lambdai;
        S = diag(S);
        
        if(curR <= nnzS)
            curR = curR + 5;
        else
            curR = nnzS + 1; 
        end

        pwIter = inf;
        V0 = V1;
        V1 = V;
        U = U*S;  
    else
        [Q, pwIter] = powerMethodMatComp( U, V1, Z, R, 5, 1e-10);
        hZ = (Q'*U)*V1' + Q'*Z;
        [ U, S, V ] = SVT(hZ, lambdai);
        U = Q*(U*S);
        V0 = V1;
        V1 = V;
        
        curR = size(R, 2);
    end
    
    objVal = lambda*sum(S(:));
    objVal = objVal + (1/2)*sum(spa.^2);

    if(i > 1)
        fprintf('iter: %d; obj: %.3d (dif: %.3d); rank %d; lambda: %.1f; power(iter %d, rank %d) \n', ...
        i, objVal, obj(i - 1)- objVal, nnz(S), lambdai, pwIter, curR)
    else
        fprintf('iter: %d; obj: %.3d; rank %d; lambda: %.1f; power(iter %d, rank %d) \n', ...
        i, objVal, nnz(S), lambdai, pwIter, curR)
    end

    % testing performance
    Time(i) = toc(t);
    if(isfield(para, 'test'))
        tempS = eye(size(U, 2), size(V1, 2));
        RMSE(i) = MatCompRMSE(V, U, tempS, ...
            para.test.row, para.test.col, para.test.data);
        fprintf('RMSE %.2d \n', RMSE(i));
    end
    
    obj(i) = objVal;
    if(i > 1 && abs(obj(i) - obj(i-1)) < tol)
        break;
    end
end

output.obj = obj(1:i);
[U, S, V] = svd(U, 'econ');
V = V1*V;
output.rank = nnz(S);
output.RMSE = RMSE(1:i);
output.Time = Time(1:i);

end
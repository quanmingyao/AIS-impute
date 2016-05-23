function [U0, S, V, output ] = AccSoftImpute( D, lambda, para )
% D: [m x n] sparse observed matrix
% lambda: nuclear norm penalty
% para:
%    tol:     convergence tolerance
%    maxIter: maximum number of iterations
%    decay:   contral decay of lambda in each iteration.
%    maxR:    maximum rank allowed during iteration
%    test:    run test RMSE, see usage in TestRecsys.m

if(isfield(para, 'decay'))
    decay = para.decay;
else
    decay = 0.8;
end

if(isfield(para, 'maxR'))
    maxR = para.maxR;
else
    maxR = min(size(D));
end


maxIter = para.maxIter;
tol = para.tol;

lambdaMax = topksvd(D, 1, 5);

[row, col, data] = find(D);
[m, n] = size(D);

R = randn(n, 1);
U0 = powerMethod( D, R, 5, 1e-6);
U1 = U0;

[~, ~, V0] = svd(U0'*D, 'econ');
V1 = V0;

a0 = 1;
a1 = 1;

spa = sparse(row, col, data, m, n);

clear D m n;

obj = zeros(maxIter, 1);
RMSE = zeros(maxIter, 1);
Time = zeros(maxIter, 1);
RankIn = zeros(maxIter, 1);
RankOut = zeros(maxIter, 1);
t = tic;
for i = 1:maxIter
    lambdai = abs(lambdaMax - lambda)*(decay^i) + lambda;
    bi = (a0 - 1)/a1;
    
    % make up sparse term Z = U*V' +spa
    part0 = partXY(U0', V0', row, col, length(data));
    part1 = partXY(U1', V1', row, col, length(data));
    
    part0 = data - (1 + bi)*part1' + bi*part0';
    setSval(spa, part0, length(part0));
    
    R = filterBase( V1, V0, 1e-6);
    R = R(:, 1:min([size(R,2), maxR]));
    RankIn(i) = size(R, 2);
    
    pwTol = max(1e-6, lambdaMax*(0.95^i));
    if(para.exact == 1)
        [Ui, S, Vi] = matcompSVD( U1, V1, U0, V0, spa, bi, size(R, 2));
        
        S = diag(S);
        nnzS = sum(S > lambdai);
        Ui = Ui(:, 1:nnzS);
        Vi = Vi(:, 1:nnzS);
        S = S(1:nnzS);
        S = S - lambdai;
        S = diag(S);
        
        Ui = Ui*S;
        pwIter = inf;
    else
        [Q, pwIter] = powerMethodAccMatComp( U1, V1, U0, V0, spa, bi, R, 10, pwTol);
        hZ = ((1+bi)*(Q'*U1))*V1' - (bi*(Q'*U0))*V0' + Q'*spa;
        
        [ Ui, S, Vi ] = SVT(hZ, lambdai);
        
        Ui = Q*(Ui*S);
    end
    
    U0 = U1;
    U1 = Ui;

    V0 = V1;
    V1 = Vi;
    
    RankOut(i) = nnz(S);
    ai = (1 + sqrt(1 + 4*a0^2))/2;
    a0 = a1;
    a1 = ai;
    
    objVal = partXY(Ui', Vi', row, col, length(data));
    objVal = (1/2)*sumsqr(data - objVal');
    objVal = objVal + lambda*sum(S(:));
    obj(i) = objVal;

    if(i > 1)
        delta = obj(i - 1)- objVal;
        fprintf('iter: %d; obj: %.3d (dif: %.3d); rank %d; lambda: %.1f; power(iter %d, rank %d, tol %.2d) \n', ...
        i, objVal, delta, nnz(S), lambdai, pwIter, size(R, 2), pwTol)

        % adaptive restart
        if(delta < 0)
            a0 = 1;
            a1 = 1;
        end
    else
        fprintf('iter: %d; obj: %d; rank %d; lambda: %.1f; power(iter %d, rank %d, tol %.2d) \n', ...
        i, objVal, nnz(S), lambdai, pwIter, size(R, 2), pwTol)
    end

    % testing performance
    if(isfield(para, 'test'))
        tempS = eye(size(U1, 2), size(V1, 2));
        RMSE(i) = MatCompRMSE(U1, V1, tempS, ...
            para.test.row, para.test.col, para.test.data);
        fprintf('RMSE %.2d \n', RMSE(i));
    end
    
    % checking covergence
    if(i > 1 && abs(delta) < tol)
        break;
    end
    
    Time(i) = toc(t);
end

output.obj = obj(1:i);
[U0, S, V] = svd(U1, 'econ');
V = V1*V;
output.Rank = nnz(S);
output.RMSE = RMSE(1:i);
output.RankIn = RankIn(1:i);
output.RankOut = RankOut(1:i);
output.Time = Time(1:i);

end

%% --------------------------------------------------------------
function [Q, maxIter] = powerMethodAccMatComp( U1, V1, U0, V0, spa, bi, ...
    R, maxIter, tol)

Y = U1*((V1'*R)*(1 + bi));
Y = Y - U0*((V0'*R)*bi);
Y = Y + spa*R;

[Q, ~] = qr(Y, 0);
err = zeros(maxIter, 1);
for i = 1:maxIter
    % Y = A*(A'*Q);
    AtQ = ((1 + bi)*(Q'*U1))*V1';
    AtQ = AtQ - (bi*(Q'*U0))*V0';
    AtQ = AtQ';
    
    Y = U1*((V1'*AtQ)*(1 + bi));
    Y = Y - U0*((V0'*AtQ)*bi);
    Y = Y + spa*AtQ;
    
    [iQ, ~] = qr(Y, 0);
    
    err(i) = norm(iQ(:,1) - Q(:,1), 2);
    Q = iQ;
    
    if(err(i) < tol)
        break;
    end
end

maxIter = i;

end

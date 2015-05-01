clear; clc;

M = 2000;
N = M;
K = 5;

U = randn(M, K);
V = randn(N, K);
O = U*V';
G = 0.05*randn(M, N);
D = O + G;
ratio = 3*M*K*log(M)/(M*N);
traD = D.*(rand(M, N) < ratio);
traD = sparse(traD);
valD = (rand(M, N) < ratio);
tstD = (rand(M, N) < ratio/5);

para.maxIter = 5000;
para.tol = 1e-6;
para.decay = 0.8;
para.exact = 0;
para.maxR = floor(2*K);

% data split
lambdaMax = svds(G, 1);
gridLambda = lambdaMax*(0.9).^(1:25);

gridNMSE = zeros(2, size(gridLambda,2 ));
gridRank = zeros(1, size(gridLambda,2 ));
for g = 1:length(gridLambda)
    lambda = gridLambda(g);
    
    [U, S, V] = SoftImpute(traD, lambda, para );
    gridRank(g) = nnz(S);
    
    X = U*S*V';
    gridNMSE(1, g) = norm((X - O).*valD, 'fro')/norm(O.*valD, 'fro');
    
    [ U, S, V ] = PostProcess( U, V, traD, 1 );
    X = U*S*V';
    
    gridNMSE(2, g) = norm((X - O).*valD, 'fro')/norm(O.*valD, 'fro');
    
    
    if(g > 1 && gridNMSE(2, g) > gridNMSE(2, g - 1))
        break;
    end   
end
clear X U S V;

gridNMSE = gridNMSE(2, 1:g);
[~, lambda] = min(gridNMSE);
lambda = gridLambda(lambda);

clear gridNMSE g gridRank lambdaMax gridLambda;

para.tol = 1e-6;
para.speedup = 1;

%% ---------------------------------------------------------------
% t = tic;
% para.decay = 0.85;
% [U, ~, V, out{1}] = APGMatComp(full(traD), lambda, para );
% [ U, S, V ] = PostProcess( U, V, traD, 1 );
% Time(1) = toc(t);
% X = U*S*V';
% NMSE(1) = norm((X - O).*tstD, 'fro')/norm(O.*tstD, 'fro');
% clear X;

% t = tic;
% para.decay = 0.9;
% para.exact = 1;
% [U, ~, V, out{2}] = SoftImpute(traD, lambda, para );
% [ U, S, V ] = PostProcess( U, V, traD, 1 );
% Time(2) = toc(t);
% X = U*S*V';
% NMSE(2) = norm((X - O).*tstD, 'fro')/norm(O.*tstD, 'fro');
% clear X;

t = tic;
para.decay = 0.85;
para.exact = 1;
[U, ~, V, out{3}] = AccSoftImpute(traD, lambda, para );
[ U, S, V ] = PostProcess( U, V, traD, 1 );
Time(3) = toc(t);
X = U*S*V';
NMSE(3) = norm((X - O).*tstD, 'fro')/norm(O.*tstD, 'fro');
clear X;

t = tic;
para.decay = 0.85;
para.exact = 0;
[U, ~, V, out{4}] = AccSoftImpute(traD, lambda, para );
[ U, S, V ] = PostProcess( U, V, traD, 1 );
Time(4) = toc(t);
X = U*S*V';
NMSE(4) = norm((X - O).*tstD, 'fro')/norm(O.*tstD, 'fro');
clear X;

%% ---------------------------------------------------------------
% jmp = 5;
% para.tol = 1e-9;
% [~, ~, ~, objMin] = AccSoftImpute(traD, lambda, para );
% objMin = min(objMin.obj);
% 
% close all;
% figure;
% minIter = min([length(out{1}.obj), length(out{2}.obj), length(out{3}.obj)]);
% close all;
% % objMin = min([min(out{1}.obj), min(out{2}.obj), min(out{3}.obj)]);
% semilogy(out{1}.obj(1:jmp:minIter) - objMin, 'Marker', 'o', 'color', 'blue', 'linewidth', 2, 'MarkerSize', 8);
% hold on;
% semilogy(out{2}.obj(1:jmp:minIter) - objMin, 'Marker', 'x', 'color', 'green', 'linewidth', 2, 'MarkerSize', 8);
% semilogy(out{3}.obj(1:jmp:minIter) - objMin, 'Marker', '+', 'color','black', 'linewidth', 2, 'MarkerSize', 8);
% semilogy(out{4}.obj(1:jmp:minIter) - objMin, 'Marker', 's', 'color', 'red', 'linewidth', 2, 'MarkerSize', 8);
% 
% xlabel('iterations');
% ylabel('relative error(obj)');
% set(gca,'xtick',1:9,'xticklabel',0:jmp:9*jmp);
% axis([1, 9, 0, 5*max([out{1}.obj(1),out{2}.obj(1),out{3}.obj(1)])]);
% legend({'APG', 'Soft-Impute', 'AS-Impute', 'AIS-Impute'},'FontSize',12);
% 
% figure;
% tempTim = Time(1)/length(out{1}.obj)*(0:length(out{1}.obj)-1);
% tempObj = out{1}.obj - objMin;
% semilogy(tempTim(1:jmp:end), tempObj(1:jmp:end), 'Marker', 'o','color', 'blue', 'linewidth', 2, 'MarkerSize', 8);
% hold on;
% 
% tempTim = Time(2)/length(out{2}.obj)*(0:length(out{2}.obj)-1);
% tempObj = out{2}.obj - objMin;
% semilogy(tempTim(1:jmp:end), tempObj(1:jmp:end), 'Marker', 'x', 'color', 'green',  'linewidth', 2, 'MarkerSize', 8);
% 
% tempTim = Time(3)/length(out{3}.obj)*(0:length(out{3}.obj)-1);
% tempObj = out{3}.obj - objMin;
% semilogy(tempTim(1:jmp:end), tempObj(1:jmp:end), 'Marker', '+', 'color', 'black', 'linewidth', 2, 'MarkerSize', 8);
% 
% tempTim = Time(4)/length(out{4}.obj)*(0:length(out{4}.obj)-1);
% tempObj = out{4}.obj - objMin;
% semilogy(tempTim(1:jmp:end), tempObj(1:jmp:end), 'Marker', 's', 'color','red', 'linewidth', 2, 'MarkerSize', 8);
% 
% axis([0.01, max(Time)/2, 0, 5*max([out{1}.obj(1),out{2}.obj(1),out{3}.obj(1), out{4}.obj(1)])]);
% xlabel('seconds');
% ylabel('relative error(obj)');
% legend({'APG', 'Soft-Impute', 'AS-Impute', 'AIS-Impute'},'FontSize',12);
% 
% clear traD U V S valD O G D tstD;
% 
% save(strcat('syn-', num2str(M), '.mat'));
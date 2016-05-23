clear; clc; 
dataset = 'data/movielen100k';
load(strcat('', dataset, '.mat'));

[row, col, val] = find(data);
idx = randperm(length(val));

val = val - mean(val);
val = val/std(val);

traIdx = idx(1:floor(length(val)*0.5));
tstIdx = idx(ceil(length(val)*0.5): end);

clear idx;

traData = sparse(row(traIdx), col(traIdx), val(traIdx));
traData(size(data,1), size(data,2)) = 0;

para.test.row  = row(tstIdx);
para.test.col  = col(tstIdx);
para.test.data = val(tstIdx);

%% start testing
lambda = 9;
para.tol = 1e-3;
para.maxIter = 5000;


%% ---------------------------------------------------------------
t = tic;
para.exact = 0;
para.decay = 0.85;
[U, S, V, out] = AccSoftImpute(traData, lambda, para );
Time = toc(t);
RMSE = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));
clear U S V t;

close all;
figure;
plot(out.Time, out.RMSE);
figure;
semilogy(out.Time, out.obj - min(out.obj));

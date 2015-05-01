function [U, S, V] = matcompSVD( U1, V1, U0, V0, spa, bi, k)

m = size(U1,1);
n = size(V1,1);
Afunc  = @(x) (spa*x + (1+bi)*(U1*(V1'*x)) - bi*(U0*(V0'*x)));
Atfunc = @(y) (spa'*y + (1+bi)*(V1*(U1'*y)) - bi*(V0*(U0'*y)));

OPTIONS.tol = 1e-10;

[U, S, V] = lansvd(Afunc,Atfunc, m, n, k, 'L', OPTIONS);

end
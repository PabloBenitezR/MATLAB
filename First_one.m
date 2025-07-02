n = 100000;
a1 = ones(1,n);
a2 = -a1;
x = 0.99*a1;
J = 2*[x-a1;x-a2];
eps = 0.1;

tic;[U, S, V]=svds(J,2);toc;

s_values = diag(S);
[~, idx_min_sv] = min(s_values);
alpha = V(:, idx_min_sv);

fprintf('alpha:\n');
disp(alpha(1:2)');  


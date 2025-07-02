clear;  clc;

n  = 500000;
a1 = ones(1,n);
a2 = -a1;
x  = 0.99 * a1;       
J  = 2 * [x -  a1; x - a2];

% sdvs
[Ux,Sx] = svds(J, 1);              
[Um,Sm,Vm] = svds(J, 1, 'smallest');  
sigma  = [Sm(1,1); Sx(1,1)];       

alpha  = sign(Um(1))*Um / sum(abs(Um)); % (11)
d = Vm / norm(Vm); % (13)

fprintf('n= %d\n', n);
fprintf('singular values = [sigma_min = %.3e, sgma_max = %.3e]\n', sigma(1), sigma(2));
fprintf('alpha = [%.6f %.6f]\n', alpha(1), alpha(2));
fprintf(' d = v_min : [%.4f %.4f]\n', ...
        d(1), d(2));

t = 0.05;
F = @(xvec) [sum((xvec - a1).^2);   % f1
             sum((xvec - a2).^2)];  % f2
F_x = F(x);
tau = 0.05 * norm(F_x);
g = @(t) norm(F_x - F(x + t*d')) - tau;


pred = x + t * d';
fprintf('predicted value = [%.4f %.4f]\n', pred(1), pred(2));


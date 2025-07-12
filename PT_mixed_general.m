function PT_mixed_general(n, tau, m)
% -------------------------------------------------------------
% Mixed-power Pareto-Tracer (unconstrained, PF only)
% -------------------------------------------------------------
% n   : dimension (≥2)
% tau : predictor factor
% m   : iterations
% -------------------------------------------------------------
a1 =  ones(1,n);
a2 = -a1;

% alternating exponent masks   p1 = [4 2 4 2 …]   p2 = [2 4 2 4 …]
pow1        = 2*ones(1,n);   pow1(1:2:end) = 4;
pow2        = 2*ones(1,n);   pow2(2:2:end) = 4;

% ----- initial point x = 0 -----------------------------------
x      = zeros(1,n);
F      = zeros(m+1,2);
F(1,:) = obj_vec(x,a1,a2,pow1,pow2);

opts = optimoptions('fmincon', ...
        'Algorithm','interior-point', ...
        'SpecifyObjectiveGradient',true, ...
        'HessianApproximation','lbfgs', ...
        'Display','none');

maxStep = 0.30;
tic
for k = 1:m
    % Jacobian at x
    J = jac_general(x,a1,a2,pow1,pow2);

    [U,~,V] = svds(J,2);
    v1   = V(:,1);
    step = min(tau/norm(J*v1), maxStep);
    p    = x + step*(-sign(U(2,1))*v1)';

    % alpha from Jacobian at p
    Jp      = jac_general(p,a1,a2,pow1,pow2);
    [Up,~,~]= svds(Jp,2);
    alpha   = sign(Up(2,2))*Up(:,2)/norm(Up(:,2),1);

    % corrector (unconstrained)
    x = fmincon(@(y)obj_and_grad(y,alpha,a1,a2,pow1,pow2), ...
                p, [],[],[],[],[],[],[], opts);

    F(k+1,:) = obj_vec(x,a1,a2,pow1,pow2);
end
fprintf('Elapsed %.2f s   (n=%d, iters=%d)\n', toc, n, m);

% ---------- anchor images ------------------------------------
fa1 = obj_vec(a1,a1,a2,pow1,pow2);
fa2 = obj_vec(a2,a1,a2,pow1,pow2);

% ---------- plot PF only -------------------------------------
figure
plot(F(:,1),F(:,2),'ro-','MarkerSize',6,'DisplayName','PT'); hold on
plot(fa1(1),fa1(2),'ks','MarkerSize',8,'MarkerFaceColor','k', ...
     'DisplayName','f(a^1)');
plot(fa2(1),fa2(2),'kd','MarkerSize',8,'MarkerFaceColor','k', ...
     'DisplayName','f(a^2)');
axis square, grid on
xlabel('f_1'), ylabel('f_2')
title(sprintf('PF (n=%d)',n))
legend('Location','northwest')
end
% =============================================================
function f = obj_vec(x,a1,a2,p1,p2)
d1 = abs(x - a1).^p1;
d2 = abs(x - a2).^p2;
f  = [sum(d1) , sum(d2)];
end
% -------------------------------------------------------------
function J = jac_general(x,a1,a2,p1,p2)
d1    = x - a1;
d2    = x - a2;
grad1 = p1 .* d1 .* abs(d1).^(p1-2);
grad2 = p2 .* d2 .* abs(d2).^(p2-2);
J     = [grad1 ; grad2];
end
% -------------------------------------------------------------
function [f,g] = obj_and_grad(x,alpha,a1,a2,p1,p2)
d1 = x - a1;   d2 = x - a2;
f1 = sum(abs(d1).^p1);
f2 = sum(abs(d2).^p2);
f  = alpha(1)*f1 + alpha(2)*f2;

g1 = p1 .* d1 .* abs(d1).^(p1-2);
g2 = p2 .* d2 .* abs(d2).^(p2-2);
g  = (alpha(1)*g1 + alpha(2)*g2).';
end

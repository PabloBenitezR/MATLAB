function PF_mixed_PTonly(n, tau, m)
% -------------------------------------------------------------
%   f1(x) = (x1-1)^4 + (x2+1)^2
%   f2(x) = (x1-1)^2 + (x2+1)^4
% -------------------------------------------------------------
%   n    : dimensión     (≥2)
%   tau  : factor del predictor
%   m    : iteraciones de cada rama
% -------------------------------------------------------------

% anchors sólo para sus imágenes
a1 =  ones(1,n);
a2 = -a1;

% exponentes alternos   [4 2 4 2 …] / [2 4 2 4 …]
pow1 = 2*ones(1,n);   pow1(1:2:end) = 4;
pow2 = 2*ones(1,n);   pow2(2:2:end) = 4;

F_all = [];                              % acumulador PF
branches = [+1 -1];                      % +1→rama x1=1,  -1→rama x2=-1
maxStep  = 0.30;                         % longitud máx. del predictor

for sgn = branches
    x = zeros(1,n);                      % siempre desde el origen
    for k = 1:m
        % Jacobiano en x
        J  = jac_mixed(x,n);
        [U,~,V] = svds(J,2);
        v1 = V(:,1);

        step = min(tau/norm(J*v1), maxStep);
        p    = x + step * (sgn*v1)';     % predictor (sgn cambia de rama)

        % alpha con Jacobiano en p
        Jp      = jac_mixed(p,n);
        [Up,~,~]= svds(Jp,2);
        alpha   = sign(Up(2,2))*Up(:,2)/norm(Up(:,2),1);

        x = fmincon(@(y)obj_and_grad(y,alpha), p, ...
                    [],[],[],[],[],[],[], ...
                    optimoptions('fmincon', ...
                                 'Algorithm','interior-point', ...
                                 'SpecifyObjectiveGradient',true, ...
                                 'HessianApproximation','lbfgs', ...
                                 'Display','none'));

        F_all = [F_all ; obj_vec(x)];    % guarda nuevo punto PF
    end
end

% --------------------- PLOT PF -------------------------------
fa1 = obj_vec(a1);    % f(a1)
fa2 = obj_vec(a2);    % f(a2)

figure
plot(F_all(:,1),F_all(:,2),'ro','MarkerSize',6,'DisplayName','PT'); hold on
plot(fa1(1),fa1(2),'ks','MarkerSize',8,'MarkerFaceColor','k', ...
     'DisplayName','f(a^1)');
plot(fa2(1),fa2(2),'kd','MarkerSize',8,'MarkerFaceColor','k', ...
     'DisplayName','f(a^2)');
axis square, grid on
xlabel('$f_1$','Interpreter','latex')
ylabel('$f_2$','Interpreter','latex')
title(sprintf('PF (n = %d)  –  PT only',n),'Interpreter','latex')
legend('Location','northwest','Interpreter','latex')
end
% =============================================================
function f = obj_vec(x)
f = [ (x(1)-1)^4 + (x(2)+1)^2 , ...
      (x(1)-1)^2 + (x(2)+1)^4 ];
end
function J = jac_mixed(x,n)
dx1 = x(1)-1;  dx2 = x(2)+1;
grad_f1 = [4*dx1^3 , 2*dx2     , zeros(1,n-2)];
grad_f2 = [2*dx1   , 4*dx2^3   , zeros(1,n-2)];
J       = [grad_f1 ; grad_f2];
end
function [f,g] = obj_and_grad(x,alpha)
dx1 = x(1)-1;  dx2 = x(2)+1;
f1  = dx1^4 + dx2^2;
f2  = dx1^2 + dx2^4;
f   = alpha(1)*f1 + alpha(2)*f2;

g1 = 4*dx1^3;  g2 = 2*dx2;
h1 = 2*dx1;    h2 = 4*dx2^3;
g  = [alpha(1)*g1 + alpha(2)*h1 ;
      alpha(1)*g2 + alpha(2)*h2 ;
      zeros(numel(x)-2,1)];
end

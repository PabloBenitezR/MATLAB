function PT_mixed_PF(n, tau, m, u_max)
% -----------------------------------------------------------
% Unconstrained Pareto-Tracer for mixed-power objectives
%            f1 = (x1-1)^4 + (x2+1)^2
%            f2 = (x1-1)^2 + (x2+1)^4
% -----------------------------------------------------------
% n     : dimension  (≥2; only x1, x2 matter)
% tau   : predictor step factor   (e.g. 5)
% m     : number of iterations    (e.g. 80)
% u_max : length of the theoretical front to display (e.g. 6)
% -----------------------------------------------------------

if nargin < 4, u_max = 6; end

% --------- anchors (not used by objectives but kept for clarity) -------
a1 =  ones(1,n);
a2 = -a1;

% --------- initial point ----------------------------------------------
x  = zeros(1,n);                   % x0 = (0,0,…,0)
F  = zeros(m+1,2);                 % objective values
F(1,:) = obj_vec(x);

opts = optimoptions('fmincon', ...
        'Algorithm','interior-point', ...
        'SpecifyObjectiveGradient',true, ...
        'HessianApproximation','lbfgs', ...
        'Display','none');

maxStep = 0.30;                    % clamp predictor length

tic
for k = 1:m
    % ------- Jacobian at current x ------------------------------------
    J  = jac_mixed(x,n);

    [U,~,V] = svds(J,2);
    v1      = V(:,1);

    step = min(tau/norm(J*v1), maxStep);          % safe step length
    p    = x + step * (-sign(U(2,1))*v1)';        % predictor

    % ------- alpha from Jacobian at p ---------------------------------
    Jp       = jac_mixed(p,n);
    [Up,~,~] = svds(Jp,2);
    alpha    = sign(Up(2,2))*Up(:,2)/norm(Up(:,2),1);

    % ------- corrector (truly unconstrained) --------------------------
    x = fmincon(@(y)obj_and_grad(y,alpha), p, ...
                [],[],[],[],[],[],[], opts);

    F(k+1,:) = obj_vec(x);
end
fprintf('Elapsed %.2f s   (n=%d, iters=%d)\n', toc, n, m);

% ===========================================================
%                PLOT  :  Pareto Front
% ===========================================================
u  = linspace(0,u_max,400)';        % parameter for the two branches
PF_A = [ u.^2 , u   ];              % branch with x2 = -1
PF_B = [ u    , u.^2];              % branch with x1 =  1

figure
%plot(PF_A(:,1), PF_A(:,2),'b-','LineWidth',1.3,'DisplayName','goal'); hold on
%plot(PF_B(:,1), PF_B(:,2),'b-','LineWidth',1.3,'HandleVisibility','off');
plot(F(:,1),   F(:,2),  'ro-','MarkerSize',6,'DisplayName','PT');
axis square, grid on
xlabel('$f_1=(x_1-1)^4+(x_2+1)^2$','Interpreter','latex')
ylabel('$f_2=(x_1-1)^2+(x_2+1)^4$','Interpreter','latex')
title('Pareto Front – mixed-power, unconstrained','Interpreter','latex')
legend('Location','northwest')
end
% ===========================================================
%  vector-valued objective
% ===========================================================
function f = obj_vec(x)
f = [ (x(1)-1)^4 + (x(2)+1)^2 , ...
      (x(1)-1)^2 + (x(2)+1)^4 ];
end
% ===========================================================
%  Jacobian of [f1 ; f2]  (2 × n)
% ===========================================================
function J = jac_mixed(x,n)
dx1 = x(1)-1;   dx2 = x(2)+1;
grad_f1 = [4*dx1^3 , 2*dx2     , zeros(1,n-2)];
grad_f2 = [2*dx1   , 4*dx2^3   , zeros(1,n-2)];
J       = [grad_f1 ; grad_f2];
end
% ===========================================================
%  scalarised objective  α₁ f1 + α₂ f2    + gradient
% ===========================================================
function [f,g] = obj_and_grad(x,alpha)
dx1 = x(1)-1;   dx2 = x(2)+1;
f1  = dx1^4 + dx2^2;
f2  = dx1^2 + dx2^4;
f   = alpha(1)*f1 + alpha(2)*f2;

g1 = 4*dx1^3;   g2 = 2*dx2;
h1 = 2*dx1;     h2 = 4*dx2^3;
g  = [ alpha(1)*g1 + alpha(2)*h1 ;
       alpha(1)*g2 + alpha(2)*h2 ;
       zeros(numel(x)-2,1) ];
end

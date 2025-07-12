function PT_svds_fmincon_uncon(n, tau, m)
% -------------------------------------------------------------
% Unconstrained Pareto-Tracer with mixed-power objectives
% -------------------------------------------------------------
%   f1(x) = (x1-1)^4 + (x2+1)^2
%   f2(x) = (x1-1)^2 + (x2+1)^4
% -------------------------------------------------------------
%   n   : dimension     (≥2; only x1,x2 matter)
%   tau : predictor step factor
%   m   : number of predictor-corrector iterations
% -------------------------------------------------------------

%% data --------------------------------------------------------
a1 =  ones(1,n);        %  [ 1  1 ... 1 ]
a2 = -a1;               %  [-1 -1 ...-1 ]
x  = zeros(1,n);   
%x  = 0.9*a1;            %  start close to a1
F  = zeros(m+1,2);
X  = zeros(m+1,n);
F(1,:) = obj_vec(x);    % first PF point
X(1,:) = x;             % first PS point

opts = optimoptions('fmincon', ...
        'Algorithm','interior-point', ...
        'SpecifyObjectiveGradient',true, ...
        'HessianApproximation','lbfgs', ...
        'Display','none');

maxStep = 0.30;         % safety clamp for predictor length

tic
for k = 1:m
    %% ----- Jacobian at current x ----------------------------------
    J  = jac_mixed(x,n);

    [U,~,V] = svds(J,2);
    v1      = V(:,1);

    stepLen = min(tau/norm(J*v1), maxStep);          % clamp length
    p       = x + stepLen * (-sign(U(2,1))*v1)';     % predictor

    %% ----- Jacobian at predictor p  (for alpha) -------------------
    Jp      = jac_mixed(p,n);
    [Up,~,~]= svds(Jp,2);
    alpha   = sign(Up(2,2))*Up(:,2)/norm(Up(:,2),1);

    %% ----- corrector (truly unconstrained) ------------------------
    x = fmincon(@(y)obj_and_grad(y,alpha), p, ...
                [],[],[],[],[],[],[], opts);

    F(k+1,:) = obj_vec(x);
    X(k+1,:) = x;
end
fprintf('Elapsed %.2f s   (n=%d, iters=%d)\n', toc, n, m);

%% ---------- PLOTS -------------------------------------------------
% 1) Pareto Set reference: line segment from a1 to a2
t = linspace(0,3,200)';               % adjust length to taste

PS1 = [ 1*ones(size(t)) , -1+t ];     % vertical ray  x1 = 1
PS2 = [ 1-t , -1*ones(size(t)) ];     % horizontal ray x2 =-1

figure
plot(PS1(:,1),PS1(:,2),'b-','LineWidth',1.2); hold on
plot(PS2(:,1),PS2(:,2),'b-','LineWidth',1.2,'DisplayName','goal');
plot(X(:,1), X(:,2),'ro','MarkerSize',5,'DisplayName','PT');
axis equal, grid on
xlabel('x_1'), ylabel('x_2'), title('Pareto set')
legend

%% ---------- reference Pareto front ------------------------------
u = linspace(0,6,300)';               % same parameter for both branches
PF1 = [ u.^2 , u   ];                 % branch with  x2 = -1  (v=0)
PF2 = [ u    , u.^2];                 % branch with  x1 =  1  (u=0)

figure
plot(PF1(:,1),PF1(:,2),'b-','LineWidth',1.2); hold on
plot(PF2(:,1),PF2(:,2),'b-','LineWidth',1.2,'DisplayName','goal');
plot(F(:,1),F(:,2),'ro','MarkerSize',5,'DisplayName','PT');
axis square, grid on
xlabel('f_1'), ylabel('f_2'), title('Pareto front')
legend
end
% ===================================================================
function f = obj_vec(x)
% vector-valued objective
f = [ (x(1)-1)^4 + (x(2)+1)^2 , ...
      (x(1)-1)^2 + (x(2)+1)^4 ];
end
% -------------------------------------------------------------------
function J = jac_mixed(x,n)
% Jacobian [∇f1 ; ∇f2]  (2×n)
dx1 = x(1)-1;  dx2 = x(2)+1;
grad_f1 = [4*dx1^3 , 2*dx2     , zeros(1,n-2)];
grad_f2 = [2*dx1   , 4*dx2^3   , zeros(1,n-2)];
J       = [grad_f1 ; grad_f2];
end
% -------------------------------------------------------------------
function [f,g] = obj_and_grad(x,alpha)
% scalarised objective  α₁ f1 + α₂ f2  and its gradient
dx1 = x(1)-1;  dx2 = x(2)+1;
f1  = dx1^4 + dx2^2;
f2  = dx1^2 + dx2^4;
f   = alpha(1)*f1 + alpha(2)*f2;

g1 = 4*dx1^3;  g2 = 2*dx2;      % ∇f1
h1 = 2*dx1;    h2 = 4*dx2^3;    % ∇f2
g  = [alpha(1)*g1 + alpha(2)*h1 ;
      alpha(1)*g2 + alpha(2)*h2 ;
      zeros(numel(x)-2,1)];
end

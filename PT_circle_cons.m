%PT_circle_cons(2,0.2,100)
function PT_circle_cons(n,tau,m)

n=2;
tau=0.001;
m=100;

A = eye(n);        
alpha = ones(n,1)/n; 
a_bar = alpha' * A;  
%x0 = a_bar / norm(a_bar);
x0 = zeros(1,n); x0(1)=0.8; x0(2)=0.6;
x = x0;
disp(x);
F(1,:) = obj(x, A);

options = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
    'SpecifyObjectiveGradient',true, ...
    'SpecifyConstraintGradient',true, ...
    'HessianApproximation','lbfgs','Display','none');

tic
for k = 1:m
    J = jacobian(x,A);
    [U,S,V] = svds(J,2);
    v1 = -sign(U(2,1))* V(:,1);
    %step = tau / norm(J*v1);
    t = tau / S(1,1);
    %d = -sign(U(2,1));
    

    Jh = (2*x);
    K = null(Jh);
    H = J * K;
    [Uh,Sh,Vh] = svds(H,2);
    lambda = Vh(:,:);
    v1h = K*lambda;
    
    p = x + t * v1h';
    disp(p)
    

    Jp = jacobian(p,A);
    [Up,~,~] = svds(Jp,2);
        %disp(Up)
        %disp(Sp)
        %disp(Vp)
        alpha = sign(Up(2,2)) * Up(:,2) / norm(Up(:,2),1);
        disp(alpha)
        if min(alpha) < 0, break, end
    x = fmincon(@(y)fmin(x, alpha,A),p,[],[],[],[],[],[],@constraint,options);
    fprintf('x_new')
    disp(x)
    F(k+1,:) = obj(x, A);

end


fprintf('alpha = %s\n', mat2str(alpha,6));
fprintf('time %.2f s (n=%d, iters=%d\n', toc, n, m);

fa1 = obj(A(1,:), A);
fa2 = obj(A(2,:), A);
%%
figure;
hold on;
axis square;
grid off
plot(F(:,1),F(:,2),'o-','MarkerSize',6,'Color',"r");
plot(fa1(1),fa1(2),'ks','MarkerSize',8,'MarkerFaceColor','k','DisplayName','f(a^{1})');
plot(fa2(1),fa2(2),'ks','MarkerSize',8,'MarkerFaceColor','k','DisplayName','f(a^{2})');
xlabel('f_1');
ylabel('f_2');
title(sprintf('Pareto Front (n= %d)',n));
legend('Location','northeast');
end

function f = obj(x, A)
   m = size(A,1);
    f = zeros(m,1);
    for i = 1:m
        f(i) = norm(x - A(i,:))^2;
    end
end

function J = jacobian(x, A)
    m = size(A,1);
    J = zeros(m, size(A,2));
    for i = 1:m
        J(i,:) = 2*(x - A(i,:));
    end
end
    
function [f,g] = fmin(x, alpha, A)
    x = x(:);
    m = size(A,1); 

    f = 0;
    g = zeros(size(x));

    for i = 1:m
        ai = A(i,:)';      
        di = x - ai;
        f  = f + alpha(i) * (di' * di);
        g  = g + 2 * alpha(i) * di;       
    end
end

function [c, ceq, gradc, gradceq] = constraint(x)
    x = x(:);
    c = [];

    ceq = x' * x - 1;
    gradc = [];

    gradceq = 2 * x; 
end


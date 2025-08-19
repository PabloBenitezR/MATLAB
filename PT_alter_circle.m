%PT_circle_cons(2,0.2,100)
function PT_alter_circle(n,tau,m)

n=2000000;
tau=0.02; 
m=100;

a1 = zeros(1,n); a1(1)=1;
a2 = zeros(1,n); a2(2)=1;

x0=zeros(1,n); x0(1)=0.8; x0(2)=0.6; 
d = [+1,-1];

options = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
    'SpecifyObjectiveGradient',true, ...
    'SpecifyConstraintGradient',true, ...
    'HessianApproximation','lbfgs','Display','none');

tic

for s = 1:numel(d)
    twodir = d(s);
    x =x0;
    %disp(x);
    F(1,:,s) = obj(x,a1,a2);
    
    
for k = 1:m
    J = jacobian(x,a1,a2);
    [U,S,V] = svds(J,2);
    %v1 = -sign(U(2,1))* V(:,1);
    %step = tau / norm(J*v1);
    t = tau / S(1,1);
    %d = -sign(U(2,1));
    

    Jh = (2*x);
    %[UJh,SJh,VJh] = svds(Jh,2);
    
    %K = null(Jh);

    vt = zeros(n,1);
    vt(1) = x(2);     % 0.6
    vt(2) = -x(1);    % -0.8
    % v(3:end) = 0, ya queda implícito
    %Kt = Jh*vt;
 
    %t = 10; 
    %[Ug,Sg,Vg] = svds(Jh, t, 'smallest');
    %tol = max(size(Jh)) * eps(max(diag(Sg)));
    %nullin = diag(Sg) < tol;
    %Ng = Vg(:, nullin);
    
    %[Us,Ss,Vs] = svds(K,2);
    %Ht = J * Us; 
    %[~,~,Vth] = svds(Ht,2);
    %lambdda = Vth(:,1);
    %v1h = twodir *Us*lambdda;
     
    %[Q,R]=qr(Jh');
    %Ks = Q(:,end-n+2:end); 

    %t = 1000; 
    %[Ug,Sg,Vg] = svds(Jh, t, "largest");
    %Kn = Vg;
    
    
    %H = J * K;
    Ht = J * vt;
    %[~,~,Vh] = svds(H,2);
    %lambda = Vh(:,1);
    %v1h = twodir *K*lambda; 
    
    [~,~,Vht] = svds(Ht,2);
    lambdat = Vht(:,1);
    v1ht = twodir *vt*lambdat;
    
    %pt = x + t * v1h';
    p = x + t * v1ht';
    %disp(p)

    Jp = jacobian(p,a1,a2);
    [Up,Sp,Vp] = svds(Jp,2);
    v2p = Vp(2,2);
    %disp(Up)
    %disp(Sp)
    %disp(v2p)
    alpha = sign(Up(2,2)) * Up(:,2) / norm(Up(:,2),1);
    
    %x = fmincon(@(y)fmin(x, alpha,A),p,[],[],[],[],[],[],@constraint,options);
    x = p + (tau/2e8) * v2p;
    %fprintf('x_new')
    %disp(x)
    F(k+1,:,s) = obj(x,a1,a2);
    if min(alpha) < 0, break ,end
    if min(Sp(2,2)) < 0, break ,end
end
end


fprintf('alpha = %s\n', mat2str(alpha,6));
fprintf('time %.2f s, n=%d, tau= %.2e\n', toc, n,tau);

fa1 = obj(a1, a1,a2);
fa2 = obj(a2, a1,a2);

%%
y=1;
omit = 1;
idx = setdiff(1:size(F,2),omit);
figure;
hold on;
axis square;
grid on

F1 = squeeze(F(:,:,1));
F1 = F1(any(F1~=0,2),:);
F2 = squeeze(F(:,:,2));
F2 = F2(any(F2~=0,2),:);

plot(F1(:,1),F1(:,2),'ro-','MarkerSize',6,'DisplayName','+','Color',"r");
plot(F2(:,1),F2(:,2),'ro-','MarkerSize',6,'DisplayName','-','Color',"r");
plot(fa1(1),fa1(2),'ks','MarkerSize',8,'MarkerFaceColor','k','DisplayName','f(a^{1})');
plot(fa2(1),fa2(2),'ks','MarkerSize',8,'MarkerFaceColor','k','DisplayName','f(a^{2})');
xlabel('f_1');
ylabel('f_2');
title(sprintf('Pareto Front (n= %d)',n));
legend('Location','northeast');
end

%function f = obj(x, a1, a2)
%   m = size(a1,2);
%    f = zeros(m,1);
%    for i = 1:m
%        f() = norm(x - a1(i,:))^2;
%    end
%end
function f = obj(x,a1,a2)
d1 = norm(x-a1)^2;
d2 = norm(x-a2)^2;
f = [d1,d2];
end

%function J = jacobian(x, A)
%    m = size(A,1);
%    J = zeros(m, size(A,2));
%    for i = 1:m
%        J(i,:) = 2*(x - A(i,:));
%    end
%end

function J = jacobian(x,a1,a2)
d1 = x - a1;
d2 = x - a2;
grad1 = 2 * d1;
grad2 = 2 * d2;
J = [grad1; grad2];
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


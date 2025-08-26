%PT_circle_cons(2,0.2,100)
function PT_alter_circle(n,tau,m)

n=2000000;
tau=0.02; 
m=100;

a1 = zeros(1,n); a1(1)=1;
a2 = zeros(1,n); a2(2)=1;

x0=zeros(1,n); x0(1)=0.8; x0(2)=0.6; 
d = [+1,-1];

tic

for s = 1:numel(d)
    twodir = d(s);
    x =x0;
    F(1,:,s) = obj(x,a1,a2);
    
    
for k = 1:m
    J = jacobian(x,a1,a2);
    [U,S,V] = svds(J,2);
    t = tau / S(1,1);

    Jh = (2*x);

    vt = zeros(n,1);
    vt(1) = x(2);     % 0.6
    vt(2) = -x(1);    % -0.8
    
    [~,~,Vh] = svds(H,2);
    lambda = Vh(:,1);
    v1h = twodir *vt*lambda;
    
    p = x + t * v1h';

    Jp = jacobian(p,a1,a2);
    [Up,Sp,Vp] = svds(Jp,2);
    v2p = Vp(2,2);
    alpha = sign(Up(2,2)) * Up(:,2) / norm(Up(:,2),1);
    
   
    x = p + (tau/2e8) * v2p;
    
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


function f = obj(x,a1,a2)
d1 = norm(x-a1)^2;
d2 = norm(x-a2)^2;
f = [d1,d2];
end

function J = jacobian(x,a1,a2)
d1 = x - a1;
d2 = x - a2;
grad1 = 2 * d1;
grad2 = 2 * d2;
J = [grad1; grad2];
end
    

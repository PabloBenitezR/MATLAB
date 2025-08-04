%PT_alter(2,0.2,100)
function PT_alter(n,tau,m)

n=2;
tau=0.2;
m=100;

a1 = ones(1,n);
a2 = -a1; 
%a3 = 

pow1 = 2*ones(1,n); pow1(1:2:end) = 4;

pow2 = 2*ones(1,n); pow2(2:2:end) = 4;

d = [+1,-1];

options = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
    'SpecifyObjectiveGradient',true, ...
    'HessianApproximation','lbfgs','Display','none');

tic
for s = 1:numel(d)
    twodir = d(s);

    x = zeros(1,n);
    F(1,:,s) = obj(x,a1,a2,pow1,pow2);

    for k = 1:m
        J = jacobian(x,a1,a2,pow1,pow2);
        [U,S,V] = svds(J,2);
        v1 = -sign(U(2,1))* V(:,1);
        %step = tau / norm(J*v1);
        t = tau / S(1,1);

        p = x + t * (twodir * v1)';
        %disp(p)

        Jp = jacobian(p,a1,a2,pow1,pow2);

        [Up,Sp,Vp] = svds(Jp,2);
        v2p = Vp(2,2);
        disp(Sp)
        disp(v2p)

        alpha = sign(Up(2,2)) * Up(:,2) / norm(Up(:,2),1);
        if min(alpha) < 0
            break
        end
        if min(Sp(2,2)) < 0
            break
        end

        %x = fmincon(@(y)fmin(y,alpha,a1,a2,pow1,pow2),p,[],[],[],[],[],[],[],options);
        x = p + (tau/2e9) * v2p;
    %fprintf('x_new')
    %disp(x)
    F(k + 1,:,s) = obj(x,a1,a2,pow1,pow2);
    end 
end

fprintf('alpha = %s\n', mat2str(alpha,6));
fprintf('time %.2f s (n=%d, iters=%d\n', toc, n, m);

fa1= obj(a1,a1,a2,pow1,pow2);
fa2= obj(a2,a1,a2,pow1,pow2);
%%
figure;
hold on;
axis square;
grid off
plot(F(:,1,1),F(:,2,1),'o-','MarkerSize',6,'DisplayName','+','Color',"r");
plot(F(:,1,2),F(:,2,2),'o-','MarkerSize',6,'DisplayName','-','Color',"b");
plot(fa1(1),fa1(2),'ks','MarkerSize',8,'MarkerFaceColor','k','DisplayName','f(a^{1})');
plot(fa2(1),fa2(2),'ks','MarkerSize',8,'MarkerFaceColor','k','DisplayName','f(a^{2})');
xlabel('f_1');
ylabel('f_2');
title(sprintf('Pareto Front (n= %d)',n));
legend('Location','northeast');
end

function f = obj(x,a1,a2,p1,p2)
d1 = abs(x-a1).^p1;
d2 = abs(x-a2).^p2;
f = [sum(d1),sum(d2)];
end

function J = jacobian(x,a1,a2,p1,p2)
d1 = x - a1;
d2 = x - a2;
grad1 = p1 .* d1 .* abs(d1).^(p1-2);
grad2 = p2 .* d2 .* abs(d2).^(p2-2);
J = [grad1; grad2];
end
    
function [f,g] = fmin(x,alpha,a1,a2,p1,p2)
d1 = x - a1;
d2 = x - a2;
f1 = sum(abs(d1).^p1);
f2 = sum(abs(d2).^p2);
f = alpha(1)*f1 + alpha(2)*f2;

g1 = p1 .* d1 .* abs(d1).^(p1-2);
g2 = p2 .* d2 .* abs(d2).^(p2-2);
g = (alpha(1)*g1 + alpha(2)*g2).';
end





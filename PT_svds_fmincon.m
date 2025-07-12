%PT_svds_fmincon(1e5, 5000, 100);
%PT_svds_fmincon(4, 5, 100)
%PT_svds_fmincon(4, 20, 100)

function PT_svds_fmincon(n, tau, m)
a1 =  ones(1, n);          %  1×n
a2 = -a1;                  %  1×n
lb = -10*ones(1, n);        
ub = 10*ones(1, n);          


x = zeros(1, n);            
%x = 0.99 * a1;
%F = zeros(m + 1, 2);
%X = zeros(m + 1, n);
   
F(1,:) = obj_vec(x);
X(1,:) = x;
fprintf('x=')
disp(x)

opts = optimoptions('fmincon', ...
    'Algorithm',            'interior-point', ...  
    'SpecifyObjectiveGradient', true, ...           
    'HessianApproximation',  'lbfgs', ...
    'MaxIterations',         50, ...
    'Display',               'none');

tic
for k = 1:m
    J = jac(x,n);
    [U,~,V] = svds(J, 2);
    v1  = V(:,1);
    t   = min(tau / norm(J*v1), 0.3);
    d   = -sign(U(2,1));                 
    p   = x + (t * d) * v1';             

    %Jp
    Jp          = jac(p,n);
    [Up,~,~]    = svds(Jp,2);
    u2p   = Up(:,2);
    alpha = sign(u2p(2)) * u2p / norm(u2p,1);
    if min(alpha) < 0,    break,   end

    % new x
    x = fmincon(@(y)obj_and_grad(y,alpha), p, ...
                [],[],[],[], lb, ub, [], opts);

%fprintf('\nIter %3d\n', k);
%fprintf('   v1     = %s\n', mat2str(v1 , 6));          
%fprintf('   t      = %.4e\n', t);
%fprintf('   tau    = %.2f\n', tau);
%fprintf('   p      = %s\n', mat2str(p , 6));
%fprintf('   alpha  = %s\n', mat2str(alpha , 6));
%fprintf('   x_new  = %s\n', mat2str(x , 6));


    
     F(k + 1,:) = obj_vec(x);
     X(k + 1,:) = x;
end
elapsed = toc;
%F = F(1:k+1,:);   

t  = linspace(0,1,200)'; 
PS = (1-t).*a1(1:2) + t.*a2(1:2);

figure
%plot(PS(:,1),PS(:,2),'b-','LineWidth',1.3,'DisplayName','goal'); hold on
plot(X(:,1),X(:,2),'ro-','MarkerSize',6,'DisplayName','PT');
axis equal, grid on
xlabel('x_1'), ylabel('x_2'), title('Pareto Set')
legend('Location','southwest')

PF = [ (PS(:,1)-1).^4 + (PS(:,2)+1).^2 , ...
       (PS(:,1)-1).^2 + (PS(:,2)+1).^4 ];

figure
%plot(PF(:,1),PF(:,2),'b-','LineWidth',1.3,'DisplayName','goal'); hold on
plot(F(:,1),F(:,2),'ro-','MarkerSize',6,'DisplayName','PT');
axis square, grid on
xlabel('f_1'), ylabel('f_2'), title('Pareto Front')
legend('Location','northeast')

%figure
%plot(F(:,1),F(:,2),'o-','LineWidth',1.3);
%grid on, axis square
%xlabel('f_1'), ylabel('f_2')
%title(sprintf('PF with fmincon  (n=%d, iters=%d)', n, k))

%pad = 0.05; % 5% margin
%xmin = min(F(:,1));  xmax = max(F(:,1));
%ymin = min(F(:,2));  ymax = max(F(:,2));
%dx = pad * (xmax - xmin);   dy = pad * (ymax - ymin);
%xlim([xmin - dx, xmax + dx]);
%ylim([ymin - dy, ymax + dy]);

fprintf('Time %.2f s   (n=%d, iters=%d)\n', elapsed, n, k);
end

function f = obj_vec(x)
f1 = (x(1)-1)^4 + (x(2)+1)^2;
f2 = (x(1)-1)^2 + (x(2)+1)^4;
f  = [f1 , f2];
end

function J = jac(x,n)
dx1 = x(1)-1;  dx2 = x(2)+1;
grad_f1 = [4*dx1^3  , 2*dx2   , zeros(1,n-2)];
grad_f2 = [2*dx1    , 4*dx2^3 , zeros(1,n-2)];
J       = [grad_f1 ; grad_f2];
end

function [f,g] = obj_and_grad(x, alpha)

dx1 = x(1) - 1; 
dx2 = x(2) + 1;

f1  = dx1^4 + dx2^2;
f2  = dx1^2 + dx2^4;

f   = alpha(1)*f1 + alpha(2)*f2;

g1 = 4*dx1^3;   g2 = 2*dx2;    
h1 = 2*dx1;     h2 = 4*dx2^3;  

%f  = alpha(1)*sum(d1.^2) +alpha(2)*sum(d2.^2);
%g  = 2*alpha(1)*d1 + 2*alpha(2)*d2;   %  gradient
g  = [alpha(1)*g1 + alpha(2)*h1 ;
      alpha(1)*g2 + alpha(2)*h2 ;
      zeros(numel(x)-2,1) ];
end



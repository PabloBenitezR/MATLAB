%PT_svds_fmincon_sphere(5, 100)

function PT_svds_fmincon_sphere(tau, m)
n= 2;
a1    = [1 0];
a2    = [0 1];
a3 = [0.8 0.6];
%a1 = zeros(1,n); a1(1) = 1;         
%a2 = zeros(1,n); a2(2) = 1;      
lb = -ones(1, n);               
ub =  ones(1, n);           

%theta0 = 0.05;                       
%x = [cos(theta0) sin(theta0)]; 
x = a3; 
%F  = zeros(m +1, 2);  
%X_tracer = zeros(m+1,n);
f1 = (x(1) - a1).^4 + (x(2) - a2).^2;   
f2 = (x(1) - a1).^2 + (x(2) - a2).^4;   
F(1,:) = [f1 , f2];

disp(f1)
disp(f2)
fprintf('F(1,:)=') 
disp(F(1,:))

X_tracer(1,:) = x;
fprintf('x=')
    disp(x)

opts = optimoptions('fmincon', ...
    'Algorithm',            'interior-point', ...  
    'SpecifyObjectiveGradient', true, ...  
    'SpecifyConstraintGradient', true, ... 
    'HessianApproximation',  'lbfgs', ...
    'Display',               'none');

tic
%m = 100;
for k = 1:m

    grad1 = 4*(x(1) - a1).^3 + 2*(x(2) - a2);
    grad2 = 2*(x(1) - a1) + 4*(x(2) - a2).^3;
    %grad1 = [4 * dx1 2 * dx2];
    %grad2 = [2 * dx2 4 * dx2];
    J = [grad1' grad2'];
    %disp(J)

    [U,~,V] = svds(J, 2);
    v1  = V(:,1);
    nom  = norm(J*v1);
    %t   = tau / norm(J*v1);
    d   = -sign(U(2,1)); 
    %disp(d)

    %tau = 0.1;
    max_step = 0.2;

    t  = min(tau/nom, max_step);
    p   = x + (t * d) * v1';   
    %p = p / norm(p);
    %disp (p)
    %Jp

    gradp1 = 4*(p(1) - a1).^3 + 2*(p(2) - a2);
    gradp2 = 2*(p(1) - a1) + 4*(p(2) - a2).^3;

    Jp          = [gradp1' gradp2'];
    [Up,~,~]    = svds(Jp,2);
    u2p   = Up(:,2);
    alpha = sign(u2p(2)) * u2p / norm(u2p,1);
    %if min(alpha) < 0,    break,   end

    % new x
    x = fmincon(@(y)obj_and_grad(y,alpha,a1,a2), ...
                p, [],[],[],[], lb, ub, @sphere_constr, opts);
    fprintf('\nIter %3d\n', k);

%fprintf('   v1     = %s\n', mat2str(v1 , 6));          % 6-dec-digit vector
%fprintf('   t      = %.4e\n', t);
%fprintf('   tau    = %.2f\n', tau);

%fprintf('   p      = %s\n', mat2str(p , 6));
fprintf('   alpha  = %s\n', mat2str(alpha , 6));

fprintf('   x_new  = %s\n', mat2str(x , 6));

    
    f1 = (x(1) - a1).^4 + (x(2) - a2).^2;   
    f2 = (x(1) - a1).^2 + (x(2) - a2).^4;   
    F(k+1,:) = [f1 , f2];
    
    X_tracer(k+1,:) = x;
end
elapsed = toc;
%F = F(1:k+1,:);   

th   = linspace(0,pi/2,200);
arc  = [cos(th)', sin(th)'];

theta  = linspace(0,pi/2,200);
f1_th  = 2*(1-cos(theta));
f2_th  = 2*(1-sin(theta));
truePF = [f1_th'  f2_th'];

%[~,idx] = sort(F(:,1));      
%Fplot   = F(idx,:);

%figure
%plot(F(:,1), F(:,2), 'o-','LineWidth',1.3);
%grid on, axis square
%xlabel('f_1'), ylabel('f_2')
%title(sprintf('PF with fmincon  (n=%d, iters=%d)', n, k))

figure
plot(arc(:,1),arc(:,2),'b-','LineWidth',1.3,'DisplayName','goal'); hold on
plot(X_tracer(:,1), X_tracer(:,2),'ro-','MarkerSize',6,'DisplayName','PT');
axis equal, grid on
xlabel('x_1'), ylabel('x_2'), title('PS')
legend('Location','southwest')

figure
plot(truePF(:,1),truePF(:,2),'b-','LineWidth',1.3,'DisplayName','goal'); hold on
plot(F(:,1),F(:,2),'ro-','MarkerSize',6,'DisplayName','PT');  
axis square, grid on
xlabel('f_1');  ylabel('f_2');
title('Pareto Front');
legend('Location','northeast')

%pad = 0.05; % 5% margin
%xmin = min(F(:,1));  xmax = max(F(:,1));
%ymin = min(F(:,2));  ymax = max(F(:,2));
%dx = pad * (xmax - xmin);   dy = pad * (ymax - ymin);
%xlim([xmin - dx, xmax + dx]);
%ylim([ymin - dy, ymax + dy]);

    


fprintf('Time %.2f s   (n=%d, iters=%d)\n', elapsed, n, k);
end


function [f,g] = obj_and_grad(x,alpha,a1,a2)
d1=x-a1; 
d2=x-a2;
f  = alpha(1)*sum(d1.^2)+alpha(2)*sum(d2.^2);
g  = (2*alpha(1)*d1 + 2*alpha(2)*d2).';   % ¡vector-columna!
end
function [c,ceq,gc,gceq] = sphere_constr(x)
c=[];  ceq=sum(x.^2)-1;
if nargout>2, gc=[]; gceq = 2*x(:); end
end


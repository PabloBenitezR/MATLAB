% PT_svds_cons(5, 100)

function PT_svds_cons(tau, m)
n = 2;
a1 = [1 0];
a2 = [0 1];
a3 = [0.8 0.6];
lb = -ones(1,n);
ub = ones(1,n);

%theta = 0.05;
%x0 = [cos(theta) sin(theta)];
x  = a3;          
%F  = zeros(m +1, 2);      
%F(1,:) = [sum((x - 1).^2), sum((x + 1).^2)];

F(1,:) = obj_fun(x,a1,a2);
X(1,:) = x;
fprintf('main value of x=') 
disp(x)

opts = optimoptions('fmincon', ...
    'Algorithm',            'interior-point', ...  
    'SpecifyObjectiveGradient', true, ... 
    'SpecifyConstraintGradient', true, ...
    'HessianApproximation',  'lbfgs', ...
    'MaxIterations', 50, ...
    'Display',   'none');

max_step = 0.1; % this one uses for prevent big steps
tic
for k = 1:m
    J = jaco(x,a1,a2);
    [U,~,V] = svds(J, 2);
    v1  = V(:,1);
    t   = tau / norm(J*v1);
    d   = -sign(U(2,1));
    tt = min(t, max_step);
    p   = x + (tt * d) * v1';

    % normalize the p?, idk if is needed
    p = p / norm(p);
    p = max(p,0);

    Jp = jaco(p,a1,a2);
    [Up,~,~]    = svds(Jp,2);
    u2p   = Up(:,2);
    alpha = sign(u2p(2)) * u2p / norm(u2p,1);
    
    % alpha stop condition could be omited
    %if min(alpha) < 0,    break,   end
    
    % new value of x
    x = fmincon(@(y)grad(y,alpha), p, ...
                [],[],[],[], lb, ub, [], @cons, opts);

    fprint('\nIter %3d\n', k)
    fprintf('   v1     = %s\n', mat2str(v1 , 6));          % 6-dec-digit vector
    fprintf('   t      = %.4e\n', t);
    fprintf('   tau    = %.2f\n', tau);
    fprintf('   p      = %s\n', mat2str(p , 6));
    fprintf('   alpha  = %s\n', mat2str(alpha , 6));
    fprintf('   x_new  = %s\n', mat2str(x , 6));

    F(k + 1, :) = obj_fun(x,a1,a2);
    X(k + 1, :) = x;
end
time = toc;

figure
plot(F(:,1), F(:,2), ...
    'o-','LineWidth',1.3);
grid on, axis square
xlabel('f_1'), ylabel('f_2')
title(sprintf('Pareto Front (n=%d, iters=%d)', n, k))
legend('Location','northwest')


fprintf('time %.2f s   (n=%d, iters=%d)\n', time, n, k);
end

function f = obj_fun(x,a1,a2)
f1 = (x(1) - a1).^4 + (x(2) - a2).^2;
f2 = (x(1) - a1).^2 + (x(2) - a2).^4;
f = [f1, f2];
end

function J = jaco(x,a1,a2) % uses for this specific case
dx1 = x(1) - a1;
dx2 = x(2) - a2;
grad1 = [4 * dx1^3 2 * dx2];
grad2 = [2 * dx2 4 * dx2^3];
J = [grad1' grad2'];
end

function [f,g] = grad(x, alpha)
fun = obj_fun(x);
J = jaco(x);
f = alpha(:).' * fun(:);
g = (alpha(:).' * J);
%f  = alpha(1)*sum(d1.^2) + alpha(2)*sum(d2.^2);
%g  = (2*alpha(1)*d1 + 2*alpha(2)*d2).';   %  gradient
end

function [c, ceq, gc, gceq] = cons(x)
c    = [];                               
ceq  = sum(x.^2) - 1;                    
if nargout > 2
    gc   = [];                           
    gceq = 2 * x(:);                        
end

end

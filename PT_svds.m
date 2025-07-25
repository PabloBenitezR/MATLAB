function PT_svds(n, tau, m)

a1 =  ones(1, n);
a2 = -a1;

x  = 0.99 * a1;
%F  = zeros(m + 1, 2);
F(1,:) = [sum((x - 1).^2), sum((x + 1).^2)];

opts = optimoptions('fminunc', ...
    'HessianApproximation','lbfgs', ...
    'SpecifyObjectiveGradient', true, ...   
    'MaxIterations', 50, ...                
    'Display','none');

tic
for k = 1:m
    J = [x - a1; x - a2];
    [U,~,V] = svds(J,2);
    v1  = V(:,1);
    t   = tau / norm(J*v1);
    d   = sign(U(2,1)) * -1;     
    p   = x + (t * d) * v1';

    % Jp
    Jp          = [p - a1; p - a2];
    [Up,~,~]    = svds(Jp,2);
    u2p         = Up(:,2);
    alpha       = sign(u2p(2)) * u2p / norm(u2p,1);
    if min(alpha) < 0,  break, end

    % new x
    x = fminunc(@(y) obj_and_grad(y,alpha,a1,a2), p, opts);

    F(k+1,:) = [sum((x-1).^2) , sum((x+1).^2)];
end
elapsed = toc;

plot(F(:,1),F(:,2),'o-'), axis square
xlabel('f_1'), ylabel('f_2')
title(sprintf('PF svds (n=%d, iters=%d)', n, k))
fprintf('Time %.2f s  (n=%d, m=%d)\n', elapsed, n, k);

pad = 0.05;            % 5 % margin
xmin = min(F(:,1));  xmax = max(F(:,1));
ymin = min(F(:,2));  ymax = max(F(:,2));
dx = pad * (xmax - xmin);
dy = pad * (ymax - ymin);
xlim([xmin - dx, xmax + dx]);
ylim([ymin - dy, ymax + dy]);

end


function [f,g] = obj_and_grad(x, alpha, a1, a2)
d1 = x - a1;
d2 = x - a2;
f  = alpha(1)*sum(d1.^2) + alpha(2)*sum(d2.^2);
g  = 2*alpha(1)*d1 + 2*alpha(2)*d2;   % the gradient
end


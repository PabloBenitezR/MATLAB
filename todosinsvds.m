clear;
n      = 100000;
a1     = ones(1,n);     
a2     = -a1;           
w_vec  = linspace(0,1,10)';    
fvals  = zeros(numel(w_vec), 2);

% fmincon 
opts = optimoptions('fmincon', ...
        'Algorithm','sqp', ...
        'Display','off' );


for i = 1:numel(w_vec)
    w = w_vec(i);
    obj = @(t) w*n*(t-1).^2 + (1-w)*n*(t+1).^2;
    lb = -1;  ub = +1;
    t0 = 0;

    t_star = fmincon(obj, t0, [],[],[],[], lb, ub, [], opts);

    f1 = n*(t_star - 1)^2;
    f2 = n*(t_star + 1)^2;
    fvals(i,:) = [f1, f2];
end

% Plot
figure;
plot(fvals(:,1), fvals(:,2), 'o-','LineWidth',1.5,'MarkerSize',6);
xlabel('f_1 ');
ylabel('f_2 ');
title('Pareto Front');
grid on;

%but if only use fmincon and take the problem as an unconstrained 1-D
% minimization instead svds

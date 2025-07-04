clear;
%% parameters
n = 10;
a1 = ones(1,n);
a2 = -a1;
tau = 100; % whatever
m = 10; % iterations
tol = 1e-8; %tol

%% 
X = zeros (m+1,n); % decision points
F = zeros(m+1,2); % objective values

%% start
x = 0.99*a1; % 1xn
X(1,:) = x;
F(1,:) = [sum((x-1).^2) , sum((x+1).^2)];

%% predictor
tic;
for k=1:m
    J = [x-a1;x-a2]; % 2xn

    [U,S,V] = svds(J,2);
    u1 = U(:,1); %2x1
    u2 = U(:,2); %2x1
    v1 = V(:,1); %nx1
    s1 = S(1,1);

    t = tau/s1; % 1x1
    p = x + (u1(1)*t) * v1';     % 1×n

%fprintf('value of x:\n'); disp(x)
%fprintf('direction:\n'); disp(u1)
%fprintf('alpha:\n'); disp(u2)
%fprintf('v1:\n'); disp(v1)

    Jp = [p-a1;p-a2];
    [U_p, S_p, V_p] = svds(Jp, 2);
    u2_p = U_p(:, 2);

%fprintf('value of p:\n'); disp(p) 
%fprintf('direction_p:\n'); disp(u1_p)
%fprintf('alpha_p:\n'); disp(u2_p)
%fprintf('v1_p:\n'); disp(v1_p)

    if any(u2_p < 0),  u2_p = -u2_p;  end
    alpha = u2_p / sum(u2_p);
    disp(alpha)

    obj = @(x) alpha(1)*sum((x-1).^2) + ... 
               alpha(2)*sum((x+1).^2);
    x_new  = fminunc(obj,p);

    %opts = optimoptions('fmincon', ...
    %    'Algorithm','interior-point', ...
    %    'HessianApproximation','lbfgs', ...
    %    'Display','iter');
    %x_new = fmincon(obj,p,[],[],[],[],[],[],[],opts);

    %c     = alpha(1) - alpha(2);   
    %x_new = c * ones(1,n); 

    Jnew = [x_new-a1; x_new-a2];
    res  = norm(Jnew.' * u2_p);
    disp(res)

    if res > tol 
        if any(u2_p < 0),  u2_p = -u2_p;  end
        warning('Iter %d: ||Jt*u2|| = %.2e  > tol; KKt not yet.',k,res);
    end

    % memo
    X(k+1,:) = x_new;
    F(k+1,:) = [sum((x_new-1).^2), sum((x_new+1).^2)];
    x = x_new;
end

plot(F(:,1),F(:,2),'o-');  xlabel('f_1'); ylabel('f_2');

t_total = toc;
fprintf('\ntime: %.3f s\n', t_total);








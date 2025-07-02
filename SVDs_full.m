clear;
n      = 100;
a1     = ones(1,n);
a2     = -a1;
epsilon = 1e-2; 
tau = 1;
w_vec  = linspace(0,1,10)';   
num_w  = numel(w_vec);

% memo
fvals   = zeros(num_w,2);     
x_proj  = zeros(num_w,2);    

% fmincon 
opts = optimoptions('fmincon', ...
        'Algorithm','sqp', ...
        'Display','off' );

x_curr = 0.99 * a1;            

%% 
for i = 1:num_w
    w = w_vec(i);
    
    Jk = 2 * [ x_curr - a1 ;      
               x_curr - a2 ];
    [~, Sk, Vk] = svds(Jk, 2);    
    [~, idx_min] = min(diag(Sk));
    alpha = Vk(:, idx_min);       
    
    x_pred = x_curr + epsilon * alpha';   
    
 
    obj_full = @(x) w*norm(x - a1)^2 + (1-w)*norm(x - a2)^2;
    x_new    = fmincon(obj_full, x_pred, [],[],[],[], [], [], [], opts);

    f1 = norm(x_new - a1)^2;
    f2 = norm(x_new - a2)^2;
    fvals(i,:) = [f1, f2];
    
    x_proj(i,:) = [ x_new(1), x_new(2) ];

    x_curr = x_new;
end

%%
figure;
plot(fvals(:,1), fvals(:,2), 'o-','LineWidth',1.5,'MarkerSize',6);
xlabel('f_1 ');
ylabel('f_2 ');
title('Pareto Front');
grid on;

%% 
figure;
plot(x_proj(:,1), x_proj(:,2), 's-','LineWidth',1.5,'MarkerSize',6);
xlabel('x_1');
ylabel('x_2');
axis equal;  
grid on;



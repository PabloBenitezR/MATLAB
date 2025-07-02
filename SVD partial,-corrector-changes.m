clear;  clc;
n          = 100000;          
num_iter   = 15;                  
tau = 0.1;               

%% define objs
a1 = ones(1,n);
a2 = -a1;
F   = @(xvec) [sum((xvec -  a1).^2); ...   % (15)
               sum((xvec -  a2).^2)];      
f1f2= @(xvec) deal(n*(xvec(1)-1)^2, n*(xvec(1)+1)^2); 

%% memory
 f_path  = zeros(num_iter+1,2);
 t_path  = zeros(num_iter+1,1);   

%% define start
 x = 0.99 * a1;
 [f1,f2] = f1f2(x); 
 f_path(1,:) = [f1,f2]; 
 t_path(1) = 0.99;
 fprintf('Iter 0:  t = %.4f,  f1 = %.3e,  f2 = %.3e\n', t_path(1), f1, f2);

%% predictor
for k = 1:num_iter
    J = 2*[x -  a1; x - a2];  
    [Um,Sm,Vm] = svds(J,1,'smallest');
    alpha     = sign(Um(1))*Um / sum(abs(Um));  
    d = Vm / norm(Vm);                     

    Fx   = F(x);
    tau  = tau * norm(Fx);
    g    = @(t) norm(Fx - F(x + t*d')) - tau;
    tmax = 1;
    while g(tmax) < 0,   tmax = tmax*2;   end
    t_r  = fzero(g,[0,tmax]);
    x_pred = x + t_r*d';

    %% corrector
    t_star = alpha(1) - alpha(2);          
    x_corr = t_star * x;            

    %% 
    x      = x_corr;                       % update
    [f1,f2]= f1f2(x);
    f_path(k+1,:) = [f1,f2];
    t_path(k+1)  = t_star;
    fprintf('Iter %d:  t_pred = %.4e,  t_corr = %.4f,  f1 = %.3e,  f2 = %.3e\n', ...
            k, t_r, t_star, f1, f2);
end


 t = linspace(-1,1,400);
 f1_curve = n*(t-1).^2;  f2_curve = n*(t+1).^2;
 figure; hold on; grid on; box on;
 plot(f1_curve,f2_curve,'b-','LineWidth',1.4,'DisplayName','Pareto front');
 plot(f_path(:,1),f_path(:,2),'ro-','LineWidth',1.5,'MarkerSize',6,'DisplayName','PC trajectory');
 xlabel('f_1'); ylabel('f_2');
 title(sprintf('Predictor–Corrector trajectory (%d steps)',num_iter));
 legend('Location','northeast');
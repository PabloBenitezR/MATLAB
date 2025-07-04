
clear;
%% 
n   = 100;                      
a1  =  ones(1,n);
a2  = -a1;
tau = 100;                         
m   = 10;                          
tol = 1e-8;                       

%% memo
X = zeros(m+1,n);                  
F = zeros(m+1,2);                  

%% predictor
x      = 0.99*a1;
X(1,:) = x;
F(1,:) = [sum((x-1).^2) , sum((x+1).^2)];


tic;
for k = 1:m
  
    J = [x-a1; x-a2];

    [U,S,V] = svd(J,'econ');       
    u1 = U(:,1);                 
    s1 = S(1,1);                 
    v1 = V(:,1);                   

    t = tau/s1;
    p = x + (u1(1)*t) * v1.';    

    Jp = [p-a1; p-a2];
    [Up,~,~] = svd(Jp,'econ');    
    u2p = Up(:,2);               

    
    if any(u2p<0),  u2p = -u2p;  end
    alpha = u2p / sum(u2p);       

    
    c     = alpha(1) - alpha(2);   % λ1 − λ2
    x_new = c * ones(1,n);         % 1×n   ← mínimo de g(x)

   
    res = norm(([x_new-a1; x_new-a2].') * u2p);
    if res > tol
        warning('Iter %d: ||J^T u2|| = %.2e  > tol',k,res);
    end

    
    X(k+1,:) = x_new;
    F(k+1,:) = [sum((x_new-1).^2), sum((x_new+1).^2)];
    x        = x_new;
end


%% gráfica frente aproximado
plot(F(:,1),F(:,2),'o-','LineWidth',1.2), grid on
tot_t = toc;

fprintf('\nTiempo total: %.3f s\n', tot_t);


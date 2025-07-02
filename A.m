n     = 1e5;
w_vec = linspace(0,1,10)';
t_vec = 2*w_vec - 1;              
f1    = n*(t_vec - 1).^2;
f2    = n*(t_vec + 1).^2;

plot(f1,f2,'o-');  grid on;
xlabel('f_1'); ylabel('f_2');


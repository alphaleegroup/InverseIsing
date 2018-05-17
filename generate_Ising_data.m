function generate_Ising_data
 
n = 50;
numMC = n*10^7;
Jo = [];
ho = [];
CCo = [];
input_Jo = [];
input_ho = [];

parfor ii = 1:16

[JJ, h, CC, input_J, input_h] = monte_carlo_test(n,numMC);
Jo = [Jo ; JJ];
ho = [ho ; h];
CCo = [CCo ; CC];
input_Jo = [input_Jo; input_J];
input_ho = [input_ho; input_h];

end

save('InverseIsingdatav2.mat','Jo','ho','CCo','input_Jo','input_ho')

end


function [JJ, h, CC, input_J, input_h] = monte_carlo_test(n,numMC)

T = 1; 

randJ = 2*rand+1;  %change the magnitude of J randomly 
randh = 2*rand+1/2;  %change the magnitude of h randomly 

J = randn(n,n);
J = - (J+J')/2;
J = J - diag(diag(J));
J = randJ*J/sqrt(n); %change the temperature and field more drastically 
h = randh*randn(n,1);

x0 = 2*round(rand(1,n))-1; 

kk = 1; 

data = zeros(numMC/(10*n),n); 

for ii = 1:numMC 
  
    e0 = -x0*(J*x0')/2 - x0*h; 
    
    ind = randi([1 n],1,1); 
    x = x0; 
    x(ind) = - x(ind); 
    
    e1 = -x*(J*x')/2 - x*h; 
   

    if e1<e0 
         
         x0 = x; 
     
     elseif exp(-(e1-e0)/T)>rand
         
        x0 = x; 
         
    end 
     
    
    if mod(ii,10*n) == 0 
        
        data(kk,:) = x0; 
        kk = kk + 1; 
    end 
    
end 
mu = mean(data,1); 
data_mu = data - repmat(mu,[size(data,1) 1]); 
C = data_mu'*data_mu/size(data_mu,1); 
Jmf = -inv(C);

%now do some ML trick 
B = triu(ones(n,n), 1); 
parCor = Jmf(B==1); 

TotCor = C(B==1); 

mux = repmat(mu,[n,1]); 
muy = repmat(mu',[1,n]);

mux = mux(B==1); 
muy = muy(B==1); 
JJ = J(B==1);

input_J= [parCor TotCor mux muy];
input_h = [mu' diag(Jmf) diag(C) (Jmf - diag(diag(Jmf)))*mu' (C - diag(diag(C)))*mu']; 
CC = C(:); 

end 
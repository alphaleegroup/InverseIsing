%Inputs
% n = number of variables 
% data = data matrix where each row is a sample 

function [J,h] = pseudolikihood(data,n)
    J = zeros(n,n); 
    for ii = 1:n 

        a = data; 
        sigma =  a(:,ii); 
        sigma = (sigma+1)/2+1; 
        a(:,ii) = [];  
        [B, FitInfo] = lassoglm(a ,sigma-1,'binomial','alpha',1e-10,'Lambda',1e-10); 
        h(ii) = FitInfo.Intercept/2; 
        j = B(1:end)'/2;
        
        if ii == 1
            J(ii,:) =[0 j];
        else
            J(ii,:) =[j(1:(ii-1)) 0 j(ii:end)];
        end 
    end 
    J = (J + J')/2; 

end 

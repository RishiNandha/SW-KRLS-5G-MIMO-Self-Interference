clear
load('data.mat')

x = TX;
y = RX;
M = 50;
y_pred = zeros(1,M-1);
e = zeros(1,M-1);

E = 0.1;

P = eye(M)/(1 + E);

for n = M:1000
    G = P(2:M,2:M) - (P(2:M,1)*P(1,2:M))/P(1,1);
    k = kappa(x, n-M+2, n, n-M+1, n-M+1);
    a = G*k;
    g = 1/(kappa(x,n,n,n,n)+E-(k')*a);
    P = [G + a*(a').*g , -a.*g ; -(a').*g , g]
    alpha = P*(y(n-M+1:n))';
    y_pred = [y_pred, ((kappa(x,n-M+1,n,n-M+1,n-M+1))')*alpha];
end

function result = kappa(x,r_start, r_end ,c_start, c_end)
    result = zeros(r_end-r_start+1, c_end-c_start+1);
    for r = r_start:r_end
        for c = c_start:c_end
            result(r-r_start+1,c-c_start+1) = exp((-(x(r)-x(c))^2)/2);
        end
    end
end
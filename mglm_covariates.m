function [stats df] = mglm_covariates(Y, Z, X, opt)

% happens at every single edge.  Will have to loop through every single edge.  

% Function mglm_covariates controls for covariates in multivariate general
% linear model. We can choose to use Wilk's Lambda or Roy's maximum root 
% to compute F-stat and corresponding p-value. 
%
% Multivariate Linear Model 
%     Full model: Y = BX + GZ + e   % Y is WaCS, X is group (0,1), Z is variable to correct for (e.g. age, sex)
%     Reduced model: Y = GZ + e 
%
% Input:
%     Y: response multivariate variable
%     X: model matrix, dimension
%     Z: covariates, dimension
%     
% Output:
%     stats.F: F-stats
%     stats.p: p-value
%
% Won Hwa Kim - wonhwa@cs.wisc.edu
% Updated: Apr. 23, 2013
%

if nargin == 3
    opt = 1;
end

n=size(Y,1);
k=size(Z,2)+1; % Constant
p=size(X,2);

Z2 = [ones(n,1) Z];

est = inv(Z2'*Z2)*Z2'*Y;   % reduced model
inter = est(1,:);
eG0 = est(2:k,:);

E0 = (Y - repmat(inter,n,1) - Z*eG0)'*(Y - repmat(inter,n,1) - Z*eG0);   % error of estimation @ 34

est = inv([Z2 X]'*[Z2 X])*[Z2 X]'*Y;   % full model
inter = est(1,:);
eG = est(2:k,:);
eB = est(k+1:end,:);

E = (Y - repmat(inter,n,1) - X*eB - Z*eG)'*(Y - repmat(inter,n,1) - X*eB - Z*eG);   % error of estimation of full model
H = E0-E;  % difference to state the errors. Low H means X isn't related to Y

%% Wilk's Lambda
if opt == 1;
    %% Wilk's lambda       % can get p value for group difference, accounting for other covariates
    WL = det(E) / det(E0);       % analog of H term above
    
    % F-stat and p-value approximation 
    m = size(E0,1); 
    s = p; 
    r = n-k-1 - (m-s+1)/2;
    u = (m*s-2)/4;

    if m^2 + s^2 - 5 > 0
        t = sqrt((m^2 * s^2 - 4) / (m^2 + s^2 - 5));
    else
        t = 1;
    end

    df = [r*t-2*u, m*s];

    F = ((1-WL^(1/t))/ WL^(1/t)) * (df(1)/df(2));   % F stat
    pval = 1-fcdf(F,df(2),df(1));                   % for F test

    %% Output
    stats.F = F;
    stats.p = pval;                             
  

%% Roy's maximum root                               % almost the same as Wilk's lambda, option 2 for F stat.  
elseif  opt == 2   
    eval = eig(pinv(E)*H);
    
    df = [size(Y,1)-size(Y,2)-(size([Z2 X],2)-1), size(Y,2)];    
    max_root = max(eval);

    F = max_root * df(1)/df(2);
    pval = 1-fcdf(F, df(2), df(1));
    stats.F = F;
    stats.p = pval;
end


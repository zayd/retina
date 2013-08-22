function [L rate rng Hxr] = calcL_LN(Model, Data, params, MC)
% function [L rate rng Hxr] = calcL_LN(Model, Data, params)
%
%  L:   cost func (-entr + lambda*nspikes)
%  r:   mean rates
%  rng: range of projected values w_j^T* x
%  Hxr: sample expected H(x|r) in bits


[I J] = size(Model.W);
N = size(Data.x,2);
[cnx cny lambda] = deal(params.sigmanx^2, params.sigmany^2, params.lambda);


cyI = cny*eye(I);
eJ = eye(J); cyJ = cny*eJ;
  

% y = W'*x
y = Model.W'*Data.x;

% z = f(W'*x)
% G = f'(W'*x)
[G unused1 unused2 z] = evalg(y, exp(Model.gc), Model.gmu, Model.gsigmas);


rate = mean(z,2);

if J<I, % neural space,
  WCW = Model.W' * Data.Cx * Model.W;
  if cnx, WW = Model.W'*Model.W; end;
end;


L = 0; 

for n=1:N,

  if J >= I, % data space

    WGD = Model.W * diag(G(:,n));
    WGDDGW = WGD*WGD';

    if cnx==0,    L = L + 0.5 * sum(log(eig(Data.iCx + WGDDGW / cny)));
    else,         L = L + 0.5 * sum(log(eig(Data.iCx + WGDDGW * inv(cnx*WGDDGW + cyI))));
    end;
  
  else,    % neural space 

    GG = G(:,n) * G(:,n)';
    DGWCWGD = GG .* WCW;
    if cnx, DGWWGD = GG .* WW;       end;
    if cnx==0,       L = L + 0.5 * sum(log(eig(eJ + DGWCWGD / cny)));
    else,            L = L + 0.5 * sum(log(eig(eJ + DGWCWGD * inv(cnx*DGWWGD + cyJ))));
    end;
    
  end;
end;

if J<I,    %neural space
  if ~isfield(Data,'logdetC'),Data.logdetC = sum(log(eig(Data.Cx))); end;
  L = L - 0.5*Data.logdetC*N; 
end;

% to maximize:
% L =   - E_p(y)[H(x|y)]      - lambda * sum(firing rate)
%  =  -1/2 ln(det(C)) - D/2ln(2pi) - D/2  (in nats)
%  =  [1/2 ln(det(C^-1)) - D/2ln(2pi) - D/2] /ln(2)   (in bits)
L = (L/N - I/2*log(2*pi) - I/2)/log(2) - sum(lambda.*rate);



if nargout > 2, rng  = [min(y,[],2) max(y,[],2)]; end;

% *negative* conditional entropy
% -H(x|r) = -E_p(y)[H(x|y)]
%        = -1/2 log(det(C)) + D/2 log(2*pi) + D/2
if nargout > 3, Hxr = L + sum(lambda.*rate); end;

function [g h ixj f] = kern_evalg(x, gc, gmu, sigma)
% function [g h ixj f] = kern_evalg(x, gc, gmu, sigma)
% 
%   gc:  J x Q matrix of kernel coefficients
%   gmu: J x Q matrix of kernel centers
%   x:   J x N projection values for N data samples

[J Q] = size(gc);
N     = size(x,2);

f = zeros(J,N); 
g = zeros(J,N); % g = f'
h = zeros(J,N); % h = g'

if nargin < 4, % 2*sigma = spacing
  sigma = 0.5*diff(gmu(:,1:2),[],2);
  if length(unique(sigma))==1, sigma = sigma(1); end;
end;


minx = repmat(min(gmu,[],2),[1 N]); maxx = repmat(max(gmu,[],2),[1 N]);
ix = round(Q*(x - minx)./(maxx-minx)); 
ix(ix<3) = 3; ix(ix>Q-2) = Q-2;

% collect the cumulatives up to (not including) the beginning of the 5 ixs
% centered on sample
fc = [zeros(J,1) cumsum(gc,2)];
% take the ix-3 element, scale by max of the integral of kernel sqrt(2pi)
fc = sqrt(2*pi) * fc(sub2ind([J Q], repmat([1:J]',[1 N]), ix-2));

ix = reshape([ix-2 ix-1 ix ix+1 ix+2],[J N 5]);
ixj = sub2ind([J Q], repmat([1:J]',[1 N 5]), ix);
gc  = gc(ixj);
gmu = gmu(ixj);

if isscalar(sigma),
  D = (repmat(x,[1 1 5]) - gmu)/sigma;
else,
  D = (repmat(x,[1 1 5]) - gmu)./repmat(sigma,[1 N 5]);
end;

gceD2 = gc .* exp(-D.^2/2);
g     = sum(gceD2,3);  % sum over the 5 

if nargout > 1,   
  h = sum(-gceD2.*D,3);
  if isscalar(sigma), h = h/sigma; 
  else,               h = h./repmat(sigma,[1 N]);
  end;
end;

if nargout > 3,   
  % f on account of 5 elements centered on sample
  f = fc + sqrt(pi)/sqrt(2)*sum(gc.*erfc(-D/sqrt(2)),3);  
  if isscalar(sigma),    f = f*sigma;
  else,                  f = f.*repmat(sigma,[1 N]);
  end;
end;

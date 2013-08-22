function cmap = cjet(N)
% returns a custom colormap, N by 3;
if nargin < 1,
  N = 256;
end;

% jet with grey in middle
c = repmat((1-exp(-abs(((1:N)-(N+1)/2)/40).^2))',1,3); 
c = (c - min(c(:))); c = c/max(c(:));
cmap = (1-c).*gray(N) + c.*jet(N);  

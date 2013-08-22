function Data = getData(Data, params);
% function Data = toy/getData(Data, params);

[I N dtype] = deal(params.I, params.N, params.dtype);

if isempty(Data) | ~isfield(Data,'Cx'),
  Data.Cx = eye(2);
  Data.iCx = eye(2);
end;
  

if strcmp(dtype,'gaussian'), 
  Data.x = randn(I,N);             % get Gaussian data
elseif strcmp(dtype,'radsym');
  Data.x = randn(I,10*N);             % get Gaussian data
  r      = sqrt(sum(Data.x.^2,1));    % and their radii

  t = linspace(0,25,1000);             % scale should not matter, will renormalize x
  p = t.*exp(-(t*sqrt(gamma(3/params.q)/gamma(1/params.q))).^params.q);

  r_new  = histoMatch(r, p, t);

  r_new = r_new(1:N); 
  r = r(1:N); 
  Data.x = Data.x(:,1:N);

  Data.x = Data.x.*(ones(I,1)*(r_new./r)); % put new radius
  Data.x = Data.x/sqrt(var(Data.x(:)));    % normalize variance
end

% inject sensory noise
if params.sigmanx,  
  Data.x = Data.x + sqrt(params.sigmanx)*randn(size(Data.x)); 
end;

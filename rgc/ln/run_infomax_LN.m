function [Model, Hist, params] = run_infomax_LN(params, Model_init, Hist_init)
% function [Model Hist] = toy/run_infomax_LN(params, Model_init, Hist_init)
%
% Linear filters - Nonlineary (pointwise) 
%
%   r = f(W'*(x+n_x)) + n_y     (n_x, n_y: noise)
%
%  goal: maximize MI(x;r)


params,

if isfield(params, 'randseed'), randn('seed',params.randseed); rand('seed',params.randseed); 
else                           randn('seed',sum(100*clock)); rand('seed',sum(100*clock)); 
end;


if ~params.dispfreq, fprintf('      '); end;

Data = getData([], params); 

[Model, Hist] = initModel(Data, params); 

if nargin > 1, Model = Model_init; end; 
if nargin > 2, Hist  = Hist_init; end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main loop
for t=params.startiter:params.iters,

  [epsW, epsg] = taperRates(params, t);
    
  % get data
  if ~rem(t, params.sampfreq), 
    Data = getData(Data, params);
  end;
  [dW, dg] = calcdL(Model, Data, params); 
  Model.W  = Model.W  + epsW * dW; 
  Model.gc = Model.gc + epsg * dg;

  % normalize W
  Model.W = Model.W*diag(1./sqrt(sum(Model.W.^2)));
  
  % update display 
  if (params.dispfreq && ~rem(t,params.dispfreq)) || (~params.dispfreq && ~rem(t,500)),

    [L, r, rng] = calcL_LN(Model, Data, params); 
    Hist = updateHist(Hist, Model, t, L, r, rng);

    if params.dispfreq,
      drawDisplay(Model, Data, Hist, params);
    else
      fprintf('\riter %d/%d', t, params.iters);
    end;
  end;
 
  if params.sumr && t>500 && ~rem(t,50), % adjust lambda to 
    if isscalar(params.lambda), % total rate
      params.lambda = params.lambda * (sum(Hist.r(:,end))/params.sumr)^.008;
    else,                       % individual rate
      params.lambda = params.lambda.* (Hist.r(:,end)/(params.sumr/params.J)).^.008;
    end;
  end;
  

  % save state file
  if params.savefreq && ~rem(t, params.savefreq),
    params.startiter = t+1;
    eval(sprintf('save %s Model Hist params', params.savefname));
  end;

end;

% done; save.
if params.savefreq,
  params.startiter = t; eval(sprintf('save %s Model Hist params', params.savefname));
end;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [dW, dg] = calcdL(Model, Data, params)
[I, J, N, Q] = deal(params.I, params.J, params.N, params.Q);
[cnx, cny, lambda] = deal(params.sigmanx^2, params.sigmany^2, params.lambda);

y = Model.W'*Data.x;

[G, gppWx, ixs] = evalg(y, exp(Model.gc), Model.gmu, Model.gsigmas);
[ii, jj] = ind2sub([J Q], ixs(:,:,3));

dW = zeros(I,J);   
dg = zeros(J,Q);  
dWn = zeros(J,N);  

eI = eye(I); cnyI = cny*eI;
eJ = eye(J); cnyJ = cny*eJ;

if J<I && cnx==0,
  CW  = Data.Cx*Model.W;  WCW = Model.W'*CW;
end;

onesQJ = ones(J,1)*(1:Q);
onesQ  = ones(1,Q);

for n=1:N,

  Gn  = diag(G(:,n));
  WG = Model.W*Gn;

  if J>=I, % data space,
    
    if cnx==0,
      %QR = inv(cny*Data.iCx + WG*WG') * WG;
      QR = (cny*Data.iCx + WG*WG')\  WG;
    else,
      WGGW = WG*WG';
      Q = inv(cnx*WGGW + cnyI);
      WGGWQ = WGGW*Q;
      %QR = Q * inv(Data.iCx + WGGWQ) * (eI - cnx*WGGWQ') * WG;
      QR = Q * ((Data.iCx + WGGWQ)\ (eI - cnx*WGGWQ')) * WG;
    end;
  else, % J<I, neural space
    
    if cnx==0,
      %QR     = CW*Gn * inv(cnyJ + (G(:,n)*G(:,n)') .* WCW);
      QR     = (CW*Gn) / (cnyJ + (G(:,n)*G(:,n)') .* WCW);
    else,
      %WGQ    = WG * inv(cnx*(WG'*WG) + cnyJ);
      WGQ    = WG /(cnx*(WG'*WG) + cnyJ);
      GWCWGQ = WG' * Data.Cx * WGQ;
      %WGQR   = WGQ * inv(eJ + GWCWGQ);
      WGQR   = WGQ /(eJ + GWCWGQ);
      QR     = Data.Cx * WGQR - cnx * WGQR * GWCWGQ;
    end;
  end;

  dgn1     = sum(Model.W .* QR,1)';
  dW   = dW + QR * Gn;
  dWn(:,n) = gppWx(:,n).*dgn1 - log(2)*lambda.* G(:,n);

  dg = dg - log(2)*sqrt(2)*sqrt(pi) * (onesQJ<= jj(:,n)*onesQ).*((lambda.*Model.gsigmas)*onesQ);
  for j=1:J, 
    dg(ixs(j,n,:)) = dg(ixs(j,n,:)) + dgn1(j)*exp(-((y(j,n) - Model.gmu(ixs(j,n,:)))/Model.gsigmas(j)).^2/2);
  end;
  
end;

dg = (exp(Model.gc)).*dg/N; 
dW = (dW + Data.x*dWn')/N;

if isfield(params,'eta') & params.eta, % decay W weights
  dW = dW - params.eta*sign(Model.W);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [Model Hist] = initModel(Data,params);
[I J Q] = deal(params.I, params.J, params.Q);

Model.gmu = 5*ones(J,1)*linspace(-1,1,Q);
Model.gsigmas = 0.5*diff(Model.gmu(:,1:2),1,2);
Model.gc = params.initgc*ones(J,Q);

% init and normalize |w|
Model.W = randn(I, J);  
Model.W = Model.W * diag(1./sqrt(sum(Model.W.^2)));

Hist.L = []; 
Hist.t = []; 
Hist.r = []; 
Hist.gc = [];

  



function Hist = updateHist(Hist, Model, t, L, r, rng);
% for history, take evenly spaced samples from gc around the central region
histix = reshape(1:prod(size(Model.gc)),size(Model.gc));
histix = histix(:,round(.4*size(Model.gc,2)):round(.6*size(Model.gc,2)));
histix = histix(round(linspace(1,prod(size(histix)),40)));
if length(Hist.t)==250,
  Hist.L  = [Hist.L(2:end)  L];
  Hist.t  = [Hist.t(2:end)  t];    
  Hist.r  = [Hist.r(:,2:end)  r];    
  Hist.gc = [Hist.gc(:,2:end) Model.gc(histix)']; 
else,
  Hist.L  = [Hist.L  L];    
  Hist.t  = [Hist.t  t];    
  Hist.r  = [Hist.r  r];    
  Hist.gc = [Hist.gc Model.gc(histix)']; 
end;

Hist.rng = rng; 

    



function [epsW epsg] = taperRates(params, t);
if  t < 5, % very start very slow
  epsW  = params.epsW/100; 
  epsg  = params.epsg/100;
elseif t < 50,  % start slow
  epsW  = params.epsW/4;
  epsg  = params.epsg/4;
elseif t > .9*params.iters, % taper end
  epsW  = params.epsW* (10*(params.iters-t)/params.iters);
  epsg  = params.epsg* (10*(params.iters-t)/params.iters);
else, 
  epsW = params.epsW;
  epsg = params.epsg;
end;

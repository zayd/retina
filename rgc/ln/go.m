
clear params

params.randseed = 2;

params.I = 2;  % input dim
params.J = 3;  % number of neurons
params.Q = 200; % number of bins in nonlinearity
params.N = 500; % batch size for training
params.dispfreq =1;
params.savefreq = 100;
params.sampfreq = 1; % data resample frequency

params.dtype = 'gaussian';
params.savefname = 'test';

% noise levels
params.sigmanx = 1e-2;
params.sigmany = 1e-2;
% metabolic cost
params.lambda = 1;
% target rate (0 = no adjustment to lambda)
params.sumr = 0; 
% initial slope of nonlinearity = exp(initgc)
params.initgc = -1;

params.startiter = 1;
params.iters = 100000;

% step size
params.epsW = .01;
params.epsg = 5;

[Model Hist params] = run_infomax_LN(params);

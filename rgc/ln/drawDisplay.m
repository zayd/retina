function drawDisplay(Model, Data, Hist, params)
% function drawDisplay(Model, Data, Hist, params)

[I, J, ~, N] = deal(params.I, params.J, params.Q, params.N);


mycol = hsv(J); 
mycol = [mycol(1:2:end,:); mycol(2:2:end,:)];

clf; 
whitebg(gcf,[0 0 0]); colormap([gray(128); cjet(128)]); 
set(gcf,'menubar','none','color',.3*[1 1 1],'inverthardcopy','off','name',params.savefname(1:end-4));


[g, ~, ~, r] = evalg(Model.gmu, exp(Model.gc), Model.gmu, Model.gsigmas);


% plot 1: filters W
subplot(2,3,1);

if I==2,
  hold on;
  prange = 2;
  Cx = Data.Cx;
  [V, D]=eig(Cx); ang = atan2(V(2,1), V(1,1)); 
  D = prange*D;
  h = ellipse(D(1,1), D(2,2), ang, 0,0); set(h,'color',[.5 .5 .5]); 
  hold on;
  for j=1:J, plot(prange*[0 Model.W(1,j)],prange*[0 Model.W(2,j)],'color',mycol(j,:),'linewidth',3); end;

  
  % place down fine grid of points
  [x y] = meshgrid(prange*linspace(-1,1,400)); 
  tol = diff(y(1:2))/sqrt(2);

  badix = find(sqrt(x.^2+y.^2)>prange); x(badix)=[]; y(badix)=[];
  [~, ~, ~, rm]=evalg(Model.W'*[x(:) y(:)]', exp(Model.gc), Model.gmu, Model.gsigmas); 
  % normalized response level
  for j=1:J,
    kk = linspace(0,max(rm(j,:)), 5+1); 
    A = null(Model.W(:,j)');
    for k=2:(length(kk)-1),
      % find (x,y) pairs where response rm is within tol of kk(k)
      ixj = find(abs(rm(j,:) - kk(k)) < tol);
      xk = x(ixj); yk = y(ixj);
      [aa, bb] = sort(A'*[xk; yk]); % sort them by null space
      if ~isempty(aa),
      	plot(xk(bb),yk(bb),'color',mycol(j,:));
	
      else
	end;
    end;
  end;
  
  for j=1:0,%J,
    % iso resp lines
    fj = f(j,:); % nonlinearity f_j(.)
    
    isos = []; %ixs = [];  
    xix = find(fj > 1e-3*rf);
    if ~isempty(xix), isos = Model.gmu(j,xix(1)); ixs = xix(1); end;
    while max(fj) > rf,
      xix = find(fj > rf); 
      isos = [isos Model.gmu(j,xix(1))]; % add ticks
      fj = fj - rf;
    end;
    A = null(Model.W(:,j)');
    t = linspace(-1,1,10);
    for kk=1:length(isos), % for each line
      
      Ak = (1-isos(kk)^2); Ak=Ak*(Ak>0); % width of isoresp curves (within the circle)
      Ak = sqrt(Ak)*A;
      sW = isos(kk)*Model.W; % center of iso response (tick)
      plot([sW(1,j)-Ak(1) sW(1,j)+Ak(1)], [sW(2,j)-Ak(2) sW(2,j)+Ak(2)], 'color',mycol(j,:));
    end;

  end;
  axis(1.1*max(diag(D))*[-1 1 -1 1]); axis square; %axis equal; 
  
  
end;



% plot 2: f()
subplot(2,3,2); hold on;
for j=1:J, plot(Model.gmu(j,:), r(j,:),'color',mycol(j,:)); end;
title('f(x)'); set(gca,'xlim',[-2 2]);

% plot 5: g()
subplot(2,3,5); 
hold on;
for j=1:J, plot(Model.gmu(j,:), g(j,:),'color',mycol(j,:)); end;
title('g(x)');

% plot 3: g()   
axes('pos',[.6867  .54 .2933 .42]); hold on;

if isfield(Hist,'gc'),
  plot(Hist.t, Hist.gc'); title('log(c)');
else
  sg = sign(Model.W'*[1;zeros(params.I-1,1)]);
  for j=1:J, plot(sg(j)*Model.gmu(j,:), f(j,:),'color',mycol(j,:),'linewidth',2);    end;
  axis([min(vec(Model.gmu)) max(vec(Model.gmu)) 0 max(f(:))]); title('f');
end;
set(gca,'xticklabel', []);



% plot 6: L, r  subp(4,3,9,.04);
axes('pos',[.6867 .27+.02 .2933 .21]);
plot(Hist.t, Hist.L,'y.','markersize',2); 
tl = min([length(Hist.L) 20]);
II = mean(Hist.L(end-tl+1:end)) + sum(params.lambda.*mean(Hist.r(:,end-tl+1:end),2),1);
title(sprintf('[%d]  L: %.3f   <r>: %.3f  Inf: %.3f',Hist.t(end), mean(Hist.L), mean(Hist.r(:)), II));
set(gca,'xticklabel',[]);


% plot 7: firing rates, %subp(4,3,12,.04); hold on;
axes('pos',[.6867 .02+.02 .2933 .21]); hold on;
if isfield(params,'sumr') & params.sumr, plot(Hist.t([1 end]), params.sumr/J*ones(1,2),'w'); end;
if J<20,
  for j=1:J,
    plot(Hist.t, Hist.r(j,:),'color',mycol(j,:));
  end;
else
  [~, b2]=min(mean(Hist.r,2));
  [~, b3]=max(mean(Hist.r,2));
  plot(Hist.t, Hist.r(b2,:),   'color',[.3 .3 .6]);
  plot(Hist.t, Hist.r(b3,:),   'color',[.6 .3 .3]);
  plot(Hist.t, mean(Hist.r),'color',[1 .3 .3]);
end
title('firing rates (min/mean/max), target rate (w)');
%drawnow;
pause(0);
drawnow;


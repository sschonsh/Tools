%% Load Surface
[surf.pts,surf.trg] = ReadOFF('Data/Kitten.off');

%% Compute Euclidan FPS
npts1 = 100;
npts2 = 1000;

pts2 = surf.pts(DefineSubsample(surf.pts,npts1),:);
pts3 = surf.pts(DefineSubsample(surf.pts,npts2),:);

%% Plot
figure
subplot(1,3,1)
ViewMesh(surf.pts,surf.trg)
title('Original Mesh')
subplot(1,3,2)
ViewPC(pts2)
title([num2str(npts1) ' Points'])
subplot(1,3,3)
ViewPC(pts3)
title([num2str(npts2) ' Points'])
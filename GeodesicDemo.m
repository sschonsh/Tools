%% Load Surface
[surf.pts,surf.trg] = ReadOFF('Data/Kitten.off');

%% Create two Correspondence Maps
IDX1 = 1:length(surf.pts);
IDX2 = IDX1;
IDX1(1:500) = IDX1(randperm(500));
IDX2(1:100) = IDX1(randperm(100)+500);

%% Check Geodeisc Errors and Plot
%Compute Geodesic Radius
[ExaustiveR, ApproxR] = computeGeodesicRadius(surf,0);

%Comute GeoErrors-Normalized by Surface Area
geoError1 = calcGeoError(surf,IDX1);
geoError2 = calcGeoError(surf,IDX2);

%Compute Geodesic Error-Normalized Radius
geoError3 = calcGeoErrorUnitRad(IDX1,surf,ExaustiveR);
geoError4 = calcGeoErrorUnitRad(IDX2,surf,ExaustiveR);

%Plot Princeton Benchmarks
figure
subplot(1,2,1)
plotGeoError(geoError1,0)
hold on
plotGeoError(geoError2,0)
title('Unit Surface Area')
subplot(1,2,2)
plotGeoError(geoError3,0)
hold on
plotGeoError(geoError4,0)
title('Unit Radius')


%Plot Errors on Surf
figure
subplot(1,2,1)
ViewMesh(surf.pts,surf.trg,geoError1);
title('Geodesic Errors IDX1');
subplot(1,2,2)
ViewMesh(surf.pts,surf.trg,geoError2);
title('Geodesic Errors IDX2');


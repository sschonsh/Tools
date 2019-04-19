function [geoError] = calcGeoErrorUnitRad(IDX,surf,Rad);

%% Load Surfs
pts = surf.pts;
trg = surf.trg;

%% Calculate Geodesic Errors
geoError = zeros(length(pts),1);
%map1to2 = IDX(:,1);
for i = 1:length(pts)
    %select points  
    start_point        = IDX(i);
    options.end_points = i; 
    %calcuale distance
    if start_point == options.end_points
        geoError(i) = 0;
    else
        [D,~,~]  = perform_fast_marching_mesh(pts', trg', start_point, options);
        geoError(i) = D(i);
    end
end
geoError = geoError /(Rad);%normalize geodesic errors
fprintf('Done \n')

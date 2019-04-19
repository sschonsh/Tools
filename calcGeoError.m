function [geoError] = calcGeoError(surf,map1to2)

fprintf('Computing Geodesic Errors...')

%Load data
pts = surf.pts;
trg = surf.trg;

%get Mass
[~, ~, M,~] = LBOeigs(pts,trg,1);



%% Calculate Geodesic Errors
geoError = zeros(length(pts),1);
%map1to2 = IDX(:,1);
for i = 1:length(pts)
    %select points  
    start_point        = map1to2(i);
    options.end_points = i; 
    %calcuale distance
    if start_point == options.end_points
        geoError(i) = 0;
    else
        [D,~,~]  = perform_fast_marching_mesh(pts', trg', start_point, options);
        geoError(i) = D(i);
    end
end
Area = sqrt(sum(M)); %sqrt(sum(w.^2'*M2));
geoError = geoError /(Area);%normalize geodesic errors
fprintf('Done \n')

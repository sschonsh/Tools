function [maxRad,tempMax] = computeGeodesicRadius(surf,ind)

pts = surf.pts;
trg = surf.trg;
maxRad = 0;
tempmax = 0;

%% Check All
if ind ~= 2
    Distances = zeros(length(pts));
    for i = 1:length(pts)
        [D,~,~]  = perform_fast_marching_mesh(pts, trg, i);
        Distances(:,i) = D;
    end
    maxRad = max(max(Distances));
end

%% Iterative
if ind ~= 1
    pt = randi(length(surf.pts));
    for i = 1:length(surf.pts)
        [D,~,~]  = perform_fast_marching_mesh(pts, trg, pt);
        [tempMaxes(i),pt] = max(D);
        if i > i && tempMaxes(i) <= max(tempMaxes(1:i-1))
            break
        end
    end
end
tempMax = max(tempMaxes);

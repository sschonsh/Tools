function [] = plotGeoError(geoError, figures)

%Calculate erros for plots.
h        = 0.00125;
graph    = zeros(100,1);
errorAx  = 0:h:0.25-h;
numPts = length(geoError)
for i = 1:length(errorAx)
    graph(i) = length(find(geoError < errorAx(i)))/numPts*100;
end

%error plot
if figures == 1
figure
end
plot(errorAx,graph,'LineWidth',2)
grid on
xlabel('Geodessic Error')
ylabel('Percent of points with error < e')
title('Geodesic Errors')
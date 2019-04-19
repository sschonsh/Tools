function [DownPts] = DefineSubsample(pts,samplesize)
%FPS Downsampling

distMat = squareform(pdist(pts));
DownPts = zeros(samplesize,1);
DownPts(1) = 1;
distVec = distMat(:,1);
idx = 1;

for i=1:samplesize-1
    distVec = min(distVec, distMat(:,idx)); 
    [~, idx] = max(distVec);
    DownPts(i+1) = idx;
end

function [eigFun, eigVal, MassV, StiffM] = LBOeigs(pts,trg,numEigs)
%eig Val is vector, MassV is diag of fullMass

% numpt = size(pts,1);
% numtrg = size(trg,1);
% 
% %% Loop through trigs find areas
% area = zeros(numpt,1);
% for i=1:length(trg)
%     p1 = trg(i,1);   p2 = trg(i,2);   p3 = trg(i,3);
%     v1 = pts(p1,:);   v2 = pts(p2,:);   v3 = pts(p3,:);
%     trgarea = 0.5*norm(cross(v2-v1,v3-v1));
%     area(p1) = area(p1)+ trgarea;
%     area(p2) = area(p2)+ trgarea;
%     area(p3) = area(p3)+ trgarea;
%     
% end
% 
% %%  Creat the LB Matrix by Desburn discretization
% diagMass = 0.5*ones(numpt,1);
% Stiff    = sparse(numpt,numpt);
% fullMass = Stiff;
% for i=1:numtrg
%     %local cords
%     p1 = trg(i,1);   p2 = trg(i,2);   p3 = trg(i,3);
%     v1 = pts(p1,:);   v2 = pts(p2,:); v3 = pts(p3,:);
%     
%     S2 = (v3-v1);
%     S3 = (v2-v1);
%     A  = .5*norm(cross(S3,S2));
%     
%     %angles
%     cot1 = dot(v2-v1,v3-v1)/norm(cross(v2-v1,v3-v1));
%     cot2 = dot(v1-v2,v3-v2)/norm(cross(v1-v2,v3-v2));
%     cot3 = dot(v1-v3,v2-v3)/norm(cross(v1-v3,v2-v3));
%     
%     %count all off diag entries
%     Stiff(p1,p2) = Stiff(p1,p2) + diagMass(p1)*(cot3);
%     Stiff(p1,p3) = Stiff(p1,p3) + diagMass(p1)*(cot2);
%     
%     Stiff(p2,p1) = Stiff(p2,p1) + diagMass(p2)*(cot3);
%     Stiff(p2,p3) = Stiff(p2,p3) + diagMass(p2)*(cot1);
%     
%     Stiff(p3,p1) = Stiff(p3,p1) + diagMass(p3)*(cot2);
%     Stiff(p3,p2) = Stiff(p3,p2) + diagMass(p3)*(cot1);
%     
%     %Make fullMass
%     fullMass(p1,p1) = A/6+fullMass(p1,p1);
%     fullMass(p2,p2) = A/6+fullMass(p2,p2);
%     fullMass(p3,p3) = A/6+fullMass(p3,p3);
%     
%     fullMass(p1,p2) = A/12+fullMass(p1,p1);
%     fullMass(p2,p1) = fullMass(p1,p2);
%     
%     fullMass(p1,p3) = A/12+fullMass(p1,p3);
%     fullMass(p3,p1) = fullMass(p1,p3);
%     
%     fullMass(p2,p3) = A/12+fullMass(p2,p3);
%     fullMass(p3,p2) = fullMass(p2,p3);
%     
% end
% 
% Stiff = Stiff - spdiags(sum(Stiff,2),0,numpt,numpt);
% diagMass = spdiags(area/3,0,numpt,numpt); 
% 
% 
% %Define Mass and Stiffness for application\
% StiffM = -Stiff;
% MassV  = diag(diagMass); %only want diag Mass
Surf.pt = pts;
Surf.trg = trg;
[StiffM, MassV] = FindStiffMass(Surf);
%fullMass = diag(MassV);

%Find and sort eigs
[eigFun, eigVal] = eigs(StiffM,diag(MassV),numEigs,'sm');
eigVal = diag(eigVal);
[~,ind] = sort(eigVal,'ascend');
eigVal = eigVal(ind);
eigFun = eigFun(:,ind);










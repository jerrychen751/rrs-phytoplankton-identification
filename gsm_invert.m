function IOPs = gsm_invert(rrs,aw,bbw,bbpstar,A,B,admstar,pnb);
% See: Kramer et al. (2022) for more details
% Note that aph parameterization is different from Maritorena et al. (2002)

options = optimset('fminsearch');
options = optimset(options,'TolX',1e-9,'TolFun',1e-9, 'MaxFunEvals', 2000, 'MaxIter', 2000);
IOPs=repmat(NaN,[size(rrs,1),3]);

IOPSinit=[.02,.01,.0029]; 
   %NOTE THAT THE CHL RETRIEVALS ARE ESPECIALLY SENSITIVE TO INITIAL GUESS!
   
for i = 1:size(rrs,1)
   rrs_obs = rrs(i,:);
   [iops,cost(i),exitFlag] = fminsearch('gsm_cost',IOPSinit, ...
					options,rrs_obs,aw,bbw,bbpstar,A,B,admstar); 

   if (exitFlag==1) % if converged then use value as IOP and inital guess
     IOPs(i,:)=iops; 
   end
end
return

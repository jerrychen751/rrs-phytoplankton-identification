function IOPs = gsm_invert(rrs,aw,bbw,bbpstar,aphstar,admstar,pnb);
% See: Maritorena et al. 2002 (GSM) for more details

options = optimset('fminsearch');
options = optimset(options,'TolX',1e-9,'TolFun',1e-9, 'MaxFunEvals', 2000, 'MaxIter', 2000);
IOPs=repmat(NaN,[size(rrs,1),3]);

IOPSinit=[.02,.01,.0029]; 
   %NOTE THAT THE CHL RETRIEVALS ARE ESPECIALLY SENSITIVE TO INITIAL GUESS!
   
for i = 1:size(rrs,1)
   rrs_obs = rrs(i,:);
   [iops,cost(i),exitFlag] = fminsearch('gsm_cost',IOPSinit, ...
					options,rrs_obs,aw,bbw,bbpstar,aphstar,admstar); 

   if (exitFlag==1) % if converged then use value as IOP and inital guess
     IOPs(i,:)=iops; 
   end
end
return
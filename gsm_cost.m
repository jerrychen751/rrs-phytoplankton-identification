function cost=gsm_cost(IOPs,rrs,aw,bbw,bbpstar,A,B,admstar)

g = [0.0949 0.0794];	%orig., constants in eq. 2 Gordon et al., 1988

aph = A.*IOPs(1).^B;
a = aw + aph + (IOPs(2)*admstar);
bb = bbw + IOPs(3)*bbpstar;
x = bb./(a + bb);
rrspred = (g(1) + g(2)*x).*x;
cost=sum((rrs-rrspred).^2);
return

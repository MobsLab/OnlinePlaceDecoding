function result = build_epanech_kernel(p1,p2,p3,p4,pos,Ha,Hx)

	syms a1 a2 a3 a4 x
	fun1=symfun(heaviside(a1-p1+1)*heaviside(p1+1-a1)*0.75*(1-(a1-p1)*(a1-p1)/Ha/Ha)/Ha, [a1]);
	fun2=symfun(heaviside(a2-p2+1)*heaviside(p2+1-a2)*0.75*(1-(a2-p2)*(a2-p2)/Ha/Ha)/Ha, [a2]);
	fun3=symfun(heaviside(a3-p3+1)*heaviside(p3+1-a3)*0.75*(1-(a3-p3)*(a3-p3)/Ha/Ha)/Ha, [a3]);
	fun4=symfun(heaviside(a4-p4+1)*heaviside(p4+1-a4)*0.75*(1-(a4-p4)*(a4-p4)/Ha/Ha)/Ha, [a4]);
	fun5=symfun(heaviside(x-pos+1)*heaviside(pos+1-x)*0.75*(1-(x-pos)*(x-pos)/Ha/Ha)/Ha, [x]);
	result=symfun(fun1(a1)*fun2(a2)*fun3(a3)*fun4(a4)*fun5(x), [a1,a2,a3,a4,x]);
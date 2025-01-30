function result = build_gauss_kernel(p1,p2,p3,p4,pos,Ha,Hx)

	syms a1 a2 a3 a4 x
	fun1=symfun(exp(0.5*(a1-p1)*(p1-a1)/Ha/Ha)/(Ha*sqrt(2*pi)), [a1]);
	fun2=symfun(exp(0.5*(a2-p2)*(p2-a2)/Ha/Ha)/(Ha*sqrt(2*pi)), [a2]);
	fun3=symfun(exp(0.5*(a3-p3)*(p3-a3)/Ha/Ha)/(Ha*sqrt(2*pi)), [a3]);
	fun4=symfun(exp(0.5*(a4-p4)*(p4-a4)/Ha/Ha)/(Ha*sqrt(2*pi)), [a4]);
	fun5=symfun(exp(0.5*(x-pos)*(pos-x)/Hx/Hx)/(Hx*sqrt(2*pi)), [x]);
	result=symfun(fun1(a1)*fun2(a2)*fun3(a3)*fun4(a4)*fun5(x), [a1,a2,a3,a4,x]);
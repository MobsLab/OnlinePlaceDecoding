function result = add_epanech_kernel(f,p1,p2,p3,p4,pos,Ha,Hx)

	syms a1 a2 a3 a4 x
	kernel=build_epanech_kernel(p1,p2,p3,p4,pos,Ha,Hx);
	result=symfun(f(a1,a2,a3,a4,x)+kernel(a1,a2,a3,a4,x), [a1,a2,a3,a4,x]);
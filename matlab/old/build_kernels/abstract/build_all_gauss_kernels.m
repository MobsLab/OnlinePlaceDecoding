function result=build_all_gauss_kernels(p1,p2,p3,p4,pos,Ha,Hx)

	if ~(size(p1,2)==size(p2,2)) || ~(size(p1,2)==size(p3,2)) || ~(size(p1,2)==size(p4,2)) || ~(size(p1,2)==size(pos,2))
		'sizes are not the same'
		result=false;
	elseif size(p1,2)==1
		result=build_gauss_kernel(p1(1),p2(1),p3(1),p4(1),pos(1),Ha,Hx);
	else
		result=add_gauss_kernel(build_all_gauss_kernels(p1(2:end),p2(2:end),p3(2:end),p4(2:end),pos(2:end),Ha,Hx),p1(1),p2(1),p3(1),p4(1),pos(1),Ha,Hx);
	end


tfb = ENCODED_DATA(3).twofirstbins;
X_step = tfb(5,2)-tfb(5,1);
Y_step = tfb(6,2)-tfb(6,1);


trueXerror = X_error * X_step;
trueYerror = Y_error * Y_step;
trueError  = sqrt(trueXerror.^2 + trueYerror.^2);
[bleuh tri] = sort(ecartT);
p=[];
for i=1:10
	p = [p mean(trueError(tri(1:i*end/10)))];
end

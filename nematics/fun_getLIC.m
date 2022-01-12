function LIC = fun_getLICs(nx,ny)

[W,L] = size(nx);
M = randn(W,L);

sigma = 30;
v = zeros(W,L,2);

v(:,:,2) = nx;%cos(Ang);% NOTE THAT COS GOES IN TO SECOND LAYER
v(:,:,1) = ny;%-sin(Ang);% NOTE THAT -SIN GOES IN TO SECOND LAYER

 % regularity of the vector field
options.bound = 'sym'; % boundary handling
options.histogram = [];
v = perform_vf_normalization(v);
% parameters for the LIC
options.histogram = 'linear'; % keep contrast fixed
options.spot_size = 3;
options.niter_lic = 9; % several iterations gives better results
options.M0 = M;
LIC = perform_lic(v, 7, options);

end
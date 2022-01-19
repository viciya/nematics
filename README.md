# Nematics codes

Examples of function use. The examples of the codes located in `.\nematics\_bacteria\get_defects_from_angle.m`

## visualize nematic field on top of raw image
	%% Load raw image corresponing to angular map
	filepath = '.\example_images\orient\Orient_1_X1.tif'
	Ang = imread(filepath);
	[f,name,ext] = fileparts(filepath);
	dir_info  = dir(['.\example_images\raw\*',name(8:end),'.tif']);
	raw_img_path = [dir_info.folder '\' dir_info.name];
	imshow(imread(raw_img_path)); hold on
	plot_nematic_field(Ang);

## order parameter
	%% Order parameter map from angular map
	filepath = '.\example_images\orient\Orient_1_X1.tif'
	Ang = imread(filepath);
	qq = order_parameter(Ang,10,3);
	imshow(qq); hold on


## defect detection
	%% Find defects from order angular map (plotted on of order parameter)
	filepath = '.\example_images\orient\Orient_1_X1.tif'
	Ang = imread(filepath);
	qq = order_parameter(Ang,10,3);
	imshow(qq); hold on
	[xf, yf] = detectDefectsFromAngle(Ang);
	scatter(xf, yf, "filled")


## [+1/2, -1/2] defect classification
	%% Classification of +1/2 and -1/2 defects
	filepath = '.\example_images\orient\Orient_1_X1.tif'
	Ang = imread(filepath);
	[f,name,ext] = fileparts(filepath);
	dir_info  = dir(['.\example_images\raw\*',name(8:end),'.tif']);
	raw_img_path = [dir_info.folder '\' dir_info.name];
	imshow(imread(raw_img_path)); hold on
	plot_nematic_field(Ang);

	% Display defects
	[ps_x, ps_y, plocPsi_vec, ns_x, ns_y, nlocPsi_vec] = ...
	    fun_get_pn_Defects_newDefectAngle(Ang);
	%         fun_get_pn_Defects_newDefectAngle_blockproc(Ang);

	plot_pdefect_ndefect(ps_x, ps_y, plocPsi_vec,...
	    ns_x, ns_y, nlocPsi_vec);
	%     fun_defectDraw(ps_x, ps_y, plocPsi_vec);



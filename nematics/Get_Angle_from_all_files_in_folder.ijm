//This code makes orientation maps of all images in folder

dir1 = getDirectory("Choose Source Directory ");
list = getFileList(dir1);
//Array.sort(list);
//print(list[0])
i=0;
while (i<list.length) {
//while (i<2) {
    print(dir1+list[i]);  
    open(dir1+list[i]);       
    getDimensions(width, height, channels, slices, frames); // get image dimensions
    print(slices);

    width1=floor(width/3); 
    height1=floor(height/3);
    // reduce image dimensions by 3
//    run("Size...", "width="+width1+" height="+height1+" depth="+slices*frames+" constrain average interpolation=Bilinear"); 

    //!!!!!!! F L I P !!!!!!!!!
//	run("Flip Horizontally", "stack");
	
	//wait(100);  
	run("OrientationJ Analysis", "log=0.0 tensor=10.0 gradient=0 orientation=on harris-index=on s-distribution=on hue=Orientation sat=Coherency bri=Original-Image ");
    selectWindow("OJ-Orientation-1");
    orientID=getImageID();
    //run("Save", "save="+dir1+"oldOrient_"+list[i]);
    run("Save", "save=["+dir1+"Orient_"+list[i]+"]");
	run("Close All");
    i = i+1; 
}

//myDir = getDirectory("Choose Source Directory ");
//print(myDir+"orient");
//File.makeDirectory(myDir+"orient");
//if (!File.exists(myDir))
//  exit("Unable to create directory");
//print("");
//print(myDir);


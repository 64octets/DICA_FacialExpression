function [parts,partsmatrix,faces] = getparts(twodinputimg)

reqToolboxes = {'Computer Vision System Toolbox', 'Image Processing Toolbox'};
if( ~checkToolboxes(reqToolboxes) )
 error('detectFaceParts requires: Computer Vision System Toolbox and Image Processing Toolbox. Please install these toolboxes.');
end

img = cat(3, twodinputimg, twodinputimg, twodinputimg);
detector = buildDetector();
[bbox bbimg faces bbfaces parts partsmatrix] = detectFaceParts(detector,img,2);


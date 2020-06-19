# OCR
Object Character Recognition API that receves 4 hands at a bridge table and serves a LIN/PBN./etc describing the hand

## Computer Vision Model
A CV model is crucial for enabling the OCR functionality. It should be able to identify existing card instances from 2D images and videos. The current choice is Yolo v4 which can be found here: https://github.com/AlexeyAB/darknet
In order to use the model follow the described procedures in the provided repository.
This procedure has already been completed and confugiration files as well as trained weights are available by request. 

### Dataset Generation
The computer vision model requires ground truth labelled 2D images in order to learn how to identify existing card instances. 
Using the provided python notebook a synthetic dataset can be generated.
For more accurate and generalizeable models a custom dataset with images captured in real use case situations has to be made. For each existing instance of interest in the images a bounding box has to be specified. This can be done using the following repository: https://github.com/tzutalin/labelImg

## OCR API
The role of the API is to make use of the trained model to provide a softaware solution capable of serving hand-describing files from 2D images. 

For additional information:
Computer Vision - Teodor Totev ( tedi.totev97@gmail.com ) 
API - Zhivko Draganov ( )

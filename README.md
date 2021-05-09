# OCR
The Object Character Recognition (OCR) project aims to detect playing cards from images/videos of the bridge table. It generates a synthetic dataset which is then used to train a computer vision model. The CV model is then integrated into an OCR API which detects cards from images and applies appropriate bridge table logic to reconstruct the bridge deals and serve them in widely used bridge formats.

## Computer Vision Model
The current computer vision model in use is Yolo v4 as provided by AlexeyAB: \
Yolo v4 Darknet: https://github.com/AlexeyAB/darknet

### Training Dataset
In order to train the Yolo v4 model a synthetic dataset was generated. The Jupyter Notebook and a Python label format conversion script can be found in: \
> datagen/

## Detection
The trained model can be used to detect images through a Python API located in: \
> detection/
It's current output is overlapping 720x720 snippets of the input images with snippet detections. These will soon be combined to form detections over the original input images.


### Synthetic Data Generation
Using the provided python notebook with the same name a synthetic dataset can be generated. \
Cards Images: https://drive.google.com/drive/folders/1bMF4GYIPJejTKQsYajixYM5RAhuyb5Ge?usp=sharing \
Synthethic Dataset: https://drive.google.com/drive/folders/1Vapg88kaifZkCv9o9GeGy8MCF9_BHD8w?usp=sharing (link not working)

## OCR API
The role of the API is to make use of the trained model to provide a software solution capable of serving hand-describing files from 2D images. 

For additional information:
Computer Vision - Teodor Totev ( tedi.totev97@gmail.com )

API - John Faben ( jdfaben@gmail.com)

Or message on the OCR channel on the online-bridge-hackathon discord 

=======
Suggested testing path:
1. Open Google Colab: https://colab.research.google.com
2. Create a new notebook
3. Mount your Google Drive using the commands:
> from google.colab import drive
> drive.mount('/content/drive')
4. Clone Yolo v4 Darknet repository: https://github.com/AlexeyAB/darknet
5. Navigate to the cloned darknet folder and edit the Makefile as follows:
> GPU=1
> CUDNN=1
> CUDNN_HALF=1
> OPENCV=1
> LIBSO=1
6. Run make in the darknet folder
> make
7. Upload detect.py to the main darknet folder. Upload any test images i.e. in darknet/data/test/ and outputs will be generated in darknet/data/test_out/. Some test images can be found here: https://drive.google.com/drive/folders/1KA7HseM2liyBhExUySehFMhzlANCs0wO?usp=sharing
8. Download the following darknet data files: https://drive.google.com/drive/folders/1rsNcTu3LQekErwQonqjSFaDLGiPg829A?usp=sharing \
Put yolov4-cards.cfg and cards.weights in the main darknet folder. \
Put cards.data and cards.names in darknet/data/.
9. In the notebook make sure you are in the main darknet folder and run:
> ! chmod +x ./darknet

> ! python detect.py -if data/test/ -of data/test_out/ -cfg yolov4-cards.cfg -df data/cards.data -w cards.weights \
If you have followed the instructions above closely this should run detection on the provided images. Otherwise make sure you change the provided paths to the appropriate files.

Contacts:
Teodor Totev ( tedi.totev97@gmail.com )
John Fabedn (jdfaben@gmail.com)

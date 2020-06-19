# OCR
An Object Character Recognition (OCR) API that receives one or more images of bridge hands on a table and serves a LIN/PBN./etc describing them.

## Computer Vision Model
A good computer vision (CV) model is crucial for enabling the OCR functionality. Ideally, it should be able to identify existing card instances from 2D images and/or videos. There are many frameworks that could be used. 
Currently we are exploring the Yolo v4 Darknet framework. The training of the model requires GPU access. 
Yolo v4 Darknet: https://github.com/AlexeyAB/darknet. In order to use the model follow the described procedures in the provided repository.
Yolo v4 Configuration file: https://drive.google.com/file/d/1yXCTUUaUzIpc6DjX4ObbFrrkQyrDcmre/view?usp=sharing
Yolo v4 Weights Pre-trained on COCO: https://drive.google.com/file/d/1IsDp30yaeAk5T9JUrfPnp_844Q7Syj4B/view?usp=sharing
Yolo v4 Weights Transfer Learned on the Synthethic dataset: https://drive.google.com/file/d/1-0V4grCW4FZAhWxcAB6HPveG7USf1rA5/view?usp=sharing
cards.data file: https://drive.google.com/file/d/1YBORw-uk8WDV0qJBMGupigEbER-Y38XE/view?usp=sharing
cards.names file: https://drive.google.com/file/d/1vQo2CDmZgAa6Jjie2zmsgeN88oMRiUHN/view?usp=sharing
train.txt file: https://drive.google.com/file/d/1RNBeSUjODZ6g_-kQW7AQ_tEKo6SFvQb7/view?usp=sharing
val.txt file: https://drive.google.com/file/d/1SMhjx2uM_nVtJ5ZIgOBt1uW4LSChOzc4/view?usp=sharing 

### Datasets
The computer vision model requires ground truth labelled 2D images in order to learn how to identify existing card instances. 

#### Labelling
A labelled 2D image should include bounding box coordinates for each of the existing card symbol instances and their corresponding category classification.
This is a laborious and manual work that can be accomplished with a special labelling tool. For use and installation instructions follow the ones provided in the following repository.
LabelImg tool: https://github.com/tzutalin/labelImg

### Synthetic Data Generation
Using the provided python notebook with the same name a synthetic dataset can be generated.
Cards Images: https://drive.google.com/drive/folders/1bMF4GYIPJejTKQsYajixYM5RAhuyb5Ge?usp=sharing
Synthethic Dataset: https://drive.google.com/drive/folders/1Vapg88kaifZkCv9o9GeGy8MCF9_BHD8w?usp=sharing

## OCR API
The role of the API is to make use of the trained model to provide a softaware solution capable of serving hand-describing files from 2D images. 

For additional information:
Computer Vision - Teodor Totev ( tedi.totev97@gmail.com ), 
API - Zhivko Draganov ( ) or message in the OCR channel in Discord


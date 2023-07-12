This repo contains all the codes which we used for the preprocessing of the spinal cord Ultrasound B-Mode Sagittal images. 

The directory DataPreprocessing contains the codes used to preprocess and format the dicom files as per the Yolov8 model requirements.

The directory DataScaling contains the codes required to scale all the images. 

The directory HematomaDetectionYolov8 contains all the codes to train and predict the hematoma. The dataset needs to be divided into three parts - train, test and val to finetune the pretrained YoloV8 model on our custom dataset.

The requirements.txt file contains all the python libraries used in the codes.
# EECS598_SelfDriving_PerceptionProject

To generate a csv file with training and validation data with pathnames suitable 
to the OS/system that you're using, just run 'modify_csv.py' 
before you start training. 
Make sure you save /deploy , labels.csv in the /model folder.

AlexNet:
To finetune Alexnet do:
python finetuning_alexnet.py

To test the trained model on testing data:
python inference.py

Alexnet has been implemented in alexnet.py
The script can be used to modify any specifications

VGG-16:
To Train the dataset using VGG-16 do:
python finetuning_vgg.py

To test the trained model on testing data:
python inference.py

VGG-16 has been implemented in vgg_16.py
The script can be used to modify any specifications

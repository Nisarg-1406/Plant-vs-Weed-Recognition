# Plant-vs-Weed
Object Detection Classifying Between Plant And Weed

Computer vision task:
This Project focuses on Classifying between Multiple Plants and Weeds Using Single Shot Multibox detector(SSD)

SSD is one of the object detection Module.

Model name used is **ssd_mobilenet_v2**

First of all, taken the Images and divided into 80 : 20 Train vs test images. 
With this images, I have generated the XML files and converted train folder annotation xml files to a single **csv** file. Same is done with test folder annotation xml files to a single **csv** file. 

The command for converting the train folder annotations xml files to single csv files is - 
```
!python xml_to_csv.py -i data/Images/train -o data/annotations/train_labels.csv -l data/annotations
```

Same the command for converting the test folder annotations xml files to single csv files is - 
```
!python xml_to_csv.py -i data/Images/test -o data/annotations/test_labels.csv
```

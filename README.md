# Plant-vs-Weed
Object Detection Classifying Between Plant And Weed

Computer vision task:
This Project focuses on Classifying between Multiple Plants and Weeds Using Single Shot Multibox detector(SSD)

SSD is one of the object detection Module.

Model name used is **ssd_mobilenet_v2**
Model Config is : 
```
 'ssd_mobilenet_v2': {
        'model_name': 'ssd_mobilenet_v2_coco_2018_03_29',
        'pipeline_file': 'ssd_mobilenet_v2_coco.config',
        'batch_size': 12
    }
```

1) First of all, taken the Images and divided into 80 : 20 Train vs test images. 

2) With this images, I have generated the XML files and converted train folder annotation xml files to a single **csv** file. Same is done with test folder annotation xml files to a single **csv** file. 

The command for converting the train folder annotations xml files to single csv files is - 
```
!python xml_to_csv.py -i data/Images/train -o data/annotations/train_labels.csv -l data/annotations
```

Same the command for converting the test folder annotations xml files to single csv files is - 
```
!python xml_to_csv.py -i data/Images/test -o data/annotations/test_labels.csv
```

3) Next is genearate the train record and the test record from both of the csv files. 

4) Next is to defining **checkpoints**, setting the **training batch_size**, Setting **training steps** and **number of classes**

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

3) Next is genearate the train record and the test record from both of the csv files. Next is to storing up the train record name as train_record_name & test record name as the test_record_name. 

4) Next is to defining **checkpoints**, setting the **training batch_size**, Setting **training steps** and **number of classes**

5) Now to train the model considering the number of the **training steps and number of evaluation steps** into consideration. Always to keep in mind that to train it on the GPU for faster result (Even on GPU, you need to wait for few minutes!!).

```
!python /content/models/research/object_detection/model_main.py \
    --pipeline_config_path={pipeline_fname} \
    --model_dir={model_dir} \
    --alsologtostderr \
    --num_train_steps={num_steps} \
    --num_eval_steps={num_eval_steps}
```

6) Next is to store the model in the output directory. 

7) Next is to do the Google Authentication as we are using Google colab for GPU purpose. 

```
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials


# Authenticate and create the PyDrive client.
# This only needs to be done once in a notebook.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

fname = os.path.basename(pb_fname)
# Create & upload a text file.
uploaded = drive.CreateFile({'title': fname})
uploaded.SetContentFile(pb_fname)
uploaded.Upload()
print('Uploaded file with ID {}'.format(uploaded.get('id')))
```

8) To provide the path for frozen detection graph as this is the actual model that is used for the object detection.
```PATH_TO_CKPT = pb_fname```

Also to provide the list of the strings that is used to add correct label for each box.
```PATH_TO_LABELS = label_map_pbtxt_fname```

9) We are going to detect - **'num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks'** and in the output dict, storing out the **'detection_boxes', 'detection_scores','detection_classes'**

10) Also Printing out the coordinates for **bounding box** that are being created around plant and the weed. 

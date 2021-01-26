# object_detection_tf_api


## How to Run

```
### Step 1: Annotate some images

Training images locate in `./data/images/`
- Train/test split those files into two directories, `./data/images/train` and `./data/images/test`

- Annotate with generate `xml` files inside `./data/images/train` and `./data/images/test` folders. 


### Step 2: Open [Colab notebook]
- open tf_api_training.ipynb on colab to run it.

Generating xml_to_csv

convert all xml anntation files to csv using python file.

!python xml_to_csv.py -i data/images/train_folder -o data/annotations/train_labels.csv -l data/annotations

Generating tfrecord file using the below command


!python generate_tfrecord.py --csv_input=data/annotations/train_labels.csv --output_path=data/annotations/train.record --img_path=data/images/train_folder --label_map data/annotations/label_map.pbtxt

```
Download the base model from tensorflow api and modify the pipeline config for the model you want to use in /models/research/pretrained_model

Training the model model_main in objection detection models cloned from github/tensorflow_models

!python /content/models/research/object_detection/model_main.py \
    --pipeline_config_path={pipeline_fname} \
    --model_dir={model_dir} \
    --alsologtostderr \
    --num_train_steps={num_steps} \
    --num_eval_steps={num_eval_steps}

```
----->Exporting inference graph

Using the check point index and data file created in trained_checkpoint_prefix folder will take the latest check point file.
This will create a frozen_infernce_graph.pb file which we use for object detection classifier.

!python /content/models/research/object_detection/export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path={pipeline_fname} \
    --output_directory={output_directory} \
    --trained_checkpoint_prefix={last_model_path}
    
 ----->Testing for a image using the frozen_inference_graph.pb file 
 
 Reading the image file using PIL library.
 
 loading the model and running the tensorflow session on the image generates this output
 
 --detection_boxes,detection_scores,detection_classes,num_detections
 
 plotting this on the final image is the final result.
 

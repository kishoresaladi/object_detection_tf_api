## How to Run

```
### Step 1: Annotate some images

Training images locate in `./data/images/`
- Train/test split those files into two directories, `./data/images/train` and `./data/images/test`

- Annotate with generate `xml` files inside `./data/images/train` and `./data/images/test` folders. 


### Step 2: Open [Colab notebook]
- open tf_api_training.ipynb on colab to run it.


## How to run inference on frozen TensorFlow graph

Requirements:
- `frozen_inference_graph.pb` Frozen TensorFlow object detection model downloaded from Colab after training. 
- `label_map.pbtxt` File used to map correct name for predicted class index downloaded from Colab after training.

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
# [How to run TensorFlow object detection model faster with Intel Graphics](https://www.dlology.com/blog/how-to-run-tensorflow-object-detection-model-faster-with-intel-graphics/) | DLology Blog

## How to deploy the trained custom object detection model with OpenVINO

Requirements:
- Frozen TensorFlow object detection model. i.e. `frozen_inference_graph.pb` downloaded from Colab after training.
- The modified pipeline config file used for training. Also downloaded from Colab after training.

You can also opt to download my [copy](https://github.com/Tony607/object_detection_demo/releases/download/V0.1/checkpoint.zip) of those files from the GitHub Release page.

Run the following Jupyter notebook locally and follow the instructions in side.
```
deploy/openvino_convert_tf_object_detection.ipynb
```
## Run the benchmark

Examples

Benchmark SSD mobileNet V2 on GPU with FP16 quantized weights.
```
cd ./deploy
python openvino_inference_benchmark.py\
     --model-dir ./models/ssd_mobilenet_v2_custom_trained/FP16\
     --device GPU\
     --data-type FP16\
     --img ../test/15.jpg
```
TensorFlow benchmark on cpu
```
python local_inference_test.py\
     --model ./models/frozen_inference_graph.pb\
     --img ./test/15.jpg\
     --cpu
```

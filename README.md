## AIDA code documents (Face, Flag, Landmark, ttl file generator)
Source provided by Brian Chen.

### Requirements
To make use of graphics cards for processing, NVIDIA graphics cards and drivers must be installed.  Confirm gpus are available by running `nvidia-smi` and see https://developer.nvidia.com/cuda-zone for more details on CUDA.


$ nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 430.26       Driver Version: 430.26       CUDA Version: 10.2     |
```## CU Face, Flag, Landmark Detection
- Task: Detection and recognition for face, flag, and landmark.
- Source: https://github.com/isi-vista/aida_face_building
- Input: LDC2019E42 unpacked data, Columbia object detection, UIUC text output.
```

## CU Face, Flag, Landmark Detection
- Task: Detection and recognition for face, flag, and landmark.
- Source: https://github.com/isi-vista/aida_face_building
- Input: LDC2019E42 unpacked data, Columbia object detection, UIUC text output.
```
# Initialization
corpus_path = LDC2019E42
working_path = shared + 'cu_FFL_shared/'
model_path = models + 'cu_FFL_models/'
Lorelei = 'LDC2018E80_LORELEI_Background_KB/data/entities.tab'

# Input Paths
# Source corpus data paths
parent_child_tab = corpus_path + 'docs/parent_children.sorted.tab'
kfrm_msb = corpus_path + 'docs/masterShotBoundary.msb'
kfrm_ldcc = corpus_path + 'data/video_shot_boundaries/representative_frames/'
jpg_ldcc = corpus_path + 'data/jpg/jpg/' 
jpg_path = working_path + 'jpg/jpg/'
kfrm_path = working_path + 'video_shot_boundaries/representative_frames/'
ltf_path = corpus_path + 'data/ltf/ltf/'

#UIUC text mention result paths
txt_mention_ttl_path = working_path + 'uiuc_ttl_results/' + version_folder + uiuc_run_folder # 1/7th May

# CU object detection result paths
det_results_path_graph = working_path + 'cu_objdet_results/' + version_folder + 'aida_output_34.pkl'
det_results_path_img = working_path + 'cu_objdet_results/' + version_folder + 'det_results_merged_34a.pkl'
det_results_path_kfrm = working_path + 'cu_objdet_results/' + version_folder + 'det_results_merged_34b.pkl'


# Model Paths
face_model_path = model_path + 'models/'

# Face detection and recognition
face_det_jpg = working_path+'face_det_jpg'
face_det_kf = working_path+'face_det_kf'
face_class_jpg = working_path+'face_class_jpg'
face_class_kf = working_path+'face_class_kf'
obj_det_results = working_path+'obj_det'

bbox_jpg = working_path+'bbox_jpg'
bbox_kf = working_path+'bbox_kf'

#Flag
flag_det_results = working_path+'flag_det'
flag_class_results = working_path+'flag_m18_2'

#Landmark
landmark_results = working_path+'building_result'

#RPI_result
RPI_entity_out = working_path+'txt_mention_out'
```
- Output: CU object detection, CU Face, Flag, Landmark Detection ttl files
```
# Output Paths
out_ttl = working_path + 'cu_object_FFT_ttl/' 
```
- Consumer: To CU Graph Merging.
Docker

```
$ docker build . --tag cu-face
$ docker run -itd -v /path-on-host-to-jpgs/:/data/jpgs/ --gpus 1 --name aida-face \
>  -v ${models}/facenet/:/data/models/facenet/ \
>  cu-face /bin/bash
$ docker exec -it aida-face /bin/bash

#initailize
python src/ldcc.py jpg_ldcc jpg_path
python src/ldcc.py kfrm_ldcc kfrm_path 

# Face
# CUDA_VISIBLE_DEVICES=${AVAILABLE_GPU} python src/align/align_dataset_mtcnn.py jpg_path face_det_jpg --image_size 160 --margin 32
# CUDA_VISIBLE_DEVICES=${AVAILABLE_GPU} python src/align/align_dataset_mtcnn.py kfrm_path face_det_kf --image_size 160 --margin 32
# CUDA_VISIBLE_DEVICES=${AVAILABLE_GPU} python src/classifier.py CLASSIFY face_det_jpg face_model_path+'facenet/20180402-114759/20180402-114759.pb' face_model_path+'google500_2_classifier.pkl' face_class_jpg --batch_size 1000 > face_class_jpg+'.txt'
# CUDA_VISIBLE_DEVICES=${AVAILABLE_GPU} python src/classifier.py CLASSIFY face_det_kf face_model_path+'facenet/20180402-114759/20180402-114759.pb' face_model_path+'google500_2_classifier.pkl' face_class_kf --batch_size 1000 > face_class_kf+'.txt'
# python src/bbox.py face_det_jpg bbox_jpg
# python src/bbox.py face_det_jpg bbox_kf

# Landmark & Flag
# python obj_preprocess.py jpg_path working_path+'m18' working_path+'building_list'
# python save_flag_crop.py flag_det_results working_path+'m18'
# python src/label_image.py flag_det_results flag_class_results
# python CUDA_VISIBLE_DEVICES=0 python extract_features.py \
  --config_path delf_config_example.pbtxt \
  --list_images_path working_path+'building_list' \
  --output_dir working_path+'building_feature'
# python match2.py working_path+'building_feature' landmark_results

# Create ttl
# python read_RPI_entity.py txt_mention_ttl_path RPI_entity_out
# python create_ttl_m18.py parent_child_tab kfrm_msb face_class_jpg face_class_kf bbox_jpg bbox_kf det_results_path_graph det_results_path_img det_results_path_kfrm Lorelei flag_class_results landmark_results RPI_entity_out jpg_path
# TODO: copy resulting ttl files out.

```

```
$ docker build . --tag aida-face
$ docker run -itd -v /path-on-host-to-jpgs/:/data/jpgs/ --gpus 1 --name face-c aida-face /bin/bash 
$ docker exec -it face-c /bin/bash
# CUDA_VISIBLE_DEVICES=1 python src/align/align_dataset_mtcnn.py /data/jpgs/v_0CSFSAVrvZQj5kTJ datasets/output --image_size 160 --margin 32

```

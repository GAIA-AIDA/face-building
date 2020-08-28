echo CU Entity Recogntion begin...

MODELS=columbia_recognition_models
OUTPUT=/output

python src/ldcc.py /corpus/data/jpg/jpg/ ${MODELS}/m18/
python src/ldcc_f.py /corpus/data/video_shot_boundaries/representative_frames/ ${MODELS}/m18_f/

python src/align/align_dataset_mtcnn.py ${MODELS}/m18/ ${MODELS}/m18_a/ --image_size 160 --margin 32
python src/align/align_dataset_mtcnn.py ${MODELS}/m18_f/ ${MODELS}/m18_f_a/ --image_size 160 --margin 32
python src/classifier.py CLASSIFY ${MODELS}/m18_a/ ${MODELS}/facenet/20180402-114759/20180402-114759.pb ${MODELS}/google500_2_classifier.pkl face_class_jpg --batch_size 1000 
python src/classifier.py CLASSIFY ${MODELS}/m18_f_a/ ${MODELS}/facenet/20180402-114759/20180402-114759.pb ${MODELS}/google500_2_classifier.pkl face_class_kf --batch_size 1000 

python src/bbox.py ${MODELS}/m18_a/ bbox_jpg
python src/bbox.py ${MODELS}/m18_f_a/ bbox_kf

cd /aida/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd /aida/models/research/object_detection
python /aida/src/object_detection/obj_preprocess.py /aida/src/${MODELS}/m18/ /aida/src/m18_flag /aida/src/building_list
cd /aida/src
python object_detection/save_to_crop.py /aida/src/flag_det_results /aida/src/m18_flag
python src/label_image.py flag_det_results/m18 flag_class_results

cd /aida/src/delf/delf/delf/examples
python extract_features.py \
  --config_path delf_config_example.pbtxt \
  --list_images_path /aida/src/building_list.txt \
  --output_dir /aida/src/building_feature

python match2.py /aida/src/building_feature/ /aida/src/${MODELS}/feature_all/ /aida/src/landmark_results

cd /aida/src
python read_RPI_entity.py ${OUTPUT}/m18_PT003_r1 RPI_entity_out

python create_ttl_m18.py /corpus/docs/parent_children.sorted.tab /corpus/docs/masterShotBoundary.msb results/face_class_jpg results/face_class_kf results/bbox_jpg results/bbox_kf ${OUTPUT}/cu_objdet_results/rdf_graphs_34.pkl ${OUTPUT}/cu_objdet_results/det_results_merged_34a_jpg.pkl ${OUTPUT}/cu_objdet_results/det_results_merged_34b_kf.pkl ${MODELS}/LDC2018E80_LORELEI_Background_KB/data/entities.tab flag_class_results.pickle landmark_results.p RPI_entity_out.pickle ${MODELS}/m18/ ${MODELS}/freebase_links_f2w.json ${OUTPUT}/m18_vision

echo CU Entity Recogntion finished!
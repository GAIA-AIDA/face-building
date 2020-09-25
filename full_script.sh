echo CU Entity Recognition begin...

MODELS=/models
###MODELS=columbia_recognition_models
OUTPUT=/output/WORKING

echo CU ldcc convert...
python src/ldcc.py /corpus/data/jpg/jpg/ ${MODELS}/m18/
python src/ldcc_f.py /corpus/data/video_shot_boundaries/representative_frames/ ${MODELS}/m18_f/

echo CU align...
python src/align/align_dataset_mtcnn.py ${MODELS}/m18/ ${MODELS}/m18_a/ --image_size 160 --margin 32
python src/align/align_dataset_mtcnn.py ${MODELS}/m18_f/ ${MODELS}/m18_f_a/ --image_size 160 --margin 32
echo CU classify...
python src/classifier.py CLASSIFY ${MODELS}/m18_a/ ${MODELS}/facenet/20180402-114759/20180402-114759.pb ${MODELS}/google500_2_classifier.pkl face_class_jpg --batch_size 1000 
python src/classifier.py CLASSIFY ${MODELS}/m18_f_a/ ${MODELS}/facenet/20180402-114759/20180402-114759.pb ${MODELS}/google500_2_classifier.pkl face_class_kf --batch_size 1000 
echo CU bbox...
python src/bbox.py ${MODELS}/m18_a/ bbox_jpg
python src/bbox.py ${MODELS}/m18_f_a/ bbox_kf

cd /aida/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd /aida/models/research/object_detection
echo CU obj_preprocess...
python /aida/src/object_detection/obj_preprocess.py /aida/src/${MODELS}/m18/ /aida/src/m18_flag /aida/src/building_list
cd /aida/src
echo CU save_to_crop...
python object_detection/save_to_crop.py /aida/src/flag_det_results /aida/src/m18_flag
echo CU label_image...
python src/label_image.py flag_det_results/m18 flag_class_results

cd /aida/src/delf/delf/delf/examples
echo CU extract_features...
python extract_features.py \
  --config_path delf_config_example.pbtxt \
  --list_images_path /aida/src/building_list.txt \
  --output_dir /aida/src/building_feature

echo CU match2...
python match2.py /aida/src/building_feature/ /aida/src/${MODELS}/feature_all/ /aida/src/landmark_results

cd /aida/src
echo CU read_RPI_entity...
python read_RPI_entity.py ${OUTPUT}/uiuc_ttl_results RPI_entity_out

echo CU create_ttl_m36...
python create_ttl_m36_dry.py \
/corpus/docs/parent_children.tab \
/corpus/docs/masterShotBoundary.msb \
results/face_class_jpg results/face_class_kf results/bbox_jpg results/bbox_kf \
${OUTPUT}/cu_objdet_results/rdf_graphs_34.pkl \
${OUTPUT}/cu_objdet_results/det_results_merged_34a_jpg.pkl \
${OUTPUT}/cu_objdet_results/det_results_merged_34b_kf.pkl \
${MODELS}/LDC2020E27_AIDA_Phase_2_Practice_Topics_Reference_Knowledge_Base_V1.1/data/entities.tab \
flag_class_results.pickle \
landmark_results.p \
RPI_entity_out.pickle \
${MODELS}/m18/ \
${MODELS}/freebase_links_f2w.json \
${OUTPUT}/m36_vision

echo CU Entity Recognition finished!

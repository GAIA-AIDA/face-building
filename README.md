# AIDA Face, Building, Flag Recognition

Source provided by Brian Chen

-----
To build and run the system:

```
$ CORPUS=/path/to/ldc/corpus
$ Columbia_root=/path/to/columbia_data_root
$ GPU_ID=[a single integer index to the GPU]

docker run --gpus 8 -it -v /dvmm-filer2/projects/AIDA/data/ldc_eval_m18/LDC2019E42_AIDA_Phase_1_Evaluation_Source_Data_V1.0:/aida/src/m18_data -v /dvmm-filer2/projects/AIDA/data/columbia_data_root:/aida/src/columbia_data_root -e CUDA_VISIBLE_DEVICES=${GPU_ID} brian271828/brian_aida:0511

$ chmod +x ./full_script.sh
$ docker build . --tag [TAG]
$ CONTAINER_ID=`docker run --gpus 8 -it -v ${CORPUS}:/aida/src/m18_data -v ${Columbia_root}:/aida/src/columbia_data_root --name aida-cu-fd [TAG] /bin/bash`
$ docker exec -it ${CONTAINER_ID} /bin/bash

root@e28efc0283e2:~/src# . ./full_script.sh 

```

If `/path/to/columbia_data_root/columbia_vision_shared/m18_vision`  exists, it means the system has run successfully.




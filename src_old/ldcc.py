import numpy as np
import cv2
from glob import glob
import sys
#for name in glob('/dvmm-filer2/projects/AIDA/data/ldc_eval_m9/data/jpg/jpg/*.jpg.ldcc'):
#for name in glob('/dvmm-filer2/projects/AIDA/data/ldc_eval_m9/data/png/png/*.png.ldcc'):
#for name in glob('/dvmm-filer2/projects/AIDA/data/ldc_eval_m9/data/gif/gif/*.gif.ldcc'):

jpg_path = sys.argv[1]
jpg_out = sys.argv[2]
jpg_out2 = jpg_out+'/'+jpg_out.split('/')[-2]+'/'
#print(jpg_out2)
import os
if not os.path.exists(jpg_out):
    os.makedirs(jpg_out)
if not os.path.exists(jpg_out2):
    os.makedirs(jpg_out2)
#for name in glob('/dvmm-filer2/projects/AIDA/data/ldc_eval_m9/data/bmp/bmp/*.bmp.ldcc'):
for name in glob(jpg_path+'*.jpg.ldcc'):
    print (name)
    #for name in glob('jpg_back/*.jpg.ldcc'):
    try:
        with open(name, 'rb') as fin:

            _ = fin.read(1024)
            imgbin = fin.read()
        imgbgr = cv2.imdecode(np.fromstring(imgbin, dtype='uint8'), cv2.IMREAD_COLOR)
        #imgbgr = cv2.resize(imgbgr, (224, 224))
        imgrgb = imgbgr[:, :, [2, 1, 0]]

        #cv2.imshow('image',imgrgb)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #print(name[6:-5])
        #print(name[9:-5])
        data = name.split('/')[-1]
        data2 = data.split('.')[0]

        cv2.imwrite(jpg_out2+data2+'.jpg',  cv2.cvtColor(imgrgb, cv2.COLOR_RGB2BGR))
    except:
        i=0
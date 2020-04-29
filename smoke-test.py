print("begin smoke test")
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
tf.test.is_gpu_available()
print("<<< See GPU info above >>>")
import numpy as np
import align
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC
print("end smoke test")

import sys
import numpy as np
import tensorflow as tf
source=sys.argv[1]
dest=sys.argv[2]
tp=int(sys.argv[3])
if tp==0:
    dtype=np.float64
else:
    dtype=np.float32

def save_tfrecords(source,dest,dtype):
    data=np.fromfile(source,dtype=dtype)
    length=data.shape[0]
    with tf.compat.v1.python_io.TFRecordWriter(dest) as writer:
        for i in range(length):
            features = tf.train.Features(feature = { "data":tf.train.Feature(bytes_list = tf.train.BytesList(value = [data[i].astype(dtype).tostring()])),})     
            example=tf.train.Example(features=features)
            serialized=example.SerializeToString()
            writer.write(serialized)



save_tfrecords(source,dest,dtype)

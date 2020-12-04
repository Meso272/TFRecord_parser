import sys
import numpy as np
import tensorflow as tf
source=sys.argv[1]
dest=sys.argv[2]
tp=int(sys.argv[3])
z=int(sys.argv[4])
if tp==0:
    dtype=np.float64
else:
    dtype=np.float32

def save_tfrecords(source,dest,dtype):
    data=np.fromfile(source,dtype=dtype).reshape((z,256,256))
    with tf.compat.v1.python_io.TFRecordWriter(dest) as writer:
        znum=z//128
        for k in range(znum):
            zstart=k*128
            if z-zstart<128:
                break
            for i in range(2):
                for j in range(2):
                    xstart=i*128
                    ystart=j*128
                    datapoint=data[zstart:zstart+128,xstart:xstart+128,ystart:ystart+128]
                    features = tf.train.Features(feature = { "data":tf.train.Feature(bytes_list = tf.train.BytesList(value = [datapoint.astype(dtype).tostring()])),})     
                    example=tf.train.Example(features=features)
                    serialized=example.SerializeToString()
                    writer.write(serialized)



save_tfrecords(source,dest,dtype)

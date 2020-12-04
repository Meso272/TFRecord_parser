import tensorflow as tf

def _parse_(example_proto,tftype):
    #tftype: tf.float32, tf.float64
    features={"data":tf.FixedLenFeature((),tf.string)}
    parsed_features=tf.parse_single_example(example_proto,features)
    data=tf.decode_raw(parsed_features['data'],tf.float32)
    return data

def load_tfrecords(srcfile):
#srcfile: input tfrecord file
    sess=tf.Session()
    dataset=tf.data.TFRecordDataset(srcfile)
    dataset=dataset.map(_parse_)

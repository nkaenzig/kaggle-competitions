import glob
import os
import pydicom
import tensorflow as tf
import numpy as np
import mask_functions

class TFRecordGenerator:
    def __init__(self):
        self.to_example = self.dicom_to_tf_example
        
    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
        
    def dicom_to_tf_example(self, dicom_path):
        ds = pydicom.dcmread(dicom_path)

        img = np.asarray(ds.pixel_array, np.uint8)
        shape = np.array(img.shape, np.int32)

        return tf.train.Example(features=tf.train.Features(feature={
                                'label': self._int64_feature(1),
                                'shape': self._bytes_feature(shape.tobytes()),
                                'image': self._bytes_feature(img.tobytes())
                                }))

    def get_dcm_file_paths(self, directory):
        file_paths = []
        for r, d, f in os.walk(directory):
            for file in f:
                if '.dcm' in file:
                    file_paths.append(os.path.join(r, file))
        return file_paths

    def write_all_to_tfrecord(self, data_dir, tfrecord_filepath):
        file_paths = self.get_dcm_file_paths(data_dir)  
        with tf.io.TFRecordWriter(tfrecord_filepath) as writer:
            for path in file_paths:
                example = self.to_example(path) 
                writer.write(example.SerializeToString())

    def parse_function(self, example_proto):
        return tf.io.parse_single_example(example_proto, self.feature_description)

    feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'shape': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
    }


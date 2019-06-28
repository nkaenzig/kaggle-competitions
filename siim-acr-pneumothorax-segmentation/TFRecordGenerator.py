import glob
import os
import pydicom
import tensorflow as tf
import numpy as np
import pandas as pd
import mask_functions

class TFRecordGenerator:
    def __init__(self, dcim_dir, rle_path):
        self.dcim_dir = dcim_dir
        self.df_rle = pd.read_csv(rle_path).set_index('ImageId', drop=True)
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
        label, mask = self.get_image_label_and_mask(ds.SOPInstanceUID, img.shape)
        mask = np.asarray(mask, np.uint8)

        return tf.train.Example(features=tf.train.Features(feature={
                                'label': self._int64_feature(label),
                                'mask': self._bytes_feature(mask.tobytes()),
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

    def write_all_to_tfrecord(self, tfrecord_filepath):
        file_paths = self.get_dcm_file_paths(self.dcim_dir)  
        with tf.io.TFRecordWriter(tfrecord_filepath) as writer:
            for path in file_paths:
                example = self.to_example(path) 
                writer.write(example.SerializeToString())

    def decode_function(self, example_proto):
        raw_data = tf.io.parse_single_example(example_proto, self.feature_description)
        label = raw_data['label']
        shape = tf.io.decode_raw(raw_data['shape'], tf.int32)
        img = tf.io.decode_raw(raw_data['image'], tf.uint8)
        img = tf.reshape(img, shape)
        mask = tf.io.decode_raw(raw_data['mask'], tf.uint8)
        mask = tf.reshape(mask, shape)

        return img, label, mask

    def get_image_label_and_mask(self, image_id, image_shape):
        rle = self.df_rle.loc[image_id, ' EncodedPixels']

        mask = np.zeros(image_shape)
        label = 0
        try:
            if isinstance(rle, pd.Series):
                rle = rle[~rle.str.contains('-1')]
                label = len(rle) > 0 
            else:
                label = '-1' not in rle
        except KeyError:
            label = 0
        
        if label == 0:
            pass
        elif isinstance(rle, pd.Series):
            masks = rle.apply(lambda x : mask_functions.rle2mask(x, image_shape[0], image_shape[1]).T)
            for i, v in masks.items():
                mask += v
            mask[np.where(mask != 0)] = 255
        else:
            mask = mask_functions.rle2mask(rle, image_shape[0], image_shape[1]).T

        return label, mask

    feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'shape': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'mask': tf.io.FixedLenFeature([], tf.string, default_value=''),
    }

    


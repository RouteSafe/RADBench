import tensorflow as tf
import os
from pathlib import Path

# 定义基础数据目录
BASE_DATA_DIR = '/data/data/xiaolan_data/MSLSTM/event_feature'
output_dir = Path('/data/data/xiaolan_data/MSLSTM/test_result_mydata')
output_dir.mkdir(parents=True, exist_ok=True)

flags = tf.app.flags
flags.DEFINE_string('data_dir', BASE_DATA_DIR, """Directory for storing data""")
flags.DEFINE_boolean('is_multi_scale',False,"""Run with multi-scale or not""")
flags.DEFINE_integer('input_dim',34,"""Input dimension size""")
flags.DEFINE_integer('num_neurons1',256,"""Number of hidden units""")#HAL(hn1=32,hn2=16)
flags.DEFINE_integer('num_neurons2',8,"""Number of hidden units""")#16,32
flags.DEFINE_integer('sequence_window',30,"""Sequence window size""")
flags.DEFINE_integer('attention_size',10,"""attention size""")
flags.DEFINE_integer('scale_levels',10,"""Scale level value""")
flags.DEFINE_integer('number_class',2,"""Number of output nodes""")
flags.DEFINE_integer('max_grad_norm',5,"""Maximum gradient norm during training""")
flags.DEFINE_string('wave_type','haar',"""Type of wavelet""")
flags.DEFINE_string('pooling_type','max pooling',"""Type of wavelet""")
flags.DEFINE_integer('batch_size',200,"""Batch size""")
flags.DEFINE_integer('max_epochs',100,"""Number of epochs to run""")
flags.DEFINE_float('learning_rate',0.01,"""Learning rate""")
flags.DEFINE_boolean('is_add_noise',False,"""Whether add noise""")
flags.DEFINE_integer('noise_ratio',0,"""Noise ratio""")
flags.DEFINE_string('option','AL',"""Operation[1L:one-layer lstm;2L:two layer-lstm;HL:hierarchy lstm;HAL:hierarchy attention lstm]""")
flags.DEFINE_string('log_dir', os.path.join(output_dir, 'log'), """Directory for logs""")
flags.DEFINE_string('output', os.path.join(output_dir, 'output'), """Directory for output""")
flags.DEFINE_string('stat', os.path.join(output_dir, 'stat'), """Directory for stat""")
flags.DEFINE_string('tf_tmp', os.path.join(output_dir, 'tf_tmp'), """Directory for tf_tmp""")
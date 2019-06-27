from net import Net, Mode
from utils import LeakyReLU, average_endpoint_error, pad, antipad
import math
import tensorflow as tf
slim = tf.contrib.slim
import argparse
import os
import cv2
from training_schedules import LONG_SCHEDULE

class FlowNetS(Net):

    def __init__(self, mode=Mode.TRAIN, debug=False):
        super(FlowNetS, self).__init__(mode=mode, debug=debug)

    def model(self, inputs, training_schedule, trainable=True):
        _, height, width, _ = inputs['input_a'].shape.as_list()
        stacked = False
        stacked = False
        with tf.variable_scope('FlowNetS'):
            if 'warped' in inputs and 'flow' in inputs and 'brightness_error' in inputs:
                stacked = True
                concat_inputs = tf.concat([inputs['input_a'],
                                           inputs['input_b'],
                                           inputs['warped'],
                                           inputs['flow'],
                                           inputs['brightness_error']], axis=3)
            else:
                concat_inputs = tf.concat([inputs['input_a'], inputs['input_b']], axis=3)
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                # Only backprop this network if trainable
                                trainable=trainable,
                                # He (aka MSRA) weight initialization
                                weights_initializer=slim.variance_scaling_initializer(),
                                activation_fn=LeakyReLU,
                                # We will do our own padding to match the original Caffe code
                                padding='VALID'):

                weights_regularizer = slim.l2_regularizer(training_schedule['weight_decay'])
                with slim.arg_scope([slim.conv2d], weights_regularizer=weights_regularizer):
                    with slim.arg_scope([slim.conv2d], stride=2):
                        conv_1 = slim.conv2d(pad(concat_inputs, 3), 64, 7, scope='conv1')
                        conv_2 = slim.conv2d(pad(conv_1, 2), 128, 5, scope='conv2')
                        conv_3 = slim.conv2d(pad(conv_2, 2), 256, 5, scope='conv3')

                    conv3_1 = slim.conv2d(pad(conv_3), 256, 3, scope='conv3_1')
                    with slim.arg_scope([slim.conv2d], num_outputs=512, kernel_size=3):
                        conv4 = slim.conv2d(pad(conv3_1), stride=2, scope='conv4')
                        conv4_1 = slim.conv2d(pad(conv4), scope='conv4_1')
                        conv5 = slim.conv2d(pad(conv4_1), stride=2, scope='conv5')
                        conv5_1 = slim.conv2d(pad(conv5), scope='conv5_1')
                    conv6 = slim.conv2d(pad(conv5_1), 1024, 3, stride=2, scope='conv6')
                    conv6_1 = slim.conv2d(pad(conv6), 1024, 3, scope='conv6_1')

                    """ START: Refinement Network """
                    with slim.arg_scope([slim.conv2d_transpose], biases_initializer=None):
                        predict_flow6 = slim.conv2d(pad(conv6_1), 2, 3,
                                                    scope='predict_flow6',
                                                    activation_fn=None)
                        deconv5 = antipad(slim.conv2d_transpose(conv6_1, 512, 4,
                                                                stride=2,
                                                                scope='deconv5'))
                        upsample_flow6to5 = antipad(slim.conv2d_transpose(predict_flow6, 2, 4,
                                                                          stride=2,
                                                                          scope='upsample_flow6to5',
                                                                          activation_fn=None))
                        concat5 = tf.concat([conv5_1, deconv5, upsample_flow6to5], axis=3)

                        predict_flow5 = slim.conv2d(pad(concat5), 2, 3,
                                                    scope='predict_flow5',
                                                    activation_fn=None)
                        deconv4 = antipad(slim.conv2d_transpose(concat5, 256, 4,
                                                                stride=2,
                                                                scope='deconv4'))
                        upsample_flow5to4 = antipad(slim.conv2d_transpose(predict_flow5, 2, 4,
                                                                          stride=2,
                                                                          scope='upsample_flow5to4',
                                                                          activation_fn=None))
                        concat4 = tf.concat([conv4_1, deconv4, upsample_flow5to4], axis=3)

                        predict_flow4 = slim.conv2d(pad(concat4), 2, 3,
                                                    scope='predict_flow4',
                                                    activation_fn=None)
                        deconv3 = antipad(slim.conv2d_transpose(concat4, 128, 4,
                                                                stride=2,
                                                                scope='deconv3'))
                        upsample_flow4to3 = antipad(slim.conv2d_transpose(predict_flow4, 2, 4,
                                                                          stride=2,
                                                                          scope='upsample_flow4to3',
                                                                          activation_fn=None))
                        concat3 = tf.concat([conv3_1, deconv3, upsample_flow4to3], axis=3)

                        predict_flow3 = slim.conv2d(pad(concat3), 2, 3,
                                                    scope='predict_flow3',
                                                    activation_fn=None)
                        deconv2 = antipad(slim.conv2d_transpose(concat3, 64, 4,
                                                                stride=2,
                                                                scope='deconv2'))
                        upsample_flow3to2 = antipad(slim.conv2d_transpose(predict_flow3, 2, 4,
                                                                          stride=2,
                                                                          scope='upsample_flow3to2',
                                                                          activation_fn=None))
                        concat2 = tf.concat([conv_2, deconv2, upsample_flow3to2], axis=3)

                        predict_flow2 = slim.conv2d(pad(concat2), 2, 3,
                                                    scope='predict_flow2',
                                                    activation_fn=None)
                    """ END: Refinement Network """

                    flow = predict_flow2 * 20.0
                    # TODO: Look at Accum (train) or Resample (deploy) to see if we need to do something different
                    flow = tf.image.resize_bilinear(flow,
                                                    tf.stack([height, width]),
                                                    align_corners=True)

                    return {
                        'predict_flow6': predict_flow6,
                        'predict_flow5': predict_flow5,
                        'predict_flow4': predict_flow4,
                        'predict_flow3': predict_flow3,
                        'predict_flow2': predict_flow2,
                        'flow': flow,
                    }

    def loss(self):
        return 0.

def get_flownet_feature(concat_inputs, trainable=True):
    with tf.variable_scope('FlowNetS'):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            # Only backprop this network if trainable
                            trainable=trainable,
                            # He (aka MSRA) weight initialization
                            weights_initializer=slim.variance_scaling_initializer(),
                            activation_fn=LeakyReLU,
                            # We will do our own padding to match the original Caffe code
                            padding='VALID'):

            weights_regularizer = slim.l2_regularizer(LONG_SCHEDULE['weight_decay'])
            with slim.arg_scope([slim.conv2d], weights_regularizer=weights_regularizer):
                with slim.arg_scope([slim.conv2d], stride=2):
                    conv_1 = slim.conv2d(pad(concat_inputs, 3), 64, 7, scope='conv1')
                    conv_2 = slim.conv2d(pad(conv_1, 2), 128, 5, scope='conv2')
                    conv_3 = slim.conv2d(pad(conv_2, 2), 256, 5, scope='conv3')

                conv3_1 = slim.conv2d(pad(conv_3), 256, 3, scope='conv3_1')
                with slim.arg_scope([slim.conv2d], num_outputs=512, kernel_size=3):
                    conv4 = slim.conv2d(pad(conv3_1), stride=2, scope='conv4')
                    conv4_1 = slim.conv2d(pad(conv4), scope='conv4_1')
                    conv5 = slim.conv2d(pad(conv4_1), stride=2, scope='conv5')
                    conv5_1 = slim.conv2d(pad(conv5), scope='conv5_1')
                conv6 = slim.conv2d(pad(conv5_1), 1024, 3, stride=2, scope='conv6')
                conv6_1 = slim.conv2d(pad(conv6), 1024, 3, scope='conv6_1')
    shape = conv6_1.shape.as_list()
    conv6_1 = tf.reshape(conv6_1, [-1, shape[-3]*shape[-2]*shape[-1]])
    return conv6_1

def main():
    net = FlowNetS(mode=Mode.TEST)
    net.test(
        checkpoint='/mnt/Work/git/flownet2-tf/checkpoints/FlowNetS/flownet-S.ckpt-0',
        input_a_path=FLAGS.input_a,
        input_b_path=FLAGS.input_b,
        out_path=FLAGS.out,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_a',
        type=str,
        required=True,
        help='Path to first image'
    )
    parser.add_argument(
        '--input_b',
        type=str,
        required=True,
        help='Path to second image'
    )
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='Path to output flow result'
    )
    FLAGS = parser.parse_args()

    # Verify arguments are valid
    if not os.path.exists(FLAGS.input_a):
        raise ValueError('image_a path must exist')
    if not os.path.exists(FLAGS.input_b):
        raise ValueError('image_b path must exist')
    if not os.path.isdir(FLAGS.out):
        raise ValueError('out directory must exist')    
    main()
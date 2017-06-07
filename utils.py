import tensorflow as tf
import numpy as np
# import anything you want

# Generate convolution layer
def ConvLayer(input, depth_in, depth_out, name="conv", kernel_size=3, acti=True):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, depth_in, depth_out], 
                                            stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[depth_out]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        if (acti==True):
            act = tf.nn.relu(conv + b)
            tf.summary.histogram("activations", act)
            result = act
        else:
            result = conv + b
        return result
        
# Generate multiple same-size convlayers, the last one has no activation function
def MultiConvLayer(data, NUM_OF_LAYERS, layer_current=0, kernel_size=3, hidden_depth=64):
    for i in range(NUM_OF_LAYERS):
        if i == 0:
            layer_data = data
        else:
            layer_data = layer_result
        if i >= NUM_OF_LAYERS - 1:
            activation = False
        else:
            activation = True
        layer_name = "conv_" + str(i)
        layer_result = ConvLayer(layer_data, 
                                 layer_data.get_shape().as_list()[3], 
                                 hidden_depth, 
                                 name=layer_name, 
                                 kernel_size=kernel_size, 
                                 acti=activation)
    return layer_result

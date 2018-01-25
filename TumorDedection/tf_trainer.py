import tensorflow as tf
import numpy as np
import datasource as ds
import json
import os
from PIL import Image

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_CHANNEL = 1
LEARNING_RATE = 0.01
EPOCH_COUNT = 10
BATCH_SIZE = 50

RANDOM_SEED = 2
WEIGHT_COUNTER = 0
BIAS_COUNTER = 0
CONVOLUTION_COUNTER = 0
POOLING_COUNTER = 0

train_accuracies = []
train_costs = []
test_accuracy = 0
test_cost = 0
isTest = False


def new_weights(shape):
    global WEIGHT_COUNTER
    weight = tf.Variable(tf.random_normal(
        shape=shape, dtype=tf.float32, seed=RANDOM_SEED), name='w_' + str(WEIGHT_COUNTER))
    WEIGHT_COUNTER += 1
    return weight


def new_biases(length):
    global BIAS_COUNTER
    bias = tf.Variable(
        tf.random_normal(shape=[length], dtype=tf.float32, seed=RANDOM_SEED + 1), name='b_' + str(BIAS_COUNTER))
    BIAS_COUNTER += 1
    return bias


def new_conv_layer(input, num_input_channels, filter_size, num_filters, pooling=2):
    global CONVOLUTION_COUNTER
    global POOLING_COUNTER
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input, filter=weights,
                         strides=[1, 1, 1, 1], padding='SAME',
                         name='conv_' + str(CONVOLUTION_COUNTER))
    CONVOLUTION_COUNTER += 1

    layer = tf.add(layer, biases)

    layer = tf.nn.relu(layer)

    if pooling is not None and pooling > 1:
        layer = tf.nn.max_pool(value=layer, ksize=[1, pooling, pooling, 1],
                               strides=[1, pooling, pooling, 1], padding='SAME',
                               name='pool_' + str(POOLING_COUNTER))
    POOLING_COUNTER += 1

    return layer, weights


def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


def new_fc_layer(input, num_inputs, num_outputs):
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.add(tf.matmul(input, weights), biases)
    # layer = tf.nn.relu(layer)
    return layer

def main():
    tf.reset_default_graph()

    TEST = True

    input_placeholder = tf.placeholder(tf.float32,
                                       shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL],
                                       name = "input_placeholder")

    output_placeholder = tf.placeholder(tf.float32,
                                        shape = [None, 1],
                                        name="output_placeholder")

    layer_conv_1, weights_conv_1 = new_conv_layer(input=input_placeholder,
                                                  num_input_channels=IMAGE_CHANNEL,
                                                  filter_size=5,
                                                  num_filters=64,
                                                  pooling=2)

    layer_conv_2, weights_conv_2 = new_conv_layer(input=layer_conv_1,
                                                  num_input_channels=64,
                                                  filter_size=3,
                                                  num_filters=64,
                                                  pooling=2)

    layer_conv_3, weights_conv_3 = new_conv_layer(input=layer_conv_2,
                                                  num_input_channels=64,
                                                  filter_size=3,
                                                  num_filters=128,
                                                  pooling=2)

    layer_flat, num_features = flatten_layer(layer_conv_3)

    layer_fc_1 = new_fc_layer(input=layer_flat,
                              num_inputs=num_features,
                              num_outputs=512)

    layer_fc_1 = tf.nn.sigmoid(layer_fc_1)

    #if TEST is not True:
        #layer_fc_1 = tf.nn.dropout(layer_fc_1, 0.5)

    layer_output = new_fc_layer(layer_fc_1, num_inputs=512, num_outputs=1)

    layer_output = tf.nn.sigmoid(layer_output)

    cost = tf.reduce_sum(tf.squared_difference(layer_output, output_placeholder) / 2)

    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

    correct_predictions = tf.equal(tf.round(layer_output), output_placeholder)

    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    init_g = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_g)
        sess.run(init_l)
        saver = tf.train.Saver()

        if TEST == False:
            train_model(sess, input_placeholder, output_placeholder, accuracy, cost, optimizer)

            saver.save(sess, "./model.ckpt")
            test_model(sess, input_placeholder, output_placeholder, accuracy, cost)
            f = open("./data.json", "w")
            f.write(json.dumps(
                {"batch_costs": train_costs, "batch_accuracies": train_accuracies, "test_cost": test_cost,
                 "test_accuracy": test_accuracy},
                indent=4, sort_keys=True))
            f.close()
        else:
            saver.restore(sess, "./model.ckpt")
            test_model(sess, input_placeholder, output_placeholder, accuracy, cost)




def train_model(sess, input_placeholder, output_placeholder, accuracy, cost, optimizer):

    for batch_index, batch_images, batch_labels in ds.training_batch_generator(BATCH_SIZE):

        for current_epoch in range(EPOCH_COUNT):
            feed = {input_placeholder: batch_images,
                    output_placeholder: batch_labels}

            epoch_accuracy, epoch_cost, _ = sess.run([accuracy, cost, optimizer],
                                                     feed_dict=feed)

            print("Batch {:3}, Epoch {:3} -> Accuracy: {:3.1%}, Cost: {}".format(
                batch_index + 1, current_epoch + 1, epoch_accuracy, epoch_cost))

        batch_validation_accuracy, batch_validation_cost = test_model(sess, input_placeholder,
                                                                          output_placeholder,
                                                                          accuracy, cost)
        train_accuracies.append(batch_validation_accuracy)
        train_costs.append(batch_validation_cost)
        print("Batch validation accuracy: {} cost: {}".format(batch_validation_accuracy, batch_validation_cost))

    print("Train Finished!")



def test_model(sess, input_placeholder, output_placeholder, accuracy, cost):

    total_accuracy = 0
    total_cost = 0
    batches = 1
    for batch_index, test_images, test_labels in ds.test_batch_generator(BATCH_SIZE):
        feed = {
            input_placeholder: test_images,
            output_placeholder: test_labels
        }

        test_accuracy, test_cost = sess.run(
            [accuracy, cost], feed_dict=feed)

        total_accuracy += test_accuracy
        total_cost += test_cost
        batches = batch_index + 1

    overall_accuracy = total_accuracy / batches
    overall_cost = total_cost / batches
    test_cost = overall_cost
    test_accuracy = overall_accuracy
    print("Total test accuracy: {:3.1%}".format(overall_accuracy))

    return overall_accuracy, overall_cost


if __name__ == '__main__':
    main()
#!/usr/bin/env python

import tensorflow as tf
import numpy as np


m = 0.0
s = 2.0
def linear(input_data, output_size):
    with tf.variable_scope("linear") as linear_scope:
        print("Linear scope ", linear_scope.name, linear_scope.reuse)
        matrix = tf.get_variable("matrix", shape=(output_size, input_data.shape[0].value), initializer=tf.random_normal_initializer(m,s))
        bias = tf.get_variable("bias", shape=(output_size, 1), initializer=tf.random_normal_initializer(m,s))
        return tf.squeeze(tf.add(tf.matmul(matrix, tf.expand_dims(input_data, 1)), bias), [1], name="output")

def print_tree(t, indent=0):
    if type(t) == list:
        for x in t:
            print_tree(x, indent+1)
    else:
        print((" " * int(indent)) + str(t))

def walk_tree_backwards(node):
    node_list = [node.name]
    parents = []
    for parent in node.op.inputs:
        parents.append(walk_tree_backwards(parent))
    if len(parents) > 0:
        node_list.append(parents)
    return node_list


clamped_params = False
num_layers = 5
nodes_per_layer = 10
input = tf.placeholder(tf.float32, shape=(nodes_per_layer), name="input")
working_in = input
# Build up a deep net
for layer_index in range(num_layers):
    scope = "layer1" if clamped_params else f"layer{layer_index}" 
    scope_reuse = tf.AUTO_REUSE if clamped_params else False
    with tf.variable_scope(scope, reuse=scope_reuse):
    #with tf.variable_scope(f"layer{layer_index}"):
    #with tf.variable_scope(f"layer1", reuse=tf.AUTO_REUSE): # with this, will see exact same tensor reused in the graph in each layer.
        layer_out = tf.sigmoid(linear(working_in, nodes_per_layer))
        working_in = layer_out

final_out = tf.sigmoid(linear(layer_out, 1))

tree = walk_tree_backwards(final_out)
print_tree(tree)
  
with tf.Session() as sess:
    # Initialize all the layers
    sess.run(tf.global_variables_initializer())

    # Create some random input data to feed in
    in_value = np.random.random([nodes_per_layer])
    m1 = tf.get_default_graph().get_tensor_by_name("layer1/linear/matrix:0")
    m2 = m1
    if not clamped_params:
        m2 = tf.get_default_graph().get_tensor_by_name("layer2/linear/matrix:0")
    results = sess.run([final_out, m1, m2], feed_dict={input: in_value})

    print()
    print("M1 ", results[1])
    print()
    print("M2 ", results[2])
    print()
    print(in_value, " => ", results[0])

    # Create some different random input data
    in_value = np.random.random([nodes_per_layer])
    results = sess.run([final_out, m1, m2], feed_dict={input: in_value})
    
    print()
    print("M1 ", results[1])
    print()
    print("M2 ", results[2])
    print()
    print(in_value, " => ", results[0])

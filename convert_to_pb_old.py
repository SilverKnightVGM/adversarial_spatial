# USE freeze_tf_graph.py
import os
import tensorflow as tf
# https://github.com/tensorflow/tensorflow/issues/22197
from tensorflow.contrib import image #this is necessary because it was imported in the cnn file definition

# trained_checkpoint_prefix = 'models/model.ckpt-49491'
trained_checkpoint_prefix = 'GREEBLES/output_inv_train/test/checkpoint-80000'
export_dir = os.path.join('saved_models', str(trained_checkpoint_prefix.split("/")[1]) + "builder")
# output_node_names = "output"
# output_node_names = "costs/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"
# output_node_names = "logit/xw_plus_b"
# output_node_names = "gradients/costs/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul"


graph = tf.Graph()
with tf.compat.v1.Session(graph=graph) as sess:
    # Restore from checkpoint
    loader = tf.compat.v1.train.import_meta_graph(trained_checkpoint_prefix + '.meta')
    loader.restore(sess, trained_checkpoint_prefix)
    ##############
    # names = [n.name for n in tf.get_default_graph().as_graph_def().node]
    # print(names)
    # Find which is the softmax (output node) name
    # for op in graph.get_operations(): 
        # t = op.type.lower()
        # if t == 'maxpool' or t == 'relu' or t == 'conv2d':
            # for input in op.inputs:
                # print(input.name, input.dtype)
        # print(op.name, op.outputs)
        
    ##############
    # graph = sess.graph
    # with graph.as_default():
        # input_graph_def = graph.as_graph_def()
        
        # for node in input_graph_def.node:
            # node.device = ""
                
        # for node in input_graph_def.node:            
            # if node.op == 'RefSwitch':
                # node.op = 'Switch'
                # for index in range(len(node.input)):
                    # if 'moving_' in node.input[index]:
                        # node.input[index] = node.input[index] + '/read'
            # elif node.op == 'AssignSub':
                # node.op = 'Sub'
                # if 'use_locking' in node.attr: del node.attr['use_locking']
            # elif node.op == 'AssignAdd':
                # node.op = 'Add'
                # if 'use_locking' in node.attr: del node.attr['use_locking']
    
    
    # Export checkpoint to SavedModel
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(sess,
                                         [tf.saved_model.TRAINING, tf.saved_model.SERVING],
                                         strip_default_attrs=True)
    builder.save()
    ###############
    # output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names.split(','))
    # tf.train.write_graph(output_graph_def, export_dir, "test.pb", False)
    ###############
    ###############
    # for fixing the bug of batch norm
    # gd = sess.graph.as_graph_def()
    # for node in gd.node:            
        # if node.op == 'RefSwitch':
            # node.op = 'Switch'
            # for index in range(len(node.input)):
                # if 'moving_' in node.input[index]:
                    # node.input[index] = node.input[index] + '/read'
        # elif node.op == 'AssignSub':
            # node.op = 'Sub'
            # if 'use_locking' in node.attr: del node.attr['use_locking']
        # elif node.op == 'AssignAdd':
            # node.op = 'Add'
            # if 'use_locking' in node.attr: del node.attr['use_locking']
            
    # converted_graph_def = tf.graph_util.convert_variables_to_constants(sess, gd, output_node_names.split(","))
    # tf.train.write_graph(converted_graph_def, export_dir, "greebles_inv_train.pb", as_text=False)
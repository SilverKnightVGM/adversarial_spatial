import os
import tensorflow as tf
# https://github.com/tensorflow/tensorflow/issues/22197
from tensorflow.contrib import image #this is necessary because it was imported in the cnn file definition

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True, fix_batch_norm=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
                
        for node in input_graph_def.node:            
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr: del node.attr['use_locking']
                
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph
		
# trained_checkpoint_prefix = 'models/model.ckpt-49491'
# trained_checkpoint_prefix = 'GREEBLES/output_inv_train/test/checkpoint-80000'
trained_checkpoint_prefix = 'output_adv_train_no-invert/test/checkpoint-80000'
if(str(trained_checkpoint_prefix.split("/")[1]) != 'test'):
    export_dir = os.path.join('saved_models', str(trained_checkpoint_prefix.split("/")[1]))
else:
    export_dir = os.path.join('saved_models', str(trained_checkpoint_prefix.split("/")[0]))
# output_node_names = "output"
output_node_names = "costs/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"
# output_node_names = "map/TensorArrayStack/TensorArrayGatherV3"
# output_node_names = "ArgMax" #not working for softmax layer
# output_node_names = "Equal" #not working


graph = tf.Graph()
with tf.compat.v1.Session(graph=graph) as sess:
    # Restore from checkpoint
    loader = tf.compat.v1.train.import_meta_graph(trained_checkpoint_prefix + '.meta')
    loader.restore(sess, trained_checkpoint_prefix)
    
            
    frozen_graph = freeze_session(sess, output_names=output_node_names.split(","))
    tf.train.write_graph(frozen_graph, export_dir, "cifar_softmax.pb", as_text=False)
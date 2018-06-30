import tensorflow as tf
from MicroExpNet import MicroExpNet

modelDir = './Models/OuluCASIA'
output_node_names = ['Add_1']

x = tf.placeholder(tf.float32, shape=[None, 84*84])
classifier = MicroExpNet(x)
weights_biases_deployer = tf.train.Saver({"wc1": classifier.w["wc1"], \
										  "wc2": classifier.w["wc2"], \
										  "wfc": classifier.w["wfc"], \
										  "wo":  classifier.w["out"], \
										  "bc1": classifier.b["bc1"], \
										  "bc2": classifier.b["bc2"], \
										  "bfc": classifier.b["bfc"], \
										  "bo":  classifier.b["out"]})

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    weights_biases_deployer.restore(sess, tf.train.latest_checkpoint(modelDir))

    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    with open('output_graph.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())

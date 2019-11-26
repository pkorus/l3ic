import os
import numpy as np
import tensorflow as tf

class DCN():

    def __init__(self, model='32c'):
        self.model = model
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        with open(os.path.join('./models', 'dcn_{}.pb'.format(model)), 'rb') as f:
            with self.graph.as_default():
                self.x = tf.placeholder(tf.float32)
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(graph_def, input_map={'x_twitterdcn': self.x})
                self.z, = self.graph.get_operation_by_name('import/twitterdcn/latent').outputs
                self.y, = self.graph.get_operations()[-1].outputs
                self.codebook = self.graph.get_tensor_by_name('import/twitterdcn/optimization/entropy/Const:0')
                self.n_latent = int(self.z.shape[-1])
        
    def compress(self, batch_x):
        with self.graph.as_default():
            return self.sess.run(self.z, feed_dict={self.x: batch_x})
            
    def decompress(self, batch_z):
        with self.graph.as_default():
            return self.sess.run(self.y, { self.z: batch_z }).clip(0, 1)

    def process(self, batch_x):
        with self.graph.as_default():
            return self.sess.run(self.y, feed_dict={self.x: batch_x})

    def get_codebook(self):
        with self.graph.as_default():
            return self.sess.run(self.codebook).reshape((-1,))

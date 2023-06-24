import numpy as np
import tensorflow as tf
from model import *

# Model
model = RecurrentModel(2, 1, [FullyConnectedComponent(2, recurrent_inputs=['a']),
                              FullyConnectedComponent(name='a')])
x, out, recurrent_inputs, recurrent_outputs = model.build()

print
print model # Displays configured model components
print

initialize = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(initialize)
    
    val_out, hidden_a = session.run([out, recurrent_outputs['a']],
                                 feed_dict={x: [[1.0, 0.0]], recurrent_inputs['a']: [[0.0]]})
    print val_out
    print hidden_a
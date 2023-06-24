import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import time
from display_utils import DynamicConsoleTable
from sound_utils import Sounds

class SupervisedTrainer(object):
    
    def __init__(self,
                 x, out,
                 use_gpu=True,
                 use_sound=True,
                 show_loss=True,
                 loss_display_interval=1,
                 loss_display_saved_iterations=float('inf'),
                 loss_display_starting_iteration=0,
                 show_accuracy=True,
                 accuracy_display_interval=1,
                 accuracy_display_saved_iterations=float('inf'),
                 accuracy_display_starting_iteration=0,
                 display_resolution=1000):
        
        self.x = x
        self.out = out
        self.use_gpu = use_gpu
        self.use_sound = use_sound
        self.show_loss = show_loss
        self.loss_display_interval = loss_display_interval 
        self.loss_display_saved_iterations = loss_display_saved_iterations
        self.loss_display_starting_iteration = loss_display_starting_iteration
        self.show_accuracy = show_accuracy
        self.accuracy_display_interval = accuracy_display_interval
        self.accuracy_display_saved_iterations = accuracy_display_saved_iterations
        self.accuracy_display_starting_iteration = accuracy_display_starting_iteration
        self.last_loss_displayed = -float('inf')
        self.last_accuracy_displayed = -float('inf')
        self.display_resolution = display_resolution
        
        ###############################################################################
        # Pyplot setup
        plt.ion() # Enable interactive mode
        ###############################################################################
        # Table setup
        self.progress_bar_size = 20
        ###############################################################################
        # Sound setup
        self.sounds = Sounds()
        if self.use_sound:
            self.sounds.open()
            
    def shrink(self, values, window_size=1):
        if window_size == 1:
            return values
        new_values = []
        total = 0.0
        for i in range(len(values)):
            total += values[i]
            if (i+1) % window_size == 0:
                new_values.append(total / window_size)
                total = 0.0
        return new_values
        
    def display_loss(self, loss_values, sustained_loss_values, iteration):
        window_size = len(loss_values) / int(self.display_resolution) + 1
        if len(loss_values) >= self.last_loss_displayed + window_size:
            self.last_loss_displayed = len(loss_values)
            loss_title = 'Loss'
            loss_fig = plt.figure(loss_title)
            loss_fig.clear()
            plt.title(loss_title)
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            l_values = self.shrink(loss_values, window_size)
            #sl_values = self.shrink(sustained_loss_values, window_size)
            xs = np.array(range(len(l_values))) * window_size + iteration - len(loss_values) + 1
            plt.plot(xs, l_values)
            #plt.plot(xs, sl_values)
            plt.pause(0.00001)
            
    def display_accuracy(self, validation_accuracy_values, max_accuracy_values, iteration):
        window_size = len(validation_accuracy_values) / int(self.display_resolution) + 1
        if len(validation_accuracy_values) >= self.last_accuracy_displayed + window_size:
            self.last_accuracy_displayed = len(validation_accuracy_values)
            accuracy_title = 'Accuracy'
            accuracy_fig = plt.figure(accuracy_title)
            accuracy_fig.clear()
            plt.title(accuracy_title)
            plt.xlabel('Batch')
            plt.ylabel('Accuracy')
            values = self.shrink(validation_accuracy_values, window_size)
            xs = np.array(range(len(values))) * window_size + \
                    iteration - len(validation_accuracy_values) + 1
            plt.plot(xs, values)
            plt.pause(0.00001)

    def update_output(self, iteration, loss_values, sustained_loss_values,
                      validation_accuracy_values, max_accuracy_values, override=False):
        # Show/update loss display
        if iteration % self.loss_display_interval == 0 and self.show_loss or override:
            self.display_loss(loss_values, sustained_loss_values, iteration)
        # Show/update accuracy display
        if not self.skip_evaluation:
            if iteration % self.accuracy_display_interval == 0 and self.show_accuracy or override:
                self.display_accuracy(validation_accuracy_values, max_accuracy_values, iteration)
    
    def train(self,
              training_data,
              training_labels,
              validation_data=None,
              validation_labels=None,
              skip_evaluation=False,
              loss_fn=None,
              optimizer_fn=None,
              accuracy_fn=None,
              max_epochs=float('inf'),
              batch_size=1,
              validation_set_size=None,
              validation_interval=1,
              loss_threshold=0.0,
              sustained_loss_decay_rate=0.9,
              row_output_interval=None,
              label_type=tf.int64):
        
        assert loss_fn, 'Must specify a loss_fn (a function that takes (out, y) as input)'
        assert optimizer_fn, 'Must specify a optimizer_fn (a function that takes loss as input)'
        
        if validation_data is not None:
            validation_set_size = (validation_set_size and \
                                   min(validation_set_size, len(validation_data))) or \
                                   len(validation_data)
            
        assert len(training_data) == len(training_labels), \
            'Number of training data and training labels do not match'
        if not skip_evaluation and (validation_data is not None or validation_labels is not None):
            assert validation_data is not None and validation_labels is not None and \
                len(validation_data) == len(validation_labels), \
                'Number of validation data and validation labels do not match'
        else:
            skip_evaluation = True
        if not skip_evaluation:
            assert accuracy_fn, \
                'Must specify an accuracy_fn (a function that takes (out, y) as input),' + \
                ' in order to evaluate the validation set'
                
        if len(training_data) % batch_size != 0:
            print 'WARNING: batch_size does not evenly divide len(training_data).' + \
                'Not all training data will be used in every epoch'
                
        validation_data = np.array(validation_data)
        validation_labels = np.array(validation_labels)
                
        num_training_batches = len(training_data) / batch_size
            
        row_output_interval = row_output_interval or num_training_batches
        
        y = tf.placeholder(label_type, [None])
        loss = loss_fn(self.out, y)
        optimizer = optimizer_fn(loss)
        accuracy = accuracy_fn(self.out, y) if accuracy_fn else None
        
        self.skip_evaluation = skip_evaluation
        
        layout = [
            dict(name='Ep.', width=3, align='center'),
            dict(name='Batch', width=2*len(str(num_training_batches))+1,
                 suffix='/'+str(num_training_batches), align='center'),
            dict(name='Loss', width=8, align='center')] + \
            ([dict(name='Val Acc', width=7, suffix='%', align='center'),
              dict(name='Max Acc', width=7, suffix='%', align='center')] \
                if not self.skip_evaluation else []) + \
            [dict(name='Progress/Timestamp', width=self.progress_bar_size+2, align='center'),
             dict(name='Elapsed (s)', width=7, align='center')]

        # Run model
        done = False
        epoch = 0
        iteration = 0
        sustained_loss = 0.0
        loss_values = []
        sustained_loss_values = []
        last_validation_accuracy = 0.0
        validation_accuracy_values = []
        max_accuracy_values = []
        max_accuracy = 0.0
        start_time = time.time()
            
        # Initialize environment
        initialize = tf.initialize_all_variables()

        # Session config
        config = tf.ConfigProto(device_count={'GPU': 1 if self.use_gpu else 0})
        
        gpu_exists = True
        try:
            g = tf.Graph()
            with g.as_default():
                with tf.device('/gpu:0'):
                    test = tf.multiply(tf.constant(1.0), tf.constant(1.0))
            with tf.Session(graph=g) as sess:
                print (sess.run(test))
        except tf.errors.InvalidArgumentError:
            gpu_exists = False
        
        with tf.Session(config=config) as session:
            session.run(initialize)
            print '=========='
            print 'GPU ' + (('enabled' if gpu_exists else 'unavailable') \
                            if self.use_gpu else 'disabled')
            print
            table = DynamicConsoleTable(layout)
            table.print_header()
            multiple_rows_per_epoch = row_output_interval < num_training_batches

            while not done:
                training_data_indices = np.random.permutation(len(training_data))
                training_data_permuted = np.array(training_data)[training_data_indices]
                training_labels_permuted = np.array(training_labels)[training_data_indices]
                training_data_batches = []
                training_label_batches = []
                for i in range(num_training_batches):
                    training_data_batches.append(
                        training_data_permuted[i*batch_size:(i+1)*batch_size])
                    training_label_batches.append(
                        training_labels_permuted[i*batch_size:(i+1)*batch_size])
            
                epoch += 1
                if self.use_sound:
                    self.sounds.alert()

                # Trains on the data, in batches
                for i in range(num_training_batches):
                    iteration += 1
                    data_batch = training_data_batches[i]
                    labels_batch = training_label_batches[i]

                    _, loss_val = session.run([optimizer, loss],
                                              feed_dict={self.x: data_batch, y: labels_batch})
                    sustained_loss = sustained_loss_decay_rate * sustained_loss + \
                        (1.0 - sustained_loss_decay_rate) * loss_val

                    if len(loss_values) == self.loss_display_saved_iterations:
                        loss_values.pop(0)
                        sustained_loss_values.pop(0)
                    if iteration == self.loss_display_starting_iteration:
                        sustained_loss = loss_val
                    if iteration >= self.loss_display_starting_iteration:
                        loss_values.append(loss_val)
                        sustained_loss_values.append(sustained_loss)

                    validation_accuracy = last_validation_accuracy
                    if not skip_evaluation and iteration % validation_interval == 0:
                        validation_set_indices = np.random.choice(
                            np.arange(len(validation_data)),
                            size=validation_set_size, replace=False)
                        validation_data_batch = validation_data[validation_set_indices]
                        validation_labels_batch = validation_labels[validation_set_indices]
                        validation_accuracy = session.run(accuracy,
                                                   feed_dict={self.x: validation_data_batch,
                                                              y: validation_labels_batch})
                        last_validation_accuracy = validation_accuracy

                        if len(validation_accuracy_values) == self.accuracy_display_saved_iterations:
                            validation_accuracy_values.pop(0)
                        if iteration >= self.accuracy_display_starting_iteration:
                            validation_accuracy_values.append(validation_accuracy)

                        if validation_accuracy > max_accuracy:
                            max_accuracy = validation_accuracy
                            if self.use_sound:
                                self.sounds.success()

                        if len(max_accuracy_values) == self.accuracy_display_saved_iterations:
                            max_accuracy_values.pop(0)
                        if iteration >= self.accuracy_display_starting_iteration:
                            max_accuracy_values.append(max_accuracy)

                    progress = int(math.ceil(self.progress_bar_size * \
                                             float((iteration - 1) % num_training_batches) /\
                                             (num_training_batches - 1)))
                    elapsed = time.time() - start_time
                    progress_string = '[' + '#' * progress + ' ' * \
                        (self.progress_bar_size - progress) + ']'
                    if iteration % num_training_batches == 0 or \
                        iteration % row_output_interval == 0:
                        progress_string = time.strftime("%I:%M:%S %p", time.localtime())

                    if not self.skip_evaluation:
                        table.update(epoch,
                                     (iteration - 1) % num_training_batches + 1,
                                     sustained_loss,
                                     validation_accuracy * 100,
                                     max_accuracy * 100,
                                     progress_string,
                                     elapsed)
                    else:
                        table.update(epoch,
                                     (iteration - 1) % num_training_batches + 1,
                                     sustained_loss,
                                     progress_string,
                                     elapsed)

                    if iteration % num_training_batches == 0 or \
                        iteration % row_output_interval == 0:
                        heavy = False
                        if multiple_rows_per_epoch and iteration % num_training_batches == 0:
                            heavy = True
                        table.finalize(heavy=heavy)

                    # Termination condition
                    if sustained_loss < loss_threshold:
                        done = True
                        break

                    self.update_output(iteration, loss_values, sustained_loss_values,
                                       validation_accuracy_values, max_accuracy_values)

                # Termination condition
                if epoch >= max_epochs or sustained_loss < loss_threshold:
                    done = True
                    self.update_output(iteration, loss_values, sustained_loss_values,
                                       validation_accuracy_values, max_accuracy_values,
                                       override=True)
                    plt.pause(0)
    
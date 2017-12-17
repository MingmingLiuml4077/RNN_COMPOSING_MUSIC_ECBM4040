'''
This script is just used for generate the tensorboard graph. 
Here is exactly the same model in 'model.py' but just add name_scope for tensorboard.
And the class 'biaxial_model' here cannnot restore the paramteres pre-trained by 'model.py'
'''

import tensorflow as tf
import numpy as np

from operations import map_output_to_input_tf
import data
import time
from midi_to_statematrix import noteStateMatrixToMidi
import os
import pickle


class biaxial_model(object):
    def __init__(self,
                 t_layer_sizes,
                 n_layer_sizes,
                 input_size=80,
                 output_size=2,
                 dropout=0.5,
                 clear_graph=True,
                 trainer = tf.train.AdamOptimizer(),
                 new_trainer=None):
        
        if clear_graph:
            tf.reset_default_graph()
        
        self.t_layer_sizes = t_layer_sizes
        self.n_layer_sizes = n_layer_sizes
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        
        self.trainer = trainer
        self.new_trainer = new_trainer
        self.setup_train()
        self.setup_predict()
        
        
        
    def setup_train(self):
        t_layer_sizes = self.t_layer_sizes
        n_layer_sizes = self.n_layer_sizes
        input_size = self.input_size
        output_size = self.output_size
        dropout = self.dropout
        
        
        #####################################
        #  placeholder for input and output # 
        #####################################
        input_mat = tf.placeholder(dtype=tf.float32,shape=(None,None,None,input_size))
        output_mat = tf.placeholder(dtype=tf.float32,shape=(None,None,None,output_size))
        # input_mat shape = [n_batch,n_time,n_note,input_size]
        self.input_mat = input_mat
        self.output_mat = output_mat


        #######################
        # Building time model #
        #######################
        input_slice = input_mat[:,0:-1]
        n_batch = tf.shape(input_slice)[0]
        n_time = tf.shape(input_slice)[1]
        n_note = tf.shape(input_slice)[2]
        
        time_inputs = tf.reshape(tf.transpose(input_slice,(0,2,1,3)),(n_batch*n_note,n_time, input_size))
        # time_inputs shape = [n_batch*n_note, max_time, input_size]
        num_time_parallel = tf.shape(time_inputs)[0]
        
        # two layer LSTM with state drop out
        t_lstm_cells = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(hs),
                                                      output_keep_prob=1-dropout,
                                                      variational_recurrent=True,
                                                      dtype=tf.float32) for hs in t_layer_sizes]
        t_multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(t_lstm_cells)
        self.t_multi_rnn_cell = t_multi_rnn_cell

        t_init_state = t_multi_rnn_cell.zero_state(num_time_parallel, tf.float32)
        time_result, time_final_state = tf.nn.dynamic_rnn(t_multi_rnn_cell,
                                                          time_inputs,
                                                          initial_state=t_init_state,
                                                          dtype=tf.float32,
                                                          scope='t_rnn')
        # time_result shape = [n_batch*n_note, max_time, n_hidden]

        n_hidden = t_layer_sizes[-1]

        time_final = tf.reshape(
            tf.transpose(
                tf.reshape(time_result,
                           (n_batch,n_note,n_time,n_hidden)),
                (0,2,1,3)),
                                (n_batch*n_time,n_note,n_hidden)
                                )
        # time_final shape = [n_batch*n_time,n_note,n_hidden]
        
        
        #######################
        # Building note model #
        #######################
        
        # take last note state e.g. [1,0] as input feature.
        start_note_values = tf.zeros(shape=[n_batch*n_time,1,2])
        correct_choices = tf.reshape(output_mat[:,1:,0:-1,:],(n_batch*n_time,n_note-1,output_size))
        note_choices_inputs = tf.concat([start_note_values, correct_choices], axis=1)
        
        note_inputs = tf.concat( [time_final, note_choices_inputs], axis=2 )
        
        num_timebatch = tf.shape(note_choices_inputs)[0]
        
        # two layer LSTM with state drop out
        n_lstm_cells = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(hs),
                                                      output_keep_prob=1-dropout,
                                                      variational_recurrent=True,
                                                      dtype=tf.float32) for hs in n_layer_sizes]
        n_multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(n_lstm_cells)
        self.n_multi_rnn_cell = n_multi_rnn_cell
        
        n_init_state = n_multi_rnn_cell.zero_state(num_timebatch, tf.float32)
        note_result, note_final_state = tf.nn.dynamic_rnn(n_multi_rnn_cell,
                                                          note_inputs,
                                                          initial_state=n_init_state,
                                                          dtype=tf.float32,
                                                          scope='n_rnn')
                
        note_final = tf.reshape(tf.layers.dense(tf.reshape(note_result,(n_batch*n_time*n_note,self.n_layer_sizes[-1])),
                                                units=output_size,
                                                activation=tf.nn.sigmoid,
                                                name='output_layer')
                                ,(n_batch,n_time,n_note,output_size))
        
        with tf.name_scope('loss'):
            active_notes = input_mat[:,1:,:,0:1]
            mask = tf.concat([tf.ones_like(active_notes),active_notes],axis=3)
            
            likelihood = mask*tf.log(2*note_final*output_mat[:,1:] - note_final - output_mat[:,1:] + 1)
            self.loss = -tf.reduce_mean(likelihood)
            tf.summary.scalar('Likelihood_loss', self.loss)
        
        with tf.name_scope('train_step'):
            trainer = self.trainer
            gradients = trainer.compute_gradients(self.loss)
            gradients_clipped = [(tf.clip_by_value(t[0],-1,1),t[1]) for t in gradients]
            self.optimizer = trainer.apply_gradients(gradients_clipped)
        if self.new_trainer is not None:
            trainer2 = self.new_trainer
            gradients2 = trainer2.compute_gradients(self.loss)
            gradients_clipped2 = [(tf.clip_by_value(t[0],-1,1),t[1]) for t in gradients2]
            self.optimizer2 = trainer.apply_gradients(gradients_clipped2)
        
        
    def setup_predict(self):
        
        def _step_note(state,indata):
            # state shape: [[100,50],2], indata shape: 300
            indata = tf.expand_dims(tf.concat((indata,state[1]),axis=-1),axis=0)
            hidden = state[0]
            note_output, new_state = self.n_multi_rnn_cell.call(inputs=indata,state=hidden) # note output shape: 50, new_state shape: 100 or 50 
            prob = tf.layers.dense(note_output,
                                     units=2,
                                     activation=tf.nn.sigmoid,
                                     name='output_layer',
                                     reuse=True)
            randomness = tf.random_uniform(shape=(1,))
            shouldplay = tf.cast(randomness < (prob[0][0] * self.conservativity), tf.float32)
            shouldartic = shouldplay * tf.cast(randomness < prob[0][1], tf.float32)
            output = tf.concat([shouldplay,shouldartic],axis=-1)
            return (new_state,output)
        
        def _step_time(states,_):
            hidden = states[0] # shape: (notes*t_hidden)*2 e.g. (78*300)*2
            indata = states[1] # shape: notes*note_features e.g. 78*80
            time = states[2]
            
            output, new_state = self.t_multi_rnn_cell.call(inputs=indata,state=hidden) # output shape: notes*t_hidden
            
            start_note_values = tf.zeros((2))
            
            n_initializer = (self.n_multi_rnn_cell.zero_state(1, tf.float32),
                             start_note_values)
            
            note_result = tf.scan(_step_note,elems=output,initializer=n_initializer) # note_result shape: 78*2
            next_input = map_output_to_input_tf(note_result[1], time)
            next_input = tf.reshape(next_input,(-1,80))
            time = time + 1
            return(new_state,next_input,time, note_result[1]) 
        
        # values needed to feed when generating new songs
        self.predict_seed = tf.placeholder(dtype=tf.float32, shape=(None,80), name='predict_seed')
        self.step_to_sumulate = tf.placeholder(dtype=tf.int32, shape=(1))
        self.conservativity = tf.placeholder(dtype=tf.float32,shape=(1))
        
        num_notes = tf.shape(self.predict_seed)[0]
        
        initializer = (self.t_multi_rnn_cell.zero_state(num_notes, tf.float32), # initial state
                       self.predict_seed, # initial input
                       tf.constant(0), # time
                       tf.placeholder(dtype=tf.float32,shape=(None,2))) # hold place for note output
        
        elems = tf.zeros(shape=self.step_to_sumulate)
        with tf.name_scope('predicting'):
            time_result = tf.scan(_step_time,elems=elems,initializer=initializer)
            
            self.new_song = time_result[3]
        
    def train(self, pieces, 
              batch_size=32, 
              predict_freq=100,
              model_save_freq=100,
              show_freq=10,
              save_freq=500,
              max_epoch=10000, 
              saveto='NewSong', 
              step=319, 
              conservativity=1,
              pre_trained_model=None):
        
        cur_model_name = 'biaxial_rnn_{}'.format(int(time.time()))
        batch_generator = data.generate_batch(pieces,batch_size)
        minloss = np.inf
        
        loss_log = []
        with tf.Session() as sess:
            
            merge = tf.summary.merge_all()
            writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            
            if pre_trained_model is not None:
                try:
                    print("Load the model from: {}".format(pre_trained_model))
                    saver.restore(sess, 'model/{}'.format(pre_trained_model))
                    #writer = tf.summary.FileWriterCache.get('log/{}'.format(pre_trained_model))
                except Exception:
                    print("Load model Failed!")
                    pass
            if self.new_trainer is not None:
                optimizer = self.optimizer2
            else: optimizer = self.optimizer
            
            for i in range(max_epoch):
                X_train, y_train = next(batch_generator)
                _, loss,merge_result = sess.run((optimizer,self.loss,merge), feed_dict={self.input_mat : X_train, self.output_mat : y_train})
                
                loss_log.append(loss)
                pickle.dump(loss_log,open('model/'+cur_model_name+'_loss_log.pkl','wb'))
                
                
                if i % show_freq == 0:
                    print('Step {0}: loss is {1}'.format(i, loss))
                if (i+1) % 10 == 0:
                    writer.add_summary(merge_result, i)
                
                if (i+1) % predict_freq == 0:
                    xIpt, xOpt = map(np.array, data.getPieceSegment(pieces))
                    new_state_matrix = sess.run(self.new_song, 
                                                feed_dict={self.predict_seed:xIpt[0], 
                                                           self.step_to_sumulate:[step],
                                                            self.conservativity:[conservativity]})
                    newsong = np.concatenate((np.expand_dims(xOpt[0], 0),new_state_matrix))
                    
                    songname = str(time.time())+'.mid'
                    if not os.path.exists(os.path.join(saveto,cur_model_name)):
                        os.makedirs(os.path.join(saveto,cur_model_name))
                    noteStateMatrixToMidi(newsong, name=os.path.join(saveto,cur_model_name,songname))
                    print('New Songs {} saved to \'{}\''.format(songname, os.path.join(saveto,cur_model_name)))
                
                if (i+1) % save_freq == 0:
                    if not os.path.exists('model/'):
                        os.makedirs('model/')
                    saver.save(sess, 'model/{}'.format(cur_model_name))
                    print('{} Saved'.format(cur_model_name))
                    
                if loss <= minloss and i >= 100:
                    minloss = loss
                    if not os.path.exists('model/'):
                        os.makedirs('model/')
                    saver.save(sess, 'model/{}_{}'.format('best',cur_model_name))
                    print('{}_{} Saved'.format('best',cur_model_name))
                
    def predict(self,pieces,pre_trained_model,saveto='NewSong',step=319,conservativity=1):
        # This function predict only
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            
            print("Load the model from: {}".format(pre_trained_model))
            saver.restore(sess, 'model/{}'.format(pre_trained_model))
            
            xIpt, xOpt = map(np.array, data.getPieceSegment(pieces))
            new_state_matrix = sess.run(self.new_song, 
                                        feed_dict={self.predict_seed:xIpt[0], 
                                                   self.step_to_sumulate:[step],
                                                    self.conservativity:[conservativity]})
            newsong = np.concatenate((np.expand_dims(xOpt[0], 0),new_state_matrix))
    
            songname = str(time.time())+'.mid'
            if not os.path.exists(os.path.join(saveto,pre_trained_model)):
                os.makedirs(os.path.join(saveto,pre_trained_model))
            noteStateMatrixToMidi(newsong, name=os.path.join(saveto,pre_trained_model,songname))
            print('New Songs {} saved to \'{}\''.format(songname, os.path.join(saveto,pre_trained_model)))
        
        
#biaxial_model(t_layer_sizes=[300,300], n_layer_sizes=[100,50])
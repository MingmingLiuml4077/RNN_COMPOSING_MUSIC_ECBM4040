import tensorflow as tf
import numpy as np
from numba.types import none
from operations import map_output_to_input_tf

class biaxial_model(object):
    def __init__(self,t_layer_sizes, n_layer_sizes, input_size=80,output_size=2,dropout=0.5):
        self.t_layer_sizes = t_layer_sizes
        self.n_layer_sizes = n_layer_sizes
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        
        self.setup_train()
        self.setup_predict()
        
        
    def setup_train(self):
        t_layer_sizes = self.t_layer_sizes
        n_layer_sizes = self.n_layer_sizes
        input_size = self.input_size
        output_size = self.output_size
        dropout = self.dropout
        
        #tf.reset_default_graph()
        
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
        note_choices_inputs = tf.concat([start_note_values, correct_choices], axis=0)
        
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
        
        active_notes = input_mat[:,1:,:,0:1]
        mask = tf.concat([tf.ones_like(active_notes),active_notes],axis=3)
        
        likelihood = mask*tf.log(2*note_final*output_mat[:,1:] - note_final - output_mat[:,1:] + 1)
        self.loss = -tf.reduce_sum(likelihood)
        
        trainer = tf.train.AdamOptimizer()
        gradients = trainer.compute_gradients(self.loss)
        gradients_clipped = [(tf.clip_by_value(t[0],-1,1),t[1]) for t in gradients]
        self.optimizer = trainer.apply_gradients(gradients_clipped)
        
        
    def setup_predict(self):
        
        def _step_note(state,indata):
            # state shape: [[100,50],2], indata shape: 300
            indata = tf.concat((tf.expand_dims(indata,axis=0),state[1]),axis=-1)
            hidden = state[0]
            note_output, new_state = self.n_multi_rnn_cell.call(inputs=indata,state=hidden) # output shape: 50, new_state shape: 100 or 50 
            output = tf.layers.dense(note_output,
                                     units=2,
                                     activation=tf.nn.sigmoid,
                                     name='output_layer',
                                     reuse=True)
            return (new_state,output)
        
        def _step_time(states,_):
            hidden = states[0] # shape: (notes*t_hidden)*2 e.g. (78*300)*2
            indata = states[1] # shape: notes*note_features e.g. 78*80
            time = states[2]
            
            output, new_state = self.t_multi_rnn_cell.call(inputs=indata,state=hidden) # output shape: notes*t_hidden
            
            start_note_values = tf.zeros((1,2))
            
            n_initializer = (self.n_multi_rnn_cell.zero_state(1, tf.float32),
                             start_note_values)
            
            note_result = tf.scan(_step_note,elems=output,initializer=n_initializer) # note_result shape: 78*2
            next_input = map_output_to_input_tf(note_result[1], time)
            next_input = tf.reshape(next_input,(-1,80))
            time = time + 1
            return(new_state,next_input,time) 
        
        self.predict_seed = tf.placeholder(dtype=tf.float32, shape=(None,80), name='predict_seed')
        self.step_to_sumulate = tf.placeholder(dtype=tf.int32, shape=(1))
        
        num_notes = tf.shape(self.predict_seed)[0]
        
        initializer = (self.t_multi_rnn_cell.zero_state(num_notes, tf.float32), # initial state
                       self.predict_seed, # initial input
                       tf.constant(0)) # time
        
        elems = tf.zeros(shape=self.step_to_sumulate)
        
        time_result = tf.scan(_step_time,elems=elems,initializer=initializer)
        
        self.new_song = time_result[1]
        

import tensorflow as tf
import numpy as np

from operations import map_output_to_input_tf
import data
import time
from midi_to_statematrix import noteStateMatrixToMidi
import os
import pickle

class biaxial_model(object):
    def __init__(self,t_layer_sizes, n_layer_sizes, input_size=80,output_size=2,dropout=0.5,clear_graph=True,trainer = tf.train.AdamOptimizer(),):
        '''
        This model is a bi-axial LSTM model to generate polyphonic music.
        It has two stacks of LSTM one for time and one for note.
        args:    
            t_layer_sizes: List of ints, the layer sizes of the time LSTM networks.
            n_layer_sizes: List of ints, the layer sizes of the notenote LSTM networks.
            input_size: Ints the input sizes for each notes, generlly the data feed to the 
                model should have shape like batch_sizes*time_step*note_step*input_size.
            output_size: this should be 2 for the statematrix, which should shape like 
                batch_sizes*time_step*note_step*output_size.
            dropout: the dropout rate of each LSTM layer.
            clear_graph: If True it will clear all graph of tensorflow created.
        '''
        
        if clear_graph:
            tf.reset_default_graph()
        
        self.t_layer_sizes = t_layer_sizes
        self.n_layer_sizes = n_layer_sizes
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        
        # Initialize the tensorflow graph
        self.trainer = trainer
        self.setup_train()
        self.setup_predict()
        
        
        
    def setup_train(self):
        '''
        This function initialize the training process. Here, we split the whole process into two stacks of LSTM,
        to save the running time of training. First we feed all notes and all batches to the time model over time.
        Then we feed the output of the time model to note model over note, but again we see time as batch as well.
        Last we use a dense layer to get the note statematrix, where we see both time, notes as batches.
        '''
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
        # shape of input_mat : [n_batch,n_time,n_note,input_size]
        
        self.input_mat = input_mat
        self.output_mat = output_mat


        #######################
        # Building time model #
        #######################
        # keep all data except the last time step as input to the model
        input_slice = input_mat[:,0:-1]
        
        n_batch = tf.shape(input_slice)[0]
        n_time = tf.shape(input_slice)[1]
        n_note = tf.shape(input_slice)[2]
        
        # Here we let note as batches too to train it parallel
        # shape of time_inputs : [n_batch*n_note, n_time, input_size]
        time_inputs = tf.reshape(tf.transpose(input_slice,(0,2,1,3)),(n_batch*n_note,n_time, input_size))
       
        # number of 'batches' we train at a time
        num_time_parallel = tf.shape(time_inputs)[0]
        
        # n layer LSTM with state with drop out
        t_lstm_cells = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(hs),
                                                      output_keep_prob=1-dropout,
                                                      variational_recurrent=True,
                                                      dtype=tf.float32) for hs in t_layer_sizes]
        t_multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(t_lstm_cells)
        self.t_multi_rnn_cell = t_multi_rnn_cell

        # shape of time_result : [n_batch*n_note, max_time, n_hidden]
        t_init_state = t_multi_rnn_cell.zero_state(num_time_parallel, tf.float32)
        time_result, time_final_state = tf.nn.dynamic_rnn(t_multi_rnn_cell,
                                                          time_inputs,
                                                          initial_state=t_init_state,
                                                          dtype=tf.float32,
                                                          scope='t_rnn')
        
        # the output size of time model and the input sizze of note model
        n_hidden = t_layer_sizes[-1]

        # shape of time_final : [n_batch*n_time,n_note,n_hidden]
        time_final = tf.reshape(
            tf.transpose(
                tf.reshape(time_result,
                           (n_batch,n_note,n_time,n_hidden)),
                (0,2,1,3)),
                                (n_batch*n_time,n_note,n_hidden)
                                )
        
        
        
        #######################
        # Building note model #
        #######################
        
        # take last note state e.g. [1,0] as input feature. First need to create a zero state for the lowest note.
        # Then concatenate the last note state to the input of note model.
        start_note_values = tf.zeros(shape=[n_batch*n_time,1,2])
        correct_choices = tf.reshape(output_mat[:,1:,0:-1,:],(n_batch*n_time,n_note-1,output_size))
        note_choices_inputs = tf.concat([start_note_values, correct_choices], axis=1)
        
        note_inputs = tf.concat( [time_final, note_choices_inputs], axis=2 )
        
        # The number of 'batches' trained at a time.
        num_timebatch = tf.shape(note_choices_inputs)[0]
        
        # n layer LSTM with state drop out
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
        # The last dense layer to translate the output size to 2. Here the final output should have the size as the 
        # output matrix
        note_final = tf.reshape(tf.layers.dense(tf.reshape(note_result,(n_batch*n_time*n_note,self.n_layer_sizes[-1])),
                                                units=output_size,
                                                activation=tf.nn.sigmoid,
                                                name='output_layer')
                                ,(n_batch,n_time,n_note,output_size))
        
        # "The cost of the entire procedure is the negative log likelihood of the events all happening.
        # For the purposes of training, if the ouputted probability is P, then the likelihood of seeing a 1 is P, and
        # the likelihood of seeing 0 is (1-P). So the likelihood is (1-P)(1-x) + Px = 2Px - P - x + 1
        # Since they are all binary decisions, and are all probabilities given all previous decisions, we can just
        # multiply the likelihoods, or, since we are logging them, add the logs.
        
        # Note that we mask out the articulations for those notes that aren't played, because it doesn't matter
        # whether or not those are articulated.
        # The padright is there because self.output_mat[:,:,:,0] -> 3D matrix with (b,x,y), but we need 3d tensor with 
        # (b,x,y,1) instead"
        # ---from Daniel Johnson. He stated in his code. 
        # Now we use the same mean negative likelihood as loss. But the loss is just half of what he stated in his paper
        # we apply mean to every single digits, while he use the mean over time and notes, where each note has 2 digits to
        # represent it.
        active_notes = input_mat[:,1:,:,0:1]
        mask = tf.concat([tf.ones_like(active_notes),active_notes],axis=3)
        
        likelihood = mask*tf.log(2*note_final*output_mat[:,1:] - note_final - output_mat[:,1:] + 1)
        self.loss = -tf.reduce_mean(likelihood)
        
        # Here we also apply clipp technique as in the assignment.
        trainer = self.trainer
        gradients = trainer.compute_gradients(self.loss)
        gradients_clipped = [(tf.clip_by_value(t[0],-1,1),t[1]) for t in gradients]
        self.optimizer = trainer.apply_gradients(gradients_clipped)
        
        
    def setup_predict(self):
        # The predicting process is a bit different from training process. Here we need to put the note model into the time model.
        # In other words, we first get a time step output from the time model, then feed it to the full note model over note. After 
        # we have the output of the note model, we translate the note output to input, from note statematrix to input fomat, which should
        # have length 80 for each note. And we just keep doing things above to generate or predict a new music. And here we will used a
        # random sampled starter of a measure from the whole training set.
        
        # When training we have tensorflow function 'tf.nn.dynamic_rnn' to help us to feedforward, but here we don't have the whole data 
        # to feed forward in the time model. We first need to get the last time step note-model's output then we could keep going in the 
        # time model.Therefore, we use 'tf.scan' here to loop through both time and notes here.
        
        
        def _step_note(state,indata):
            # the function we loop through in note model
            # shape of state: ([100,50],2), indata shape: 300
            indata = tf.expand_dims(tf.concat((indata,state[1]),axis=-1),axis=0)
            hidden = state[0]
            
            # one note step forward
            note_output, new_state = self.n_multi_rnn_cell.call(inputs=indata,state=hidden) # note output shape: 50, new_state shape: 100 or 50 
            prob = tf.layers.dense(note_output,
                                     units=2,
                                     activation=tf.nn.sigmoid,
                                     name='output_layer',
                                     reuse=True)
            
            # sample the notes, each of them has a probability to be played and articulated, and a uniform distribution 
            # can simulate this. 
            randomness = tf.random_uniform(shape=(1,))
            shouldplay = tf.cast(randomness < (prob[0][0] * self.conservativity), tf.float32)
            shouldartic = shouldplay * tf.cast(tf.random_uniform(shape=(1,)) < prob[0][1], tf.float32)
            output = tf.concat([shouldplay,shouldartic],axis=-1)
            return (new_state,output)
        
        def _step_time(states,_):
            # the function we loop through in time model
            hidden = states[0] # shape: (notes*t_hidden)*2 e.g. (78*300)*2
            indata = states[1] # shape: notes*note_features e.g. 78*80
            time = states[2]
            
            # one time step forward 
            output, new_state = self.t_multi_rnn_cell.call(inputs=indata,state=hidden) # output shape: notes*t_hidden
            
            start_note_values = tf.zeros((2))
            
            n_initializer = (self.n_multi_rnn_cell.zero_state(1, tf.float32),
                             start_note_values)
            
            # loop through note model
            note_result = tf.scan(_step_note,elems=output,initializer=n_initializer) # note_result shape: 78*2
            
            # Here we define a tensor OP used 'tf.py_fun', to translate the note model's output to input format for 
            # time model
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
        # Since we are here do not need to loop through any variable but a specific steps(ticks), which define in
        # 'step_to_sumulate', we create a tensor for the tf.scan to loop through.
        elems = tf.zeros(shape=self.step_to_sumulate)
        
        time_result = tf.scan(_step_time,elems=elems,initializer=initializer)
        
        # the statematrix of the song we generate/predict.
        self.new_song = time_result[3]
        
    def train(self, pieces, 
              batch_size=10, 
              predict_freq=100,
              save_freq=500,
              show_freq=10,
              max_epoch=10000, 
              saveto='NewSong', 
              step=319, 
              conservativity=1,
              pre_trained_model=None):
        '''
        This is the train function for the biaxial_model. It alos predicts while trianing.
        Args:
            pieces: dict, containing the all note statematrixs as training set;
            batch_size: how many sample in one batch;
            predict_freq: int, every predict_freq we generate a new song;
            save_freq: int, the frequency to save the model;
            show_freq: int, the frequency to show the loss;
            max_epoch: max steps we gonna train;
            saveto: str, the dir we save or new songs;
            step: int, the length we generate a new song. one step means on step for the time model
            conservativity: The conservativity of number of notes of song we generate.
            pre_trained_model: used for restore training
        '''
        
        cur_model_name = 'biaxial_rnn_{}'.format(int(time.time()))
        batch_generator = data.generate_batch(pieces,batch_size)
        
        minloss = np.inf
        loss_log = []
        
        with tf.Session() as sess:
            
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            
            if pre_trained_model is not None:
                try:
                    print("Load the model from: {}".format(pre_trained_model))
                    saver.restore(sess, 'model/{}'.format(pre_trained_model))
                except Exception:
                    print("Load model Failed!")
                    pass
                
            for i in range(max_epoch):
                X_train, y_train = next(batch_generator)
                _, loss = sess.run((self.optimizer,self.loss), feed_dict={self.input_mat : X_train, self.output_mat : y_train})
                
                loss_log.append(loss)
                pickle.dump(loss_log,open('model/'+cur_model_name+'_loss_log.pkl','wb'))
                
                if i % show_freq == 0:
                    print('Step {0}: loss is {1}'.format(i + 1, loss))
                
                # generate a new song
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
                
                # save the models for restoring training
                if (i+1) % save_freq == 0:
                    if not os.path.exists('model/'):
                        os.makedirs('model/')
                    saver.save(sess, 'model/{}'.format(cur_model_name))
                    print('{} Saved'.format(cur_model_name))
                
                # save the good models
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
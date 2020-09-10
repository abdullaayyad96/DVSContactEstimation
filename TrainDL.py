import tensorflow.compat.v1 as tf
import tensorflow as tf2
import numpy as np 
from DataLoader import DataLoader

class TrainDL:

    def train_nn(sess, epochs, nn_last_layer, hidden_state, carry_state, batch_size, data_loader, accuracy_op, train_op, loss_function, input_tensor,
             truth_tensor, initial_hidden_state, initial_carry_state, learning_rate, base_learning_rate,
             learning_decay_rate, learning_decay_factor):
        """
        Train neural network and print    out the loss during training.
    param sess: TF Session
        :param epochs: Number of epochs
        :param batch_size: Batch size
        :param data_loader: Object of DataLoader type. Call using get_batches_fn(batch_size)
        :param train_op: TF Operation to train the neural network
        :param loss_function: TF Tensor for the amount of loss
        :param input_tensor: TF Placeholder for input images
        :param truth_tensor: TF Placeholder for truth value
        :param learning_rate: TF Placeholder for learning rate
        :param base_learning_rate: Float for the base learning rate of optimizer
        :param learning_decay_rate: Float for the period of dropping the learning rate
        :param learning_decay_factor: Float for decaying learning rate
        """
        #initialize variables
        sess.run(tf.global_variables_initializer())
        
        print("Training...")
        print()
        scaling_rate = 1
        
        loss_output = 0
        for i in range(epochs):
            loss_output = 0
            print("EPOCH {} ...".format(i+1))
            if i%learning_decay_rate == 0 and i != 0:
                scaling_rate = learning_decay_factor * scaling_rate
            j = 0
            sum_accuracy = 0
            for image, output, batch_i_size in data_loader.get_train_batches_fn_timeseries_sequence(batch_size):
                initial_state_value = np.zeros(shape=(batch_i_size, 29, 39, 20), dtype=float)
                
                nn_output, lstm_hidden_state, lstm_carry_state, optimizer, loss = sess.run([nn_last_layer, hidden_state, carry_state, train_op, loss_function], 
                                feed_dict={input_tensor: image, truth_tensor: output, initial_hidden_state: initial_state_value, initial_carry_state: initial_state_value, learning_rate: scaling_rate*base_learning_rate})
                
                
                #print(np.shape(lstm_hidden_state))
                #print(np.shape(lstm_carry_state))
                
                accuracy = sess.run([accuracy_op], feed_dict={input_tensor: image, truth_tensor: output, initial_hidden_state: initial_state_value, initial_carry_state: initial_state_value})
                                
                sum_accuracy = sum_accuracy + accuracy[0]
                j = j+1

            valid_x, valid_y, valid_size = data_loader.get_validation_data_sequence()
            initial_state_value = np.zeros(shape=(valid_size, 29, 39, 20), dtype=float)
            valid_accuracy = sess.run([accuracy_op],
                                feed_dict={input_tensor: valid_x, truth_tensor: valid_y, initial_hidden_state: initial_state_value, initial_carry_state: initial_state_value})
            print("Train Accuracy {} ...".format(sum_accuracy/j))
            print("Validation Accuracy {} ...".format(valid_accuracy))
                                
            
            

    def SoftEntropy(nn_last_layer, correct_label, learning_rate):
        """
        Build the TensorFLow loss and optimizer operations.
        :param nn_last_layer: TF Tensor of the last layer in the neural network
        :param correct_label: TF Placeholder for the truth images       
        :param learning_rate: TF Placeholder for the learning rate of optimizer
        :return: Tuple of (logits, train_op, cross_entropy_loss)
        """    
        
        loss = tf2.keras.backend.sum(tf2.nn.softmax_cross_entropy_with_logits(tf2.stop_gradient(correct_label), nn_last_layer))
        
        #obtain training operation
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, epsilon = 1e-8) #Note default value of epsilon 1e-8 results in instability after few epochs
    
        #clip the gradients
        gvs = optimizer.compute_gradients(loss)
        #capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        training_operation = optimizer.apply_gradients(gvs)

        return training_operation, loss
    
    
    def SoftEntropy_sequence(nn_last_layer, correct_label, learning_rate):
        """
        Build the TensorFLow loss and optimizer operations.
        :param nn_last_layer: TF Tensor of the last layer in the neural network
        :param correct_label: TF Placeholder for the truth images       
        :param learning_rate: TF Placeholder for the learning rate of optimizer
        :return: Tuple of (logits, train_op, cross_entropy_loss)
        """    
        
        loss = tf2.keras.backend.sum( tf2.keras.layers.TimeDistributed(tf2.nn.softmax_cross_entropy_with_logits(tf2.stop_gradient(correct_label), nn_last_layer)))
        
        #obtain training operation
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, epsilon = 1e-8) #Note default value of epsilon 1e-8 results in instability after few epochs
    
        #clip the gradients
        gvs = optimizer.compute_gradients(loss)
        #capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        training_operation = optimizer.apply_gradients(gvs)

        return training_operation, loss


    def Evaluate(nn_last_layer, correct_label):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples

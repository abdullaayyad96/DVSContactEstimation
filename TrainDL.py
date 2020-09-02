import tensorflow.compat.v1 as tf
import tensorflow as tf2
import numpy as np 
from DataLoader import DataLoader

class TrainDL:

    def train_nn(sess, epochs, batch_size, data_loader, accuracy_op, train_op, loss_function, input_tensor,
             truth_tensor, learning_rate, base_learning_rate,
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
            for image, output in data_loader.get_batches_fn_timeseries(batch_size):
                print("Batch {} ...".format(j+1))

                optimizer, loss = sess.run([train_op, loss_function], 
                                feed_dict={input_tensor: image, truth_tensor: output, learning_rate: scaling_rate*base_learning_rate})

                accuracy = sess.run([accuracy_op],
                                feed_dict={input_tensor: image, truth_tensor: output})

                print("Accuracy {} ...".format(accuracy))
                                
            
            

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


    def Evaluate(nn_last_layer, correct_label):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples

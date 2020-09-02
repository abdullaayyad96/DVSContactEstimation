import tensorflow.compat.v1 as tf
import tensorflow as tf2
import numpy as np 

class DlModels:
    def vgg8(input_tensor):
        #Layer 1: Convolutional
        conv1_1 = tf.layers.Conv2D(kernel_size=3, filters=64, strides=1, padding='same', 
                    kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                    kernel_regularizer=tf2.keras.regularizers.l2(), bias_regularizer=tf2.keras.regularizers.l2(), activation='relu',
                    name='conv1_1')(input_tensor)

        #pooling function
        pool_1 = tf.layers.MaxPooling2D(pool_size=2, strides=2, padding='same', name='pool_1')(conv1_1)

        #Layer 2: Convolutional
        conv2_1 = tf.layers.Conv2D(kernel_size=3, filters=128, strides=1, padding='same', 
                    kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                    kernel_regularizer=tf2.keras.regularizers.l2(), bias_regularizer=tf2.keras.regularizers.l2(), activation='relu',
                    name='conv2_1')(pool_1)


        #pooling function
        pool_2 = tf.layers.MaxPooling2D(pool_size=2, strides=2, padding='same', name='pool_2')(conv2_1)

        #Layer 3: Convolutional
        conv3_1 = tf.layers.Conv2D(kernel_size=3, filters=256, strides=1, padding='same', 
                    kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                    kernel_regularizer=tf2.keras.regularizers.l2(), bias_regularizer=tf2.keras.regularizers.l2(), activation='relu',
                    name='conv3_1')(pool_2)

        #pooling function
        pool_3 = tf.layers.MaxPooling2D(pool_size=2, strides=2, padding='same', name='pool_3')(conv3_1)


        #Layer 4: Convolutional
        conv4_1 = tf.layers.Conv2D(kernel_size=3, filters=512, strides=1, padding='same', 
                    kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                    kernel_regularizer=tf2.keras.regularizers.l2(), bias_regularizer=tf2.keras.regularizers.l2(), activation='relu',
                    name='conv4_1')(pool_3)

        #pooling function
        pool_4 = tf.layers.MaxPooling2D(pool_size=2, strides=2, padding='same', name='pool_4')(conv4_1)

        #Layer 5: Convolutional
        conv5_1 = tf.layers.Conv2D(kernel_size=3, filters=512, strides=1, padding='same', 
                    kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                    kernel_regularizer=tf2.keras.regularizers.l2(), bias_regularizer=tf2.keras.regularizers.l2(), activation='relu',
                    name='conv5_1')(pool_4)

        #pooling function
        pool_5 = tf.layers.MaxPooling2D(pool_size=2, strides=2, padding='same', name='pool_5')(conv5_1)

        return pool_5, conv5_1, conv4_1, conv3_1, conv2_1, conv1_1




    def vgg11(input_tensor):
        #Layer 1: Convolutional
        conv1_1 = tf.layers.Conv2D(kernel_size=3, filters=64, strides=1, padding='same', 
                    kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                    kernel_regularizer=tf2.keras.regularizers.l2(), bias_regularizer=tf2.keras.regularizers.l2(), activation='relu',
                    name='conv1_1')(input_tensor)

        #pooling function
        pool_1 = tf.layers.MaxPooling2D(pool_size=2, strides=2, padding='same', name='pool_1')(conv1_1)

        #Layer 2: Convolutional
        conv2_1 = tf.layers.Conv2D(kernel_size=3, filters=128, strides=1, padding='same', 
                    kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                    kernel_regularizer=tf2.keras.regularizers.l2(), bias_regularizer=tf2.keras.regularizers.l2(), activation='relu',
                    name='conv2_1')(pool_1)


        #pooling function
        pool_2 = tf.layers.MaxPooling2D(pool_size=2, strides=2, padding='same', name='pool_2')(conv2_1)

        #Layer 3: Convolutional
        conv3_1 = tf.layers.Conv2D(kernel_size=3, filters=256, strides=1, padding='same', 
                    kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                    kernel_regularizer=tf2.keras.regularizers.l2(), bias_regularizer=tf2.keras.regularizers.l2(), activation='relu',
                    name='conv3_1')(pool_2)

        #Layer 4: Convolutional
        conv3_2 = tf.layers.Conv2D(kernel_size=3, filters=256, strides=1, padding='same', 
                    kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                    kernel_regularizer=tf2.keras.regularizers.l2(), bias_regularizer=tf2.keras.regularizers.l2(), activation='relu',
                    name='conv3_2')(conv3_1)

        #pooling function
        pool_3 = tf.layers.MaxPooling2D(pool_size=2, strides=2, padding='same', name='pool_3')(conv3_2)


        #Layer 5: Convolutional
        conv4_1 = tf.layers.Conv2D(kernel_size=3, filters=512, strides=1, padding='same', 
                    kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                    kernel_regularizer=tf2.keras.regularizers.l2(), bias_regularizer=tf2.keras.regularizers.l2(), activation='relu',
                    name='conv4_1')(pool_3)

        #Layer 6: Convolutional
        conv4_2 = tf.layers.Conv2D(kernel_size=3, filters=512, strides=1, padding='same', 
                    kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                    kernel_regularizer=tf2.keras.regularizers.l2(), bias_regularizer=tf2.keras.regularizers.l2(), activation='relu',
                    name='conv4_2')(conv4_1)


        #pooling function
        pool_4 = tf.layers.MaxPooling2D(pool_size=2, strides=2, padding='same', name='pool_4')(conv4_2)

        #Layer 7: Convolutional
        conv5_1 = tf.layers.Conv2D(kernel_size=3, filters=512, strides=1, padding='same', 
                    kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                    kernel_regularizer=tf2.keras.regularizers.l2(), bias_regularizer=tf2.keras.regularizers.l2(), activation='relu',
                    name='conv5_1')(pool_4)

        #Layer 8: Convolutional
        conv5_2 = tf.layers.Conv2D(kernel_size=3, filters=512, strides=1, padding='same', 
                    kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                    kernel_regularizer=tf2.keras.regularizers.l2(), bias_regularizer=tf2.keras.regularizers.l2(), activation='relu',
                    name='conv5_2')(conv5_1)


        #pooling function
        pool_5 = tf.layers.MaxPooling2D(pool_size=2, strides=2, padding='same', name='pool_5')(conv5_2)

        return pool_5, conv5_2, conv4_2, conv3_2, conv2_1, conv1_1





    def vgg16(input_tensor):
        #Layer 1: Convolutional
        conv1_1 = tf.layers.Conv2D(kernel_size=5, filters=64, strides=1, padding='same', 
                kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                kernel_regularizer=tf2.keras.regularizers.l2(), bias_regularizer=tf2.keras.regularizers.l2(), activation='relu',
                name='conv1_1')(input_tensor)

        #Layer 2: Convolutional
        conv1_2 = tf.layers.Conv2D(kernel_size=5, filters=64, strides=1, padding='same', 
                    kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                    kernel_regularizer=tf2.keras.regularizers.l2(), bias_regularizer=tf2.keras.regularizers.l2(), activation='relu',
                    name='conv1_2')(conv1_1)
        

        #pooling function
        pool_1 = tf.layers.MaxPooling2D(pool_size=2, strides=2, padding='same', name='pool_1')(conv1_2)

        #Layer 3: Convolutional
        conv2_1 = tf.layers.Conv2D(kernel_size=4, filters=128, strides=1, padding='same', 
                    kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                    kernel_regularizer=tf2.keras.regularizers.l2(), bias_regularizer=tf2.keras.regularizers.l2(), activation='relu',
                    name='conv2_1')(pool_1)

        #Layer 4: Convolutional
        conv2_2 = tf.layers.Conv2D(kernel_size=4, filters=128, strides=1, padding='same', 
                    kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                    kernel_regularizer=tf2.keras.regularizers.l2(), bias_regularizer=tf2.keras.regularizers.l2(), activation='relu',
                    name='conv2_2')(conv2_1)

        #pooling function
        pool_2 = tf.layers.MaxPooling2D(pool_size=2, strides=2, padding='same', name='pool_2')(conv2_2)

        #Layer 5: Convolutional
        conv3_1 = tf.layers.Conv2D(kernel_size=3, filters=256, strides=1, padding='same', 
                    kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                    kernel_regularizer=tf2.keras.regularizers.l2(), bias_regularizer=tf2.keras.regularizers.l2(), activation='relu',
                    name='conv3_1')(pool_2)

        #Layer 6: Convolutional
        conv3_2 = tf.layers.Conv2D(kernel_size=3, filters=256, strides=1, padding='same', 
                    kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                    kernel_regularizer=tf2.keras.regularizers.l2(), bias_regularizer=tf2.keras.regularizers.l2(), activation='relu',
                    name='conv3_2')(conv3_1)

        #Layer 7: Convolutional
        conv3_3 = tf.layers.Conv2D(kernel_size=3, filters=256, strides=1, padding='same', 
                    kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                    kernel_regularizer=tf2.keras.regularizers.l2(), bias_regularizer=tf2.keras.regularizers.l2(), activation='relu',
                    name='conv3_3')(conv3_2)

        #pooling function
        pool_3 = tf.layers.MaxPooling2D(pool_size=2, strides=2, padding='same', name='pool_3')(conv3_3)


        #Layer 8: Convolutional
        conv4_1 = tf.layers.Conv2D(kernel_size=3, filters=512, strides=1, padding='same', 
                    kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                    kernel_regularizer=tf2.keras.regularizers.l2(), bias_regularizer=tf2.keras.regularizers.l2(), activation='relu',
                    name='conv4_1')(pool_3)

        #Layer 9: Convolutional
        conv4_2 = tf.layers.Conv2D(kernel_size=3, filters=512, strides=1, padding='same', 
                    kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                    kernel_regularizer=tf2.keras.regularizers.l2(), bias_regularizer=tf2.keras.regularizers.l2(), activation='relu',
                    name='conv4_2')(conv4_1)

        #Layer 10: Convolutional
        conv4_3 = tf.layers.Conv2D(kernel_size=3, filters=512, strides=1, padding='same', 
                    kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                    kernel_regularizer=tf2.keras.regularizers.l2(), bias_regularizer=tf2.keras.regularizers.l2(), activation='relu',
                    name='conv4_3')(conv4_2)

        #pooling function
        pool_4 = tf.layers.MaxPooling2D(pool_size=2, strides=2, padding='same', name='pool_4')(conv4_3)

        #Layer 11: Convolutional
        conv5_1 = tf.layers.Conv2D(kernel_size=2, filters=512, strides=1, padding='same', 
                    kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                    kernel_regularizer=tf2.keras.regularizers.l2(), bias_regularizer=tf2.keras.regularizers.l2(), activation='relu',
                    name='conv5_1')(pool_4)

        #Layer 12: Convolutional
        conv5_2 = tf.layers.Conv2D(kernel_size=2, filters=512, strides=1, padding='same', 
                    kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                    kernel_regularizer=tf2.keras.regularizers.l2(), bias_regularizer=tf2.keras.regularizers.l2(), activation='relu',
                    name='conv5_2')(conv5_1)

        #Layer 13: Convolutional
        conv5_3 = tf.layers.Conv2D(kernel_size=2, filters=512, strides=1, padding='same', 
                    kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                    kernel_regularizer=tf2.keras.regularizers.l2(), bias_regularizer=tf2.keras.regularizers.l2(), activation='relu',
                    name='conv5_3')(conv5_2)

        #pooling function
        pool_5 = tf.layers.MaxPooling2D(pool_size=2, strides=2, padding='same', name='pool_5')(conv5_3)
        
        return pool_5, conv5_3, conv4_3, conv3_3, conv2_2, conv1_2

    
    def Conv2Dx1(input_tensor):
        #Layer 1: Convolutional
        conv1_1 = tf2.keras.layers.TimeDistributed(tf.layers.Conv2D(kernel_size=5, filters=12, strides=1, padding='same', 
                    kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                    kernel_regularizer=tf2.keras.regularizers.l2(), bias_regularizer=tf2.keras.regularizers.l2(), activation='relu',
                    name='conv1_1'))(input_tensor)

        #pooling function
        pool_1 = tf2.keras.layers.TimeDistributed(tf.layers.MaxPooling2D(pool_size=3, strides=3, padding='same', name='pool_1'))(conv1_1)

        #Layer 2: Convolutional
        conv2_1 = tf2.keras.layers.TimeDistributed(tf.layers.Conv2D(kernel_size=5, filters=24, strides=1, padding='same', 
                    kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                    kernel_regularizer=tf2.keras.regularizers.l2(), bias_regularizer=tf2.keras.regularizers.l2(), activation='relu',
                    name='conv2_1'))(pool_1)

        #pooling function
        pool_2 = tf2.keras.layers.TimeDistributed(tf.layers.MaxPooling2D(pool_size=3, strides=3, padding='same', name='pool_2'))(conv2_1)

        return pool_2, conv2_1, conv1_1

    def vggDecoder(input_tensor, encoder):
        encoder_output, layer5, layer4, layer3, layer2, layer1 = encoder(input_tensor)

        #1x1 convolutions 1
        conv1x1_1 = tf.layers.Conv2D(kernel_size=1, filters=256, strides=1, padding='same',
                        kernel_initializer=tf2.keras.initializers.GlorotNormal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                        kernel_regularizer=tf2.keras.regularizers.l2(), activation='relu', 
                        name='conv1x1_1')(encoder_output)

        
        #1x1 convolutions 2
        conv1x1_2 = tf.layers.Conv2D(kernel_size=1, filters=128, strides=1, padding='same',
                        kernel_initializer=tf2.keras.initializers.GlorotNormal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                        kernel_regularizer=tf2.keras.regularizers.l2(), activation='relu',
                        name='conv1x1_2')(conv1x1_1)

        #1x1 convolutions 3
        conv1x1_3 = tf.layers.Conv2D(kernel_size=1, filters=64, strides=1, padding='same',
                        kernel_initializer=tf2.keras.initializers.GlorotNormal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                        kernel_regularizer=tf2.keras.regularizers.l2(), activation='relu',
                        name='conv1x1_3')(conv1x1_2)


        #Deconvolution 1
        deconv_1 = tf2.keras.layers.Conv2DTranspose(kernel_size=2, filters=32, strides=2, padding='same',
                        kernel_initializer=tf2.keras.initializers.GlorotNormal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                        kernel_regularizer=tf2.keras.regularizers.l2(), 
                        name='deconv_1')(conv1x1_3)
                        
        #Skipping layer 1
        skip_1 = tf2.keras.layers.Concatenate(name='skip_1')([deconv_1, layer5])

        #Deconvolution 2
        deconv_2 = tf2.keras.layers.Conv2DTranspose(kernel_size=3, filters=32, strides=2, padding='same',
                        kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                        kernel_regularizer=tf2.keras.regularizers.l2(), 
                        name='deconv_2')(skip_1)    
                        
        #Skipping layer 2
        skip_2 = tf2.keras.layers.Concatenate(name='skip_2')([deconv_2, layer4])

        #Deconvolution 3
        deconv_3 = tf2.keras.layers.Conv2DTranspose(kernel_size=4, filters=16, strides=2, padding='same',
                        kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                        kernel_regularizer=tf2.keras.regularizers.l2(),
                        name='deconv_3')(skip_2)

        
        #Skipping layer 3
        skip_3 = tf2.keras.layers.Concatenate(name='skip_3')([deconv_3, layer3])

        #Deconvolution 4
        deconv_4 = tf2.keras.layers.Conv2DTranspose(kernel_size=4, filters=8, strides=2, padding='same',
                        kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                        kernel_regularizer=tf2.keras.regularizers.l2(),
                        name='deconv_4')(skip_3)
        
        #Skipping layer 4
        skip_4 = tf2.keras.layers.Concatenate(name='skip_4')([deconv_4, layer2])


        #Deconvolution 5
        deconv_5 = tf2.keras.layers.Conv2DTranspose(kernel_size=5, filters=8, strides=2, padding='same',
                        kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                        kernel_regularizer=tf2.keras.regularizers.l2(),
                        name='deconv_5')(skip_4)
        
        #Skipping layer 5
        skip_5 = tf2.keras.layers.Concatenate(name='skip_5')([deconv_5, layer1])


        #Final layers
        output_layer = tf.layers.Conv2D(kernel_size=5, filters=1, strides=1, padding='same',
                        kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                        kernel_regularizer=tf2.keras.regularizers.l2(), 
                        name='output_layer')(skip_5)


        return output_layer


    def Conv2dx2LSTMDecoder(input_tensor, encoder=Conv2Dx1):

        encoder_output, layer2, layer1 = encoder(input_tensor)

        #Convolutional LSTM
        LSTMLayer = tf2.keras.layers.ConvLSTM2D(filters = 64, kernel_size=2, padding='same', activation='relu')(encoder_output)
        
        #Deconvolution 1
        deconv_1 = tf2.keras.layers.Conv2DTranspose(kernel_size=2, filters=32, strides=2, padding='same',
                        kernel_initializer=tf2.keras.initializers.GlorotNormal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                        kernel_regularizer=tf2.keras.regularizers.l2(), 
                        name='deconv_1')(LSTMLayer)

        #Skipping layer 1
        skip_1 = tf2.keras.layers.Concatenate(name='skip_1')([deconv_1, layer2[:, -1, :, :, :]])

        #Deconvolution 2
        deconv_2 = tf2.keras.layers.Conv2DTranspose(kernel_size=3, filters=32, strides=2, padding='same',
                        kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                        kernel_regularizer=tf2.keras.regularizers.l2(), 
                        name='deconv_2')(skip_1)    
                        
        #Skipping layer 2
        skip_2 = tf2.keras.layers.Concatenate(name='skip_2')([deconv_2, layer1[:, -1, :, :, :]])

        #Final layers
        output_layer = tf.layers.Conv2D(kernel_size=5, filters=1, strides=1, padding='same',
                        kernel_initializer=tf2.keras.initializers.Orthogonal(), bias_initializer=tf2.keras.initializers.GlorotNormal(),
                        kernel_regularizer=tf2.keras.regularizers.l2(), 
                        name='output_layer')(skip_2)


        return output_layer


    def Conv2dx2LSTMFC(input_tensor, encoder=Conv2Dx1, N_outputs=6):

        encoder_output, layer2, layer1 = encoder(input_tensor)

        #Convolutional LSTM
        LSTMLayer = tf2.keras.layers.ConvLSTM2D(filters = 24, kernel_size=5, padding='same', activation='relu', return_sequences=False)(encoder_output)    

        #pooling function
        pool = tf.layers.MaxPooling2D(pool_size=3, strides=3, padding='same', name='pool')(LSTMLayer) 

        flat =tf2.keras.layers.Flatten()(pool)

        output_layer = tf2.keras.layers.Dense(N_outputs)(flat)

        return output_layer


import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
import matplotlib.pyplot as plt
import csv
import cv2
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def prep_crop(img, crop_top, crop_bottom):
    #crop the img from top to bottom
    return img[crop_top:-crop_bottom,:,:]

def preprocessing(img):
    #crop the top 50 and bottom 20
    img = prep_crop(img, 50, 20)

    #resize the img to (66,200)
    img = cv2.resize(img, (200, 66), interpolation=cv2.INTER_AREA)

    return img

###generator function
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_angle = float(batch_sample[3])

                correction = 0.3
                left_angle = center_angle + correction
                right_angle = center_angle - correction
                #name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = preprocessing(cv2.imread('./IMG/' + batch_sample[0].split('/')[-1]))
                left_image = preprocessing(cv2.imread('./IMG/' + batch_sample[1].split('/')[-1]))
                right_image = preprocessing(cv2.imread('./IMG/' + batch_sample[2].split('/')[-1]))

                #center_image = preprocessing(center_image)
                images.extend([center_image, left_image, right_image])
                angles.extend([center_angle, left_angle, right_angle])
                #images.extend([cv2.flip(center_image, 1), cv2.flip(left_image, 1), cv2.flip(right_image, 1)])
                #angles.extend([center_angle*-1.0, left_angle*-1.0, right_angle*-1.0])
                images.append(cv2.flip(center_image, 1))
                angles.append(center_angle * -1.0)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def NvidiaNet():
    '''
    According the paper <End to End Learning for Self-Driving Cars> from NVIDIA Corporation.
    The network consists of 9 layers, including a normalization layer, 5 convolutional layers 
    and 3 fully connected layers. 
    :return: model
    '''
    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(66, 200, 3)))
    #Add three 5x5 convolutional layers, stride=(2,2)
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    #Add two 3x3 convolutional layers, stride=(1,1)
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    #Add a flatten layer
    model.add(Flatten())
    #Add three fully connected layers
    model.add(Dense(100, W_regularizer=l2(0.001)))
    model.add(ELU())
    #model.add(Dropout(0.5))
    model.add(Dense(50, W_regularizer=l2(0.001)))
    model.add(ELU())
    #model.add(Dropout(0.5))
    model.add(Dense(10, W_regularizer=l2(0.001)))
    model.add(ELU())
    #model.add(Dropout(0.5))
    #Add a fully connected output layer
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model

if __name__ == "__main__":

    #read the csvfile to samples
    samples = []
    with open('./driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    samples.pop(0)

    #split the train dataset(0.8) and validation dataset(0.2)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    print("The count of train samples:", len(train_samples))
    print("The count of validation samples:", len(validation_samples))

    #call the generator function
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    model = NvidiaNet()

    history_object = model.fit_generator(train_generator,
                                         samples_per_epoch = len(train_samples)*4,
                                         validation_data = validation_generator,
                                         nb_val_samples = len(validation_samples)*4,
                                         nb_epoch = 5,
                                         verbose = 1)

    ### save the model
    model.save('model.h5')

    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()






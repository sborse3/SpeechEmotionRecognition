
# coding: utf-8

# ## Importing the required libraries
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
from matplotlib.pyplot import specgram
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras import regularizers
from sklearn.utils import shuffle
import os
import pandas as pd
import librosa
import glob 
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import os
import scipy.io.wavfile
import numpy as np
import sys



mylist= os.listdir('RawData/')



# ## Plotting the audio file's waveform and its spectrogram
data, sampling_rate = librosa.load('RawData/f11 (2).wav')
sr,x = scipy.io.wavfile.read('RawData/f10 (2).wav')

## Parameters: 10ms step, 30ms window
nstep = int(sr * 0.01)
nwin  = int(sr * 0.03)
nfft = nwin
window = np.hamming(nwin)
nn = range(nwin, len(x), nstep)

# ## Setting the labels

feeling_list=[]

for item in mylist:
    if item[6:-16]=='02' and int(item[18:-4])%2==0:
        feeling_list.append('calm')
    elif item[6:-16]=='02' and int(item[18:-4])%2==1:
        feeling_list.append('calm')
    elif item[6:-16]=='03' and int(item[18:-4])%2==0:
        feeling_list.append('happy')
    elif item[6:-16]=='03' and int(item[18:-4])%2==1:
        feeling_list.append('happy')
    elif item[6:-16]=='04' and int(item[18:-4])%2==0:
        feeling_list.append('sad')
    elif item[6:-16]=='04' and int(item[18:-4])%2==1:
        feeling_list.append('sad')
    elif item[6:-16]=='05' and int(item[18:-4])%2==0:
        feeling_list.append('angry')
    elif item[6:-16]=='05' and int(item[18:-4])%2==1:
        feeling_list.append('angry')
    elif item[6:-16]=='06' and int(item[18:-4])%2==0:
        feeling_list.append('fearful')
    elif item[6:-16]=='06' and int(item[18:-4])%2==1:
        feeling_list.append('fearful')
    elif item[:1]=='a':
        feeling_list.append('angry')
    elif item[:1]=='f':
        feeling_list.append('fearful')
    elif item[:1]=='h':
        feeling_list.append('happy')
    elif item[:2]=='sa':
        feeling_list.append('sad')


labels = pd.DataFrame(feeling_list)
labels[:10]


# ## Getting the features of audio files using librosa

df = pd.DataFrame(columns=['feature'])
bookmark=0

for index,y in enumerate(mylist):
    if mylist[index][6:-16]!='01' and mylist[index][6:-16]!='07' and mylist[index][6:-16]!='08' and mylist[index][:2]!='su' and mylist[index][:1]!='n' and mylist[index][:1]!='d':
        X, sample_rate = librosa.load('RawData/'+y, res_type='kaiser_fast',duration=3,sr=22050,offset=0.5)
        sample_rate = np.array(sample_rate)
        centroids = (librosa.feature.spectral_centroid(y=X, 
                                            sr=sample_rate, 
                                            n_fft=512))
        
        mfccs = np.matrix(librosa.feature.mfcc(y=X, 
                                            sr=sample_rate, 
                                            n_mfcc=40))
                          
        centroids = (sklearn.preprocessing.normalize(centroids))
        mfccs = (sklearn.preprocessing.normalize(mfccs))

        
        #feature = (np.hstack((mfccs,centroids)))
        #rint (np.shape(feature[0]))
        #[float(i) for i in feature]
        #feature1=feature[30:]
        #df.loc[bookmark] = [feature[0]]
        df.loc[bookmark] = [mfccs.ravel()]
        bookmark=bookmark+1


df3 = pd.DataFrame(df['feature'].values.tolist())
newdf = pd.concat([df3,labels], axis=1)
rnewdf = newdf.rename(index=str, columns={"0": "label"})

rnewdf = shuffle(newdf)
rnewdf=rnewdf.fillna(0)


# ## Dividing the data into test and train

newdf1 = np.random.rand(len(rnewdf)) < 0.8
train = rnewdf[newdf1]
test = rnewdf[~newdf1]

trainfeatures = train.iloc[:, :-1]
trainlabel = train.iloc[:, -1:]

testfeatures = test.iloc[:, :-1]
testlabel = test.iloc[:, -1:]

X_train = np.array(trainfeatures)
y_train = np.array(trainlabel)
X_test = np.array(testfeatures)
y_test = np.array(testlabel)

lb = LabelEncoder()

y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))


[r,c] = X_train.shape


# ## Padding sequence for CNN model

print('Pad sequences')
x_traincnn =np.expand_dims(X_train, axis=2)
x_testcnn= np.expand_dims(X_test, axis=2)

np.save('x_test',x_testcnn)
np.save('y_test',y_test)

model = Sequential()

model.add(Conv1D(32, 5,padding='same',
                 input_shape=(c,1)))
model.add(Activation('relu'))
model.add(Conv1D(32, 10,padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=(4)))
model.add(Conv1D(64, 5,padding='same',))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Conv1D(64, 10,padding='same',))
model.add(Activation('relu'))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(MaxPooling1D(pool_size=(4)))
model.add(Conv1D(128, 10,padding='same',))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 15,padding='same',))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(5))
model.add(Activation('softmax'))
opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)


print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

cnnhistory=model.fit(x_traincnn, y_train, batch_size=32, epochs=1010, validation_data=(x_testcnn, y_test))


# ## Plotting the accuracy and loss graph

#sigmoid
plt.plot(cnnhistory.history['acc'])
plt.plot(cnnhistory.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ## Saving the model

model_name = 'Emotion_Voice_Detection_Model.h5'
save_dir = os.path.join(os.getcwd(), 'saved_models')
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)



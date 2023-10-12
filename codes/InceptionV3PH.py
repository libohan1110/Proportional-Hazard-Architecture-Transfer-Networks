
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import tensorflow.keras.backend as K
from PIL import Image
import cv2
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import deepsurvk
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import filedialog
import lifelines

#导入图像
root = tk.Tk()
root.withdraw()
Path1 = filedialog.askopenfilename(initialdir = "/", title = "Please enter the image of the uterine cavity", filetypes = (("All files", "*.*"),("Text files", "*.txt")))
Path2 = filedialog.askopenfilename(initialdir = "/", title = "Please enter the image of the uterine corner", filetypes = (("All files", "*.*"),("Text files", "*.txt")))
img_1 = image.load_img(Path1,target_size=(336,336))
img_1 = np.array(img_1)
img_2 = image.load_img(Path2,target_size=(336,336))
img_2 = np.array(img_2)
img = np.hstack((img_1,img_2))
img = img.reshape(1,336,672,3)
train_x_ = img.astype("float32") * 1/255
train_x_ = train_x_[:,::2,::2]


#载入模型
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
model = keras.models.load_model('InceptionV3.h5')
input = train_x_
last_conv_layer = model.get_layer('dense')
iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
model_out, last_conv_layer = iterate(input)
outcomes = np.array(last_conv_layer)
outcomes =pd.DataFrame(outcomes)



conv_base = keras.models.load_model('InceptionV3conv_base.h5')
with tf.GradientTape() as tape:
        last_conv_layer = conv_base.get_layer('conv2d_4')
        iterate = tf.keras.models.Model([conv_base.inputs], [conv_base.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(input)
        grads = tape.gradient(model_out, last_conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
heatmap = 3*heatmap
heatmap[heatmap>1]=1
heatmap = heatmap.reshape((38,80))


INTENSITY = 0.5

raw =train_x_*255
#raw = raw.resize(168, 336)
heatmap = cv2.resize(heatmap, (336, 168))
if model.predict (input)>0.5:
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_WINTER)
else:
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)


img_combine = ((heatmap + np.array(raw)) * INTENSITY)/255

#比例风险
X_train=np.load('X_train.npy')
Y_train=np.load('Y_train.npy')
E_train=np.load('E_train.npy')
X_train=pd.DataFrame(X_train)
Y_train=pd.DataFrame(Y_train)
E_train=pd.DataFrame(E_train)
n_patients_train = X_train.shape[0]
n_features = X_train.shape[1]


cols_standardize = X_train.iloc[:,0:256].columns
X_ct = ColumnTransformer([('standardizer', StandardScaler(), cols_standardize)])
X_ct.fit(X_train[cols_standardize])
X_train[cols_standardize] = X_ct.transform(X_train[cols_standardize])
outcomes[cols_standardize] = X_ct.transform(outcomes[cols_standardize])


sort_idx = np.argsort(Y_train.to_numpy(), axis=None)[::-1]
X_train = X_train.iloc[sort_idx, :]
Y_train = Y_train.iloc[sort_idx, :]
E_train = E_train.iloc[sort_idx, :]


params = {'n_layers':2,
          'n_nodes':8,
          'activation':'selu',
          'learning_rate':0.154,
          'decays':5.667e-3,
          'momentum':0.887,
          'dropout':0.5,
          'optimizer':'nadam'}
dsk = deepsurvk.DeepSurvK(n_features=n_features,
                          E=E_train,
                          **params)
loss = deepsurvk.negative_log_likelihood(E_train)
dsk.compile(loss=loss)
callbacks = deepsurvk.common_callbacks()

epochs = 1000
history = dsk.fit(X_train, Y_train,
                  batch_size=n_patients_train,
                  epochs=epochs,
                  shuffle=False)
Y_pred_train = np.exp(-dsk.predict(X_train))
Y_pred_train2=Y_pred_train[:,0]
time = np.array(Y_train.iloc[:,0])
event = np.array(E_train.iloc[:,0])
# 创建 Cox 比例风险模型并拟合数据
cox_model = lifelines.CoxPHFitter()
cox_model.fit(pd.DataFrame({
    'time': time,
    'event': event,
    'Y_pred_train': Y_pred_train2
}),duration_col='time', event_col='event', show_progress=True)
# 预测当时间为6，Y_pred_test=0.8 时发生事件的概率
t1 = 12
t2 = 24
Y_pred_test = np.exp(-dsk.predict(outcomes))  # 假设 Y_pred_test 为 0.8
surv_func = cox_model.predict_survival_function(Y_pred_test)
prob_event1 = 1 - surv_func.iloc[t1][0]
prob_event2 = 1 - surv_func.iloc[t2][0]
prob_event1= round(prob_event1,3)
prob_event2= round(prob_event2,3)


img_combine=img_combine.reshape(168, 336, 3)


words2 = "Conception Probability Within 1 year: "+str(prob_event1)
words3= "Conception Probability Within 2 year: "+str(prob_event2)
img_combine=cv2.resize(img_combine, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
cv2.putText(img_combine,words2, (10,280), cv2.FONT_HERSHEY_SIMPLEX,
0.6,(255,255,255), 2, cv2.LINE_AA)
cv2.putText(img_combine,words3, (10,300), cv2.FONT_HERSHEY_SIMPLEX,
0.6,(255,255,255), 2, cv2.LINE_AA)
#cv2.imwrite("predict_result.jpg", img_combine)
cv2.imshow("predict result",img_combine)
cv2.waitKey()





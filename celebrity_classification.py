import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import classification_report

#loading data


root_dir = r"D:\SEM 3\DL\DL-ALGORITHMS\Celebrity_Img_classification\cropped"
celebrities=os.listdir(root_dir)

celebrity_file=[]

for i, celebrity_name in tqdm(enumerate(celebrities), desc="Loading Data"):
    celebrity_path = os.path.join(root_dir, celebrity_name)
    celebrity_file.append(celebrity_path)
    celebrity_images = os.listdir(celebrity_path)
print((celebrity_file))



dataset=[]
label=[]
img_siz=(128,128)
categories = ["Lionel Messi", "Maria Sharapova", "Roger Federer", "Serena Williams", "Virat Kohli"]
for category, celebrity_path in zip(categories, celebrity_file):
    celebrity_images = os.listdir(celebrity_path)
    for i, image_name in tqdm(enumerate(celebrity_images), desc=category):
        if image_name.split('.')[1] == 'png':
            image_path = os.path.join(celebrity_path, image_name)
            image = cv2.imread(image_path)
            image = Image.fromarray(image, 'RGB')
            image = image.resize(img_siz)
            dataset.append(np.array(image))
            label.append(category.lower().replace(" ", "_"))
print(label)

dataset=np.array(dataset)
label = np.array(label)

print("--------------------------------------\n")
print('Dataset Length: ',len(dataset))
print('Label Length: ',len(label))
print("--------------------------------------\n")

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
Y = encoder.fit_transform(label)


print("--------------------------------------\n")
print("Train-Test Split")
x_train,x_test,y_train,y_test=train_test_split(dataset,Y,test_size=0.25,random_state=42, stratify=Y)
print("--------------------------------------\n")

print("--------------------------------------\n")
print("Normalising the Dataset. \n")

x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)

shape = x_train.shape[1:]
print(shape)
print("--------------------------------------\n")

#modeling

model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=shape),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5,activation='softmax')
])

print("--------------------------------------\n")
model.summary()
print("--------------------------------------\n")

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])


print("--------------------------------------\n")
print("Training Started.\n")
history=model.fit(x_train,y_train,epochs=10,batch_size=32,validation_split=0.1)
print("Training Finished.\n")
print("--------------------------------------\n")

# Plot and save accuracy
plt.plot(history.epoch,history.history['accuracy'], label='accuracy')
plt.plot(history.epoch,history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig(r'D:\SEM 3\DL\DL-ALGORITHMS\Celebrity_Img_classification\accuracy_plot.png')

plt.clf()

#plotting the loss
plt.plot(history.epoch,history.history['loss'], label='loss')
plt.plot(history.epoch,history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig(r'D:\SEM 3\DL\DL-ALGORITHMS\Celebrity_Img_classification\sample_loss_plot.png')


print("--------------------------------------\n")
print("Model Evaluation Phase.\n")
loss,accuracy=model.evaluate(x_test,y_test)
print(f'Accuracy: {round(accuracy*100,2)}')
print("--------------------------------------\n")
y_pred=model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print('classification Report\n', classification_report(y_test,y_pred_classes))
print("--------------------------------------\n")


def make_prediction(img, model, celebrities):
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img = Image.fromarray(img)
    img = img.resize((128, 128))
    img = np.array(img)
    input_img = np.expand_dims(img, axis=0)
    input_img = tf.keras.utils.normalize(input_img, axis=1) 
    predictions = model.predict(input_img)
    predicted_class = np.argmax(predictions)
    celebrity_name = celebrities[predicted_class]
    print(f"Predicted Celebrity: {celebrity_name}")

model.save('celebrity_model.h5')

make_prediction(r'D:\SEM 3\DL\DL-ALGORITHMS\Celebrity_Img_classification\cropped\lionel_messi\lionel_messi36.png',model,celebrities)
print("--------------------------------------\n")
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
import matplotlib.pyplot as plt
from PIL import Image

# img = plt.imread('C:/Users/Shayan/Desktop/Dog vs. Cat/test1/2.jpg')
# plt.imshow(img)
# plt.show()

img = Image.open('C:/Users/Shayan/Desktop/Dog vs. Cat/test1/2.jpeg')
model = load_model('DogsVsCats.h5')

model = Model(model)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])
model.predict(img)

# images = []
# for img in os.listdir('C:/Users/Shayan/Desktop/Dog vs. Cat/test1'):
#     img = os.path.join('C:/Users/Shayan/Desktop/Dog vs. Cat/test1', img)
#     img = image.load_img(img, target_size=(150, 150))
#     img = image.img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     images.append(img)
#
# # stack up images list to pass for prediction
# images = np.vstack(images)
# classes = model.predict_classes(images, batch_size=10)
# print(classes)

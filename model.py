import numpy as np
import platform
import csv
import cv2
import math
import os
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Lambda, SpatialDropout2D, Flatten
from keras.layers import Convolution2D, Cropping2D, Input
from keras.optimizers import Adam

DATA_PATH = "./data/"
CSV_FILE_PATH = DATA_PATH + "driving_log.csv"

BATCH_SIZE = 128
EPOCHS = 6
LEARNING_RATE = 0.0001
FLAG_DEBUGING = False

# Column Indices for csv data
CENTER = 0
LEFT = 1
RIGHT = 2
STEERING = 3
THROTTLE = 4
BRAKE = 5
SPEED = 6

def load_img(imgPath):
    """
    Load image from file to memory
    """

    base_name = os.path.basename(imgPath)
    imgPath = DATA_PATH + "IMG/" + base_name

    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img)
    # print("old img shape:", img.shape)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = img[60:126, 75:275, :] # crop height/width -> reshaped img to (66, 200, :)
    # # img = tf.image.resize_image_with_crop_or_pad(image=img, target_height=60, target_width=320)
    # print("new img shape:", img.shape)

    if FLAG_DEBUGING:
        display_img(img)

    return img


def display_img(img):
    plt.imshow(img)
    plt.show()
    exit("exiting display_img")


def import_csv_data():
    """
    Import the CSV data from file to memory
    Extract images (center, left, right) and steering angles
    """

    images = []
    steering_angles = []
    correction_factor = 0.1

    with open(CSV_FILE_PATH) as CSVFile:
        reader = csv.reader(CSVFile)
        for index, line in enumerate(reader):
            # Avoid first index/line since it contains column headers e.g. center, left, right etc.
            # Also avoid near zero steering angles
            if index != 0 and not math.isclose(float(line[STEERING]), 0.0):
                img_center = load_img(imgPath=str(line[CENTER]).strip())
                img_left = load_img(imgPath=str(line[LEFT]).strip())
                img_right = load_img(imgPath=str(line[RIGHT]).strip())
                steering_center = float(line[STEERING])
                steering_left = steering_center + correction_factor  # steer a bit to the right
                steering_right = steering_center - correction_factor  # steer a bit to the left

                # Append corresponding (same index) images and steering angles
                images.append(img_center)
                images.append(img_left)
                images.append(img_right)
                steering_angles.append(steering_center)
                steering_angles.append(steering_left)
                steering_angles.append(steering_right)

    print("import_csv_data returned successfully")
    return np.array(images), np.array(steering_angles)


def augment_imported_data(images, steering_angles):
    """
    Augument the data to balance the training data
    Avoiding overfitting the model to learn how to turn in mostly one direction
    """

    augmented_images = []
    augmented_steering_angles = []

    for img, steering in zip(images, steering_angles):

        # if steering == 0.0:
        #     exit("steering == 0.0 should never happen")

        # Note - # Append corresponding (same index) image and steering angle
        augmented_images.append(img)
        augmented_steering_angles.append(steering)

        # Note - # Append corresponding (same index) flipped image and steering angle
        flipped_image = cv2.flip(img, flipCode=1)  # flipCode-> 1 is y-axis and 0 is x-axis
        flipped_steering = steering * -1.0

        augmented_images.append(flipped_image)
        augmented_steering_angles.append(flipped_steering)

    print("augment_imported_data returned successfully")
    return np.array(augmented_images), np.array(augmented_steering_angles)


def nvidia_model(img):
    """
    Model from Nvidia
    http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    """

    shape = img.shape

    model = Sequential()

    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=shape))

    model.add(Cropping2D(cropping=((70, 24), (60, 60))))  # crop off 70px top, 25px bottom and 60px off left/right

    model.add(Convolution2D(24, 5, 5, border_mode="same", subsample=(2, 2), activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(36, 5, 5, border_mode="same", subsample=(2, 2), activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(48, 5, 5, border_mode="valid", subsample=(2, 2), activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
    model.add(SpatialDropout2D(0.2))

    model.add(Flatten())
    # model.add(Dropout(0.25))
    model.add(Dense(100, activation="elu"))
    model.add(Dense(50, activation="elu"))
    model.add(Dense(10, activation="elu"))
    # model.add(Dropout(0.25))
    model.add(Dense(1))

    print("nvidia model returned successfully\n")

    return model


#  ..................................................................
#  Import data and extract (images, steering_angles)
#  ..................................................................

images, steering_angles = import_csv_data()
augmented_images, augmented_steering_angles = augment_imported_data(images, steering_angles)

print("Non-augmented data: {}: {}".format(len(images), len(steering_angles)))
print("Augmented data: {}: {}".format(len(augmented_images), len(augmented_steering_angles)))


#  ..................................................................
#  Get the Model
#  ..................................................................

model = nvidia_model(images[0])
model.summary()

#  ..................................................................
#  Training the model
#  ..................................................................

X_train = augmented_images
Y_train = augmented_steering_angles

model.compile(optimizer=Adam(lr=LEARNING_RATE), loss='mean_squared_error')
model.fit(X_train, Y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_split=0.20, shuffle=True)

model.save('model.h5')
print("\nTraining completed. Model saved as model.h5")

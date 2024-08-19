# Data preprocess
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#

# Define the base directory
base_dir = '/Users/colleenjung/Downloads/Image_data/Raw_material/Pictures of Damage/Transit Damage'

# Data generators
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

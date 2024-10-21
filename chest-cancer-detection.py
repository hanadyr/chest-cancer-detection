import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from PIL import UnidentifiedImageError
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.layers import Dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.callbacks import EarlyStopping

#path to the dataset
base_dir = '/Users/hanady/Desktop/Data'

train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

#data exploration

    #class names
print("These are the class names in training data: ", os.listdir(train_dir), "\nThese are the class names in test data: ", os.listdir(test_dir), "\nThese are the class names in valid data: ", os.listdir(valid_dir))

    #corrupted image check
def check_corrupted_images(directory):
    corrupted_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img.verify()  #verify that image can be opened
            except (IOError, UnidentifiedImageError):
                print(f"Corrupted file found: {file_path}")
                corrupted_files.append(file_path)
    return corrupted_files

corrupted_images = check_corrupted_images(train_dir) #train data
print(f"Corrupted images in training set: {corrupted_images}")

corrupted_images = check_corrupted_images(valid_dir) #valid data
print(f"Corrupted images in valid set: {corrupted_images}")

corrupted_images = check_corrupted_images(test_dir) #test data
print(f"Corrupted images in test set: {corrupted_images}")

 #statistical summary of dataset
def image_statistics(directory): #function to get statistics about image sizes
    widths = []
    heights = []
    valid_extensions = ('.jpg', '.jpeg', '.png') #only allow valid image file types
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(valid_extensions): #skip non-image files
                img = Image.open(os.path.join(root, file))
                widths.append(img.size[0])
                heights.append(img.size[1])
    
    print(f"Mean width: {np.mean(widths)}")
    print(f"Mean height: {np.mean(heights)}")
    print(f"Std width: {np.std(widths)}")
    print(f"Std height: {np.std(heights)}")

image_statistics(train_dir) #stats for train set
image_statistics(valid_dir) #stats for valid set
image_statistics(test_dir) #stats for test set

    #data distribution per class
test_class_counts = {class_name: len(os.listdir(os.path.join(test_dir, class_name))) 
                     for class_name in os.listdir(test_dir) 
                     if os.path.isdir(os.path.join(test_dir, class_name))}
train_class_counts = {class_name: len(os.listdir(os.path.join(train_dir, class_name))) 
                      for class_name in os.listdir(train_dir) 
                      if os.path.isdir(os.path.join(train_dir, class_name))}
valid_class_counts = {class_name: len(os.listdir(os.path.join(valid_dir, class_name))) 
                      for class_name in os.listdir(valid_dir) 
                      if os.path.isdir(os.path.join(valid_dir, class_name))}

    #plot the distribution
def plot_distribution(counts, dataset_type):
    plt.figure(figsize=(10,5))
    plt.bar(counts.keys(), counts.values())
    plt.title(f'Image Distribution in {dataset_type} Set')
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.tight_layout()
    plt.show()

plot_distribution(train_class_counts, 'Train')
plot_distribution(valid_class_counts, 'Valid')
plot_distribution(test_class_counts, 'Test')

    #image size and aspect ratio analysis
def check_image_sizes(directory):
    image_sizes = []
    valid_extensions = ('.jpg', '.jpeg', '.png')
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(valid_extensions):  #only process valid image files
                image_path = os.path.join(root, file)
                try:
                    img = Image.open(image_path)
                    image_sizes.append(img.size)
                except (IOError, UnidentifiedImageError):
                    print(f"Cannot identify image file: {image_path}")
    return image_sizes

train_image_sizes = check_image_sizes(train_dir)
valid_image_sizes = check_image_sizes(valid_dir)
test_image_sizes = check_image_sizes(test_dir)

    #convert to numpy array for easier analysis
train_image_sizes = np.array(train_image_sizes)
valid_image_sizes = np.array(train_image_sizes)
test_image_sizes = np.array(train_image_sizes)

    #print unique image sizes or plot distribution
print("Average image size in Train Data:", np.mean(train_image_sizes, axis=0))
print("Average image size in Valid Data:", np.mean(valid_image_sizes, axis=0))
print("Average image size in Test Data:", np.mean(test_image_sizes, axis=0))

    #plot image size distribution
plt.figure(figsize=(10, 5)) #training data
plt.scatter(train_image_sizes[:, 0], train_image_sizes[:, 1], alpha=0.5)
plt.title("Image Size Distribution in Training Set")
plt.xlabel("Width")
plt.ylabel("Height")
plt.show()

plt.figure(figsize=(10, 5)) #valid data
plt.scatter(valid_image_sizes[:, 0], valid_image_sizes[:, 1], alpha=0.5)
plt.title("Image Size Distribution in Valid Set")
plt.xlabel("Width")
plt.ylabel("Height")
plt.show()

plt.figure(figsize=(10, 5)) #test data
plt.scatter(test_image_sizes[:, 0], test_image_sizes[:, 1], alpha=0.5)
plt.title("Image Size Distribution in Test Set")
plt.xlabel("Width")
plt.ylabel("Height")
plt.show()

#data augmentation

    #rename class folders in test set
test_dir = '/Users/hanady/Desktop/Data/test' #path to test directory

    #define mapping between current test folder names and desired train/valid names
class_name_mapping = {
    'squamous.cell.carcinoma': 'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa',
    'adenocarcinoma': 'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib',
    'large.cell.carcinoma': 'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa'
    #'normal' folder is the same, so no need to map it
}

    #rename class folders in test set
for old_name, new_name in class_name_mapping.items():
    old_path = os.path.join(test_dir, old_name)
    new_path = os.path.join(test_dir, new_name)
    
    #check if old folder exists, then rename it
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        print(f"Renamed '{old_name}' to '{new_name}'")
    else:
        print(f"Folder '{old_name}' not found in test directory.")
        
    #verify changes
print("Updated class names in test data: ", os.listdir(test_dir))

    #remove corrupted files
ds_store_path = os.path.join(test_dir, '.DS_Store')

if os.path.exists(ds_store_path):
    os.remove(ds_store_path)
    print("Deleted '.DS_Store' file.")
else:
    print("'.DS_Store' file not found.")
    
    #image transformations in training set
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

    #data normalization
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  #resize image
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)  #normalize pixel values
    return img_array

#model building

    #machine learning approach (feature extraction + classifier)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)) #VGG16-based feature extraction

model = Model(inputs=base_model.input, outputs=base_model.output) #model outputs features from last convulational layer

def extract_features(directory, model): #function to preprocess images and extract features using VGG16
    images = []
    labels = []
    class_names = os.listdir(directory)
    valid_extensions = ('.jpg', '.jpeg', '.png')

    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            for file in os.listdir(class_dir):
                if file.endswith(valid_extensions):
                    img_path = os.path.join(class_dir, file)
                    img = Image.open(img_path).convert('RGB').resize((224, 224))  #resize to VGG16 input size
                    img_array = np.array(img)
                    img_array = np.expand_dims(img_array, axis=0)  #add batch dimension
                    img_array = preprocess_input(img_array)  #preprocess for VGG16
                    features = model.predict(img_array)  #extract features
                    images.append(features.flatten())  #flatten the features into 1D array
                    labels.append(class_name)
    
    return np.array(images), np.array(labels)

    #extract features from the datasets
train_features, train_labels = extract_features(train_dir, model)
valid_features, valid_labels = extract_features(valid_dir, model)
test_features, test_labels = extract_features(test_dir, model)

print(f"Train Features Shape: {train_features.shape}")
print(f"Validation Features Shape: {valid_features.shape}")
print(f"Test Features Shape: {test_features.shape}")
   
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42) #initialize the RandomForest Classifier

rf_classifier.fit(train_features, train_labels) #train the classifier on the extracted features

valid_predictions = rf_classifier.predict(valid_features) #predict on the validation set

print(f"RF Validation Accuracy: {accuracy_score(valid_labels, valid_predictions)}") #evaluate the model
print("RF-valid Confusion Matrix:")
print(confusion_matrix(valid_labels, valid_predictions))
print("RF-valid Classification Report:")
print(classification_report(valid_labels, valid_predictions))

test_predictions = rf_classifier.predict(test_features) #predict on the test set

print(f"RF-test Test Accuracy: {accuracy_score(test_labels, test_predictions)}") #evaluate the model on the test data
print("RF-test Confusion Matrix:")
print(confusion_matrix(test_labels, test_predictions))
print("RF-test Classification Report:")
print(classification_report(test_labels, test_predictions))

    #hyperparameter tuning
param_grid = { #define the parameter grid
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42) #initialize Random Forest classifier

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2) #initialize GridSearchCV to find the best parameters

grid_search.fit(train_features, train_labels) #fit on training data

print(f"RF Best Parameters: {grid_search.best_params_}") #print best parameters

best_rf = grid_search.best_estimator_ #evaluate the best model on the validation set
valid_predictions = best_rf.predict(valid_features)
print(f"RF Validation Accuracy: {accuracy_score(valid_labels, valid_predictions)}")

test_predictions_rf = best_rf.predict(test_features) #evaluate the best Random Forest model on the test set

print(f"RF Test Accuracy (Random Forest): {accuracy_score(test_labels, test_predictions_rf)}") #calculate accuracy

print("RF Confusion Matrix (Random Forest):") #generate confusion matrix and classification report
print(confusion_matrix(test_labels, test_predictions_rf))
print("RF Classification Report (Random Forest):")
print(classification_report(test_labels, test_predictions_rf))

    #rf feature importance
importances = best_rf.feature_importances_ #get feature importance from the best Random Forest model
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6)) #plot the feature importance
plt.title("Feature Importances by Random Forest")
plt.bar(range(train_features.shape[1]), importances[indices], align="center")
plt.show()

    #deep learning approach - CNN

    #data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

valid_generator = test_datagen.flow_from_directory(
    valid_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)) #load pre-trained VGG16 model + higher level layers

    #add custom layers on top of VGG16
x = base_model.output
x = Flatten()(x)  #flatten the output layer to 1 dimension
x = Dense(128, activation='relu')(x)  #add a fully connected layer
x = Dropout(0.5)(x)  #add dropout for regularization
predictions = Dense(4, activation='softmax')(x)  #add a final softmax layer for 4 classes

model = Model(inputs=base_model.input, outputs=predictions) #combine the base model and new layers

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy']) #compile the model

early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

history = model.fit( #train the model
    train_generator,
    validation_data=valid_generator,
    epochs=30, 
    callbacks=[early_stopping],
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=valid_generator.samples // valid_generator.batch_size
)

    #fine-tuning cnn
for layer in base_model.layers[-4:]: #unfreeze last 4 layers of VGG16
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy']) #recompile model with lower learning rate

history_fine_tune = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=10,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=valid_generator.samples // valid_generator.batch_size
)

test_loss, test_acc = model.evaluate(test_generator) #evaluate the model on test data
print(f"CNN Test Accuracy: {test_acc}")

plt.plot(history.history['accuracy']) #plot training & validation accuracy values
plt.plot(history.history['val_accuracy'])
plt.title('CNN Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss']) #plot training & validation loss values
plt.plot(history.history['val_loss'])
plt.title('CNN Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

test_predictions = model.predict(test_generator) #generate predictions for the test set
test_predictions_classes = np.argmax(test_predictions, axis=1)

true_classes = test_generator.classes #get true labels from test set

print("CNN Classification Report:") #generate classification report
print(classification_report(true_classes, test_predictions_classes, target_names=test_generator.class_indices.keys()))

print("CNN Confusion Matrix:") #generate confusion matrix
print(confusion_matrix(true_classes, test_predictions_classes))

#compare model performances

    #compare accuracy
rf_test_acc = accuracy_score(test_labels, test_predictions_rf)
cnn_test_acc = test_acc  #this was calculated from the CNN evaluation

print(f"Random Forest Test Accuracy: {rf_test_acc}")
print(f"CNN Test Accuracy: {cnn_test_acc}")

    # Precision, recall, and f1-score:
print("Random Forest Classification Report:")
print(classification_report(test_labels, test_predictions_rf))

print("CNN Classification Report:")
print(classification_report(true_classes, test_predictions_classes))

# Confusion Matrices
print("Random Forest Confusion Matrix:")
print(confusion_matrix(test_labels, test_predictions_rf)) 

print("CNN Confusion Matrix:")
print(confusion_matrix(true_classes, test_predictions_classes))

Chest Cancer Detection Using Machine Learning and Deep Learning

This project aims to detect and classify types of chest cancer using CT-scan images. It uses a combination of traditional machine learning methods and deep learning approaches to achieve high accuracy in cancer detection. The project leverages a Convolutional Neural Network (CNN) and Random Forest (RF) classifiers.


Project Overview

Chest cancer detection from CT-scan images is an important task in the medical domain. The dataset used for this project includes three types of chest cancer:
	•	Adenocarcinoma
	•	Large Cell Carcinoma
	•	Squamous Cell Carcinoma

Additionally, it includes a fourth class for healthy/normal cells.

This project implements two main approaches:
	•	Machine Learning Approach: A Random Forest Classifier trained on extracted features from the images using a pre-trained CNN.
	•	Deep Learning Approach: A CNN model trained end-to-end to classify the images directly.


Project Structure

The project includes the following files and directories:
	•	chest-cancer-detection.py: Main script for running the machine learning and deep learning models.
	•	requirements.txt: List of Python dependencies required to run the project.
	•	README.md: This file, providing an overview of the project and instructions for setup.
	•	/data: Contains training, validation, and test datasets (separated into corresponding folders).


Dataset

The dataset is structured into four folders:
	•	Adenocarcinoma
	•	Large Cell Carcinoma
	•	Squamous Cell Carcinoma
	•	Normal

Each folder contains CT-scan images categorized by the type of chest cancer or normal cells.


Installation

To run the project locally, you'll need to set up the environment and install the necessary dependencies.

Steps to set up the environment:

	•Clone the repository:
		git clone https://github.com/YOUR_USERNAME/chest-cancer-detection.git
		cd chest-cancer-detection
	•Create and activate a virtual environment:
		python3 -m venv venv
		source venv/bin/activate #on macOS/LLinux
		#Or, on Windows:
		#venv\Scripts\activate
	•Install the dependencies:
		pip install -r requirements.txt
	•Download the dataset: Place the dataset in the /data folder, ensuring the structure 		follows the project’s requirements (i.e., having subfolders for training, validation, 		and test sets).


How to Run the Project

After setting up the environment and installing the dependencies, you can run the models.

Random Forest (Machine Learning Approach): The Random Forest model uses features extracted from a pre-trained CNN. 
To run the Random Forest classifier:
	python chest-cancer-detection.py --model random-forest

CNN (Deep Learning Approach): The CNN model is trained end-to-end using the raw CT-scan images. 
To run the CNN classifier:
	python chest-cancer-detection.py --model cnn


Results

Random Forest Classifier:
	•	Validation Accuracy: ~80.5%
	•	Test Accuracy: ~72.3%

CNN Classifier:
	•	Validation Accuracy: ~62.5%
	•	Test Accuracy: ~54.9%

The CNN model's performance can potentially be improved with more fine-tuning, hyperparameter adjustments, or additional data.


Visualization

The project includes several visualizations to analyze the dataset and the model's performance:
	•	Image distribution across training, validation, and test sets.
	•	Image size distribution.
	•	Feature importance from the Random Forest classifier.
	•	Model accuracy and loss curves over the training epochs.


Future Work
	•	Explore other deep learning architectures such as ResNet, VGG, etc.
	•	Improve the data augmentation techniques to enhance the CNN model’s generalization.
	•	Investigate class imbalance issues and implement solutions like oversampling or class 		weighting.

References
	•	Kaggle Chest Cancer CT-Scan Dataset








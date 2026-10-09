Fingerprint-Based Blood Group Detection using CNN
A convolutional neural network (CNN) that predicts a person's blood group from a fingerprint image, with a small Flask web app to upload an image and see the prediction.

Published at ICWiCOM 2025 (International Conference on Wireless Communication, DJSCE Mumbai; Springer). Team of 4 students guided by Sayali Badhan. I am the first-listed author.

Results
Item	Value
Dataset	Public Kaggle fingerprint blood group dataset, 6,000 images, 8 classes (A+, A-, B+, B-, AB+, AB-, O+, O-)
Image size	64 x 64
Split	70% train, 15% validation, 15% test
Test accuracy	about 88%
We also tried the model on our own fingerprints and it gave correct results.

Model
Three convolution blocks (32, 64, 128 filters, each with max pooling and dropout) followed by dense layers and an 8-way softmax output. Trained for up to 50 epochs with learning-rate reduction and early stopping.

Project files
fingerprint_blood_group_cnn.ipynb: data loading, class balancing, model training and evaluation
app.py and templates/index.html: Flask web app (upload a fingerprint image, get the predicted blood group)
requirements.txt: Python packages
How to run
Install packages: pip install -r requirements.txt
Open the notebook, download the dataset from Kaggle, run all cells, and save the trained model as model/model.h5. (The trained model file is not included in this repository.)
Start the app: python app.py, then open the address shown in the terminal.
Limitations
The classes were balanced by repeating images before the train/test split, so the same image can appear in both sets. The 88% figure may therefore be higher than real-world performance.
A 6,000-image dataset is small, and fingerprints from one dataset may not represent other scanners or people.
This is a research prototype. It is not a medical tool and must not be used to decide anyone's blood group.
Team
Maitreyi Phadke, Arya Raut, Chinmay Sawant, Gauri Ramekar. Guided by Sayali Badhan.

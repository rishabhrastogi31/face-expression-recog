# face-expression-recog



FER2013
FER2013 is an open-source dataset which is first, created for an ongoing
project by Pierre-Luc Carrier and Aaron Courville, then shared publicly
for a Kaggle competition, shortly before it was introduced in the ICML
2013 Challenges in Representation Learning. The database was created
using the Google image search API that matched a set of 184
emotion-related keywords to capture the six basic expressions as well as
the neutral expression. Images were resized to 48x48 pixels and
converted to grayscale. Human labelers rejected incorrectly labeled
images, corrected the cropping if necessary, and ltered out some
duplicate images. The resulting database contains 35,887 images most of
which are in the wild settings with various emotions -7 emotions, all
labeled:-
Emotion labels in the dataset:
0: -4593 images- Angry
1: -547 images- Disgust
2: -5121 images- Fear
3: -8989 images- Happy
4: -6077 images- Sad
5: -4002 images- Surprise
6: -6198 images- Neutral

 Facial expressions
can be used as important cues in determining the emotion or state of a
person. our first model of cnn having 6 convolutional layers and 2 dense
fully connected layers shows training accuracy of 78% and validation
accuracy of 58% for 20 epochs . our second cnn model which contains
11 convolutional layers with 4 pooling layers for dimensionality
reduction and 7 dense fully connected layers applied with l2
regularization shows train accuracy of 87% and validation accuracy of
maximum 63% when trained for 55 epochs and the same model number
2 is modified with data augmentation and enhanced learning rate to
reduce overfitting , which when trained for 100 epochs reduced
overfitting keeping accuracy same.



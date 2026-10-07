# Face Detection Project

This project involves face detection using a combination of feature descriptors and classifiers. The models are trained on a set of training and test images and are then used to detect faces in larger images. Various feature extraction techniques (Full Image, PCA Reduced, Gabor, LBP and HOG) are combined with classifiers (KNN, SVM, Decision Tree, Random Forest, MLP and AdaBoost) to evaluate detection performance. PCA was utilized only for ``Full Image`` feature descriptor for brevity. LOOCV was performed only for ``Full Image -> PCA Reduced -> KNN`` and ``HOG -> KNN`` for brevity.

## Notes:

I first started with ``Full Image -> KNN``, and then improve it to v2, where Confidence Score is implemented into the KNN model, which all other feature descriptors using KNN then also used.
Then on ``Full Image -> SVM``, I improved it to v2, where multiscale sliding window detector was implemented instead of a fixed one. Hence forth, all models utilized multiscale sliding window detectors.

PCA can also be applied to all feature descriptors, but due to brevity, I applied it only to the ``Full Image`` feature descriptor, since this typically contains the most number of dimensions, compared to HOG or Gabor, where features are already reduced.

Additionally, LOOCV can be applied to all classifiers, but due to brevity, I only did it for ``Full Image -> PCA Reduced -> KNN`` and ``HOG -> KNN`` That's why you will find a sub-folder "LOOCV" only in there.

.mat files are available for some models that are ready to load without running the scripts again, especially for those that take a long time to run.

Default code provided and resources like the test and training images are all stored in code_and_resources. Add this to your path in MATLAB to run the .m scripts for the models.
The only extra files in that folder that wasn't there by default is ``car-2683858_1280.jpg`` (which is for fun, to visualize using HOG), and ``face_all.cdataset`` which was a simple merge of the ``face_train.cdataset`` + ``face_test.cdataset``, in order to use it for Leave-One-Out-Cross-Validation (LOOCV).

## Directory Structure

The project files are organized as follows:

```
├── code_and_resources
├── Feature Descriptor - Full Image
│    ├── Classifier - AdaBoost
│    ├── Classifier - Standalone Decision Tree
│    ├── Classifier - SVM
│    ├── Classifier - KNN
│    ├── Classifier - MLP
│    ├── Classifier - Random Forest
│    └── PCA Reduced
│         ├── Classifier - AdaBoost
│         ├── Classifier - Standalone Decision Tree
│         ├── Classifier - SVM
│         ├── Classifier - KNN
│         |     └── LOOCV
│         ├── Classifier - MLP
│         └── Classifier - Random Forest
├── Feature Descriptor - Gabor
│    ├── Classifier - AdaBoost
│    ├── Classifier - Standalone Decision Tree
│    ├── Classifier - SVM
│    ├── Classifier - KNN
│    ├── Classifier - MLP
│    └── Classifier - Random Forest
├── Feature Descriptor - HOG
│    ├── Classifier - AdaBoost
│    ├── Classifier - Standalone Decision Tree
│    ├── Classifier - SVM
│    ├── Classifier - KNN
│    |     └── LOOCV
│    ├── Classifier - MLP
│    └── Classifier - Random Forest
└── Feature Descriptor - LBP
     ├── Classifier - AdaBoost
     ├── Classifier - Standalone Decision Tree
     ├── Classifier - SVM
     ├── Classifier - KNN
     ├── Classifier - MLP
     └── Classifier - Random Forest
```
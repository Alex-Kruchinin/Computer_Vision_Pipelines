clear 
close all

%% Full Image -> PCA -> KNN with Leave-One-Out Cross Validation
%% OPTION: USE HISTOGRAM EQUALIZATION TO PREPROCESS?
applyHistogramEqualization = false;

% ON:
% LOOCV average accuracy for K values from 1 to 90:
% K = 1, Average LOOCV Accuracy = 95.16%
% K = 2, Average LOOCV Accuracy = 96.26%
% K = 3, Average LOOCV Accuracy = 93.41%
% K = 4, Average LOOCV Accuracy = 94.07%
% K = 5, Average LOOCV Accuracy = 92.09%
% K = 6, Average LOOCV Accuracy = 93.19%
% K = 7, Average LOOCV Accuracy = 91.87%
% K = 8, Average LOOCV Accuracy = 92.31%
% K = 9, Average LOOCV Accuracy = 90.99%
% K = 10, Average LOOCV Accuracy = 92.09%
% K = 11, Average LOOCV Accuracy = 90.11%
% K = 12, Average LOOCV Accuracy = 90.55%
% K = 13, Average LOOCV Accuracy = 89.45%
% K = 14, Average LOOCV Accuracy = 89.89%
% K = 15, Average LOOCV Accuracy = 88.79%
% K = 16, Average LOOCV Accuracy = 89.01%
% K = 17, Average LOOCV Accuracy = 88.13%
% K = 18, Average LOOCV Accuracy = 88.13%
% K = 19, Average LOOCV Accuracy = 87.69%
% K = 20, Average LOOCV Accuracy = 88.57%
% K = 21, Average LOOCV Accuracy = 87.47%
% K = 22, Average LOOCV Accuracy = 87.91%
% K = 23, Average LOOCV Accuracy = 87.25%
% K = 24, Average LOOCV Accuracy = 87.47%
% K = 25, Average LOOCV Accuracy = 86.81%
% K = 26, Average LOOCV Accuracy = 87.69%
% K = 27, Average LOOCV Accuracy = 87.25%
% K = 28, Average LOOCV Accuracy = 87.47%
% K = 29, Average LOOCV Accuracy = 87.03%
% K = 30, Average LOOCV Accuracy = 87.03%
% K = 31, Average LOOCV Accuracy = 86.15%
% K = 32, Average LOOCV Accuracy = 86.81%
% K = 33, Average LOOCV Accuracy = 85.93%
% K = 34, Average LOOCV Accuracy = 86.15%
% K = 35, Average LOOCV Accuracy = 85.93%
% K = 36, Average LOOCV Accuracy = 86.15%
% K = 37, Average LOOCV Accuracy = 85.93%
% K = 38, Average LOOCV Accuracy = 86.15%
% K = 39, Average LOOCV Accuracy = 85.49%
% K = 40, Average LOOCV Accuracy = 85.71%
% K = 41, Average LOOCV Accuracy = 85.05%
% K = 42, Average LOOCV Accuracy = 85.27%
% K = 43, Average LOOCV Accuracy = 84.84%
% K = 44, Average LOOCV Accuracy = 85.05%
% K = 45, Average LOOCV Accuracy = 84.62%
% K = 46, Average LOOCV Accuracy = 84.84%
% K = 47, Average LOOCV Accuracy = 84.62%
% K = 48, Average LOOCV Accuracy = 84.62%
% K = 49, Average LOOCV Accuracy = 84.40%
% K = 50, Average LOOCV Accuracy = 84.62%
% K = 51, Average LOOCV Accuracy = 84.62%
% K = 52, Average LOOCV Accuracy = 84.84%
% K = 53, Average LOOCV Accuracy = 84.40%
% K = 54, Average LOOCV Accuracy = 84.62%
% K = 55, Average LOOCV Accuracy = 83.30%
% K = 56, Average LOOCV Accuracy = 83.74%
% K = 57, Average LOOCV Accuracy = 83.08%
% K = 58, Average LOOCV Accuracy = 83.52%
% K = 59, Average LOOCV Accuracy = 82.86%
% K = 60, Average LOOCV Accuracy = 82.86%
% K = 61, Average LOOCV Accuracy = 82.64%
% K = 62, Average LOOCV Accuracy = 82.64%
% K = 63, Average LOOCV Accuracy = 82.64%
% K = 64, Average LOOCV Accuracy = 82.64%
% K = 65, Average LOOCV Accuracy = 82.64%
% K = 66, Average LOOCV Accuracy = 82.64%
% K = 67, Average LOOCV Accuracy = 82.42%
% K = 68, Average LOOCV Accuracy = 82.64%
% K = 69, Average LOOCV Accuracy = 82.20%
% K = 70, Average LOOCV Accuracy = 82.42%
% K = 71, Average LOOCV Accuracy = 82.42%
% K = 72, Average LOOCV Accuracy = 82.20%
% K = 73, Average LOOCV Accuracy = 82.42%
% K = 74, Average LOOCV Accuracy = 82.20%
% K = 75, Average LOOCV Accuracy = 82.42%
% K = 76, Average LOOCV Accuracy = 82.20%
% K = 77, Average LOOCV Accuracy = 82.20%
% K = 78, Average LOOCV Accuracy = 81.98%
% K = 79, Average LOOCV Accuracy = 81.98%
% K = 80, Average LOOCV Accuracy = 81.98%
% K = 81, Average LOOCV Accuracy = 81.54%
% K = 82, Average LOOCV Accuracy = 81.76%
% K = 83, Average LOOCV Accuracy = 81.10%
% K = 84, Average LOOCV Accuracy = 81.10%
% K = 85, Average LOOCV Accuracy = 80.88%
% K = 86, Average LOOCV Accuracy = 80.88%
% K = 87, Average LOOCV Accuracy = 80.88%
% K = 88, Average LOOCV Accuracy = 80.88%
% K = 89, Average LOOCV Accuracy = 80.66%
% K = 90, Average LOOCV Accuracy = 80.88%
% 
% Best K = 2 with Average LOOCV Accuracy = 96.26%

% OFF:
% LOOCV average accuracy for K values from 1 to 90:
% K = 1, Average LOOCV Accuracy = 98.68%
% K = 2, Average LOOCV Accuracy = 98.46%
% K = 3, Average LOOCV Accuracy = 95.82%
% K = 4, Average LOOCV Accuracy = 96.48%
% K = 5, Average LOOCV Accuracy = 94.51%
% K = 6, Average LOOCV Accuracy = 94.73%
% K = 7, Average LOOCV Accuracy = 93.85%
% K = 8, Average LOOCV Accuracy = 93.85%
% K = 9, Average LOOCV Accuracy = 93.41%
% K = 10, Average LOOCV Accuracy = 94.51%
% K = 11, Average LOOCV Accuracy = 92.75%
% K = 12, Average LOOCV Accuracy = 93.41%
% K = 13, Average LOOCV Accuracy = 92.09%
% K = 14, Average LOOCV Accuracy = 92.31%
% K = 15, Average LOOCV Accuracy = 91.65%
% K = 16, Average LOOCV Accuracy = 92.09%
% K = 17, Average LOOCV Accuracy = 91.65%
% K = 18, Average LOOCV Accuracy = 89.89%
% K = 19, Average LOOCV Accuracy = 87.91%
% K = 20, Average LOOCV Accuracy = 89.01%
% K = 21, Average LOOCV Accuracy = 87.69%
% K = 22, Average LOOCV Accuracy = 87.69%
% K = 23, Average LOOCV Accuracy = 87.25%
% K = 24, Average LOOCV Accuracy = 87.25%
% K = 25, Average LOOCV Accuracy = 87.25%
% K = 26, Average LOOCV Accuracy = 87.25%
% K = 27, Average LOOCV Accuracy = 86.15%
% K = 28, Average LOOCV Accuracy = 86.15%
% K = 29, Average LOOCV Accuracy = 85.71%
% K = 30, Average LOOCV Accuracy = 85.93%
% K = 31, Average LOOCV Accuracy = 85.05%
% K = 32, Average LOOCV Accuracy = 85.05%
% K = 33, Average LOOCV Accuracy = 84.40%
% K = 34, Average LOOCV Accuracy = 84.18%
% K = 35, Average LOOCV Accuracy = 83.52%
% K = 36, Average LOOCV Accuracy = 83.52%
% K = 37, Average LOOCV Accuracy = 83.74%
% K = 38, Average LOOCV Accuracy = 83.74%
% K = 39, Average LOOCV Accuracy = 83.96%
% K = 40, Average LOOCV Accuracy = 84.18%
% K = 41, Average LOOCV Accuracy = 83.96%
% K = 42, Average LOOCV Accuracy = 83.96%
% K = 43, Average LOOCV Accuracy = 83.52%
% K = 44, Average LOOCV Accuracy = 83.74%
% K = 45, Average LOOCV Accuracy = 83.52%
% K = 46, Average LOOCV Accuracy = 83.74%
% K = 47, Average LOOCV Accuracy = 83.30%
% K = 48, Average LOOCV Accuracy = 83.52%
% K = 49, Average LOOCV Accuracy = 83.52%
% K = 50, Average LOOCV Accuracy = 83.30%
% K = 51, Average LOOCV Accuracy = 82.86%
% K = 52, Average LOOCV Accuracy = 82.86%
% K = 53, Average LOOCV Accuracy = 82.86%
% K = 54, Average LOOCV Accuracy = 82.86%
% K = 55, Average LOOCV Accuracy = 83.08%
% K = 56, Average LOOCV Accuracy = 83.08%
% K = 57, Average LOOCV Accuracy = 82.42%
% K = 58, Average LOOCV Accuracy = 82.64%
% K = 59, Average LOOCV Accuracy = 82.42%
% K = 60, Average LOOCV Accuracy = 82.42%
% K = 61, Average LOOCV Accuracy = 82.42%
% K = 62, Average LOOCV Accuracy = 82.42%
% K = 63, Average LOOCV Accuracy = 82.42%
% K = 64, Average LOOCV Accuracy = 82.42%
% K = 65, Average LOOCV Accuracy = 82.42%
% K = 66, Average LOOCV Accuracy = 82.42%
% K = 67, Average LOOCV Accuracy = 82.20%
% K = 68, Average LOOCV Accuracy = 81.98%
% K = 69, Average LOOCV Accuracy = 81.76%
% K = 70, Average LOOCV Accuracy = 81.54%
% K = 71, Average LOOCV Accuracy = 81.32%
% K = 72, Average LOOCV Accuracy = 81.54%
% K = 73, Average LOOCV Accuracy = 81.54%
% K = 74, Average LOOCV Accuracy = 81.54%
% K = 75, Average LOOCV Accuracy = 81.10%
% K = 76, Average LOOCV Accuracy = 81.10%
% K = 77, Average LOOCV Accuracy = 80.88%
% K = 78, Average LOOCV Accuracy = 80.66%
% K = 79, Average LOOCV Accuracy = 81.10%
% K = 80, Average LOOCV Accuracy = 80.66%
% K = 81, Average LOOCV Accuracy = 80.88%
% K = 82, Average LOOCV Accuracy = 80.66%
% K = 83, Average LOOCV Accuracy = 80.22%
% K = 84, Average LOOCV Accuracy = 80.22%
% K = 85, Average LOOCV Accuracy = 80.44%
% K = 86, Average LOOCV Accuracy = 80.44%
% K = 87, Average LOOCV Accuracy = 80.22%
% K = 88, Average LOOCV Accuracy = 80.22%
% K = 89, Average LOOCV Accuracy = 80.22%
% K = 90, Average LOOCV Accuracy = 80.22%

%% Step 1: Load All Data (Combine face and non-face images)
[images, labels] = loadFaceImages('face_all.cdataset');  % Load ALL 124 samples (910 samples if data augmented) instead of a 94/30 split, for LOOCV

% Perform histogram equalization if enabled
if applyHistogramEqualization
    for i = 1:size(images, 1)
        originalImage = reshape(images(i, :), [27, 18]);
        originalImage = uint8(255 * mat2gray(originalImage));
        equalizedImage = histeq(originalImage);
        images(i, :) = double(reshape(equalizedImage, 1, []));
    end
end

disp('Loaded all images, histeq performed if turned on.')

%% Step 2: Apply PCA for Dimensionality Reduction

nReducedDimensions = 100; % Number of dimensions to reduce to
[U, S, imagesPCA] = pca(images, nReducedDimensions);

disp(['Applied PCA to reduce dimensions to ', num2str(nReducedDimensions)])

%% Step 3: LOOCV for K-NN with Tuning K

maxK = 100; % Maximum value of K to check (adjust as needed)
totalSamples = size(imagesPCA, 1);
accuracyPerK = zeros(maxK, 1); % To store average accuracy for each K

fprintf('Running LOOCV for K values from 1 to %d:\n', maxK);

% For each K within the selected range
for K = 1:maxK
    correctPredictions = 0; % Increment the num of correct predictions for this K if it predicts correctly
    
    % For X number of samples available (to form X number of folds - in this case it's 910 if data augmented)
    for testIndex = 1:totalSamples
        % Leave the current sample out as the test instance
        testFeature = imagesPCA(testIndex, :);
        testLabel = labels(testIndex);
        
        % Use remaining samples as the training set
        trainFeatures = imagesPCA([1:testIndex-1, testIndex+1:end], :);
        trainLabels = labels([1:testIndex-1, testIndex+1:end]);
        
        % Initialize KNN model with current training set
        modelNN.neighbours = trainFeatures;
        modelNN.labels = trainLabels;
        
        % Perform KNN classification on the test instance with current K
        predictedLabel = KNNTesting(testFeature, modelNN, K);
        
        % Check if the prediction is correct
        if predictedLabel == testLabel
            correctPredictions = correctPredictions + 1;
        end
    end
    
    % Calculate and store average accuracy for this K
    accuracyPerK(K) = correctPredictions / totalSamples;
    fprintf('K = %d, Average LOOCV Accuracy = %.2f%%\n', K, accuracyPerK(K) * 100);
end

%% Step 4: Determine the Best K

[bestAccuracy, bestK] = max(accuracyPerK);
fprintf('\nBest K = %d with Average LOOCV Accuracy = %.2f%%\n', bestK, bestAccuracy * 100);


%% Functions

% EuclideanDistance function
function dEuc = EuclideanDistance(sample1, sample2)
    difference = sample1 - sample2;
    squaredDifference = difference .^ 2;
    sumOfSquares = sum(squaredDifference);
    dEuc = sqrt(sumOfSquares);
end

% KNNTesting function
function prediction = KNNTesting(testImage, modelNN, K)
    trainingImages = modelNN.neighbours; 
    trainingLabels = modelNN.labels;
    numTrainingSamples = size(trainingImages, 1);
    distances = zeros(numTrainingSamples, 1); 
    
    for i = 1:numTrainingSamples
        distances(i) = EuclideanDistance(testImage, trainingImages(i, :));
    end
    
    [~, sortedIndices] = sort(distances);
    K_nearestLabels = trainingLabels(sortedIndices(1:K)); 
    prediction = mode(K_nearestLabels);
end

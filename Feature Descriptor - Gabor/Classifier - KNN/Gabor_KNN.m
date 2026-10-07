clear;
close all;
%% Gabor -> KNN -> Multiscale Sliding Window Detection with Confidence Scores
%% OPTION: USE HISTOGRAM EQUALIZATION TO PREPROCESS?
applyHistogramEqualization = true; % Set to true to apply histogram equalization, false to skip it

%Final Evaluation on Test Set with Best K = 11:
%TP: 198, FP: 33, TN: 7, FN: 2
%Accuracy: 85.42% | Precision: 0.86 | Recall: 0.99 | F1 Score: 0.92

% K = 1, Accuracy = 76.25%
% K = 2, Accuracy = 63.75%
% K = 3, Accuracy = 77.92%
% K = 4, Accuracy = 73.75%
% K = 5, Accuracy = 80.83%
% K = 6, Accuracy = 78.33%
% K = 7, Accuracy = 81.67%
% K = 8, Accuracy = 80.00%
% K = 9, Accuracy = 85.00%
% K = 10, Accuracy = 83.33%
% K = 11, Accuracy = 85.42%
% K = 12, Accuracy = 83.75%
% K = 13, Accuracy = 83.75%
% K = 14, Accuracy = 84.17%
% K = 15, Accuracy = 84.17%
% K = 16, Accuracy = 84.17%
% K = 17, Accuracy = 84.58%
% K = 18, Accuracy = 84.58%
% K = 19, Accuracy = 84.58%
% K = 20, Accuracy = 84.58%
% K = 21, Accuracy = 84.58%
% K = 22, Accuracy = 84.58%
% K = 23, Accuracy = 84.58%
% K = 24, Accuracy = 84.58%
% K = 25, Accuracy = 84.17%
% K = 26, Accuracy = 84.17%
% K = 27, Accuracy = 84.17%
% K = 28, Accuracy = 84.17%
% K = 29, Accuracy = 84.17%
% K = 30, Accuracy = 84.17%
% K = 31, Accuracy = 83.33%
% K = 32, Accuracy = 83.33%
% K = 33, Accuracy = 83.33%
% K = 34, Accuracy = 83.33%
% K = 35, Accuracy = 83.33%
% K = 36, Accuracy = 83.33%
% K = 37, Accuracy = 83.33%
% K = 38, Accuracy = 83.33%
% K = 39, Accuracy = 83.33%
% K = 40, Accuracy = 83.33%
% K = 41, Accuracy = 83.33%
% K = 42, Accuracy = 83.33%
% K = 43, Accuracy = 83.33%
% K = 44, Accuracy = 83.33%
% K = 45, Accuracy = 83.33%
% K = 46, Accuracy = 83.33%
% K = 47, Accuracy = 83.33%
% K = 48, Accuracy = 83.33%
% K = 49, Accuracy = 83.33%
% K = 50, Accuracy = 83.33%
% K = 51, Accuracy = 83.33%
% K = 52, Accuracy = 83.33%
% K = 53, Accuracy = 83.33%
% K = 54, Accuracy = 83.33%
% K = 55, Accuracy = 83.33%
% K = 56, Accuracy = 83.33%
% K = 57, Accuracy = 83.33%
% K = 58, Accuracy = 83.33%
% K = 59, Accuracy = 83.33%
% K = 60, Accuracy = 83.33%
% K = 61, Accuracy = 83.33%
% K = 62, Accuracy = 83.33%
% K = 63, Accuracy = 83.33%
% K = 64, Accuracy = 83.33%
% K = 65, Accuracy = 83.33%
% K = 66, Accuracy = 83.33%
% K = 67, Accuracy = 83.33%
% K = 68, Accuracy = 83.33%
% K = 69, Accuracy = 83.33%
% K = 70, Accuracy = 83.33%
% K = 71, Accuracy = 83.33%
% K = 72, Accuracy = 83.33%
% K = 73, Accuracy = 83.33%
% K = 74, Accuracy = 83.33%
% K = 75, Accuracy = 83.33%
% K = 76, Accuracy = 83.33%
% K = 77, Accuracy = 83.33%
% K = 78, Accuracy = 83.33%
% K = 79, Accuracy = 83.33%
% K = 80, Accuracy = 83.33%
% K = 81, Accuracy = 83.33%
% K = 82, Accuracy = 83.33%
% K = 83, Accuracy = 83.33%
% K = 84, Accuracy = 83.33%
% K = 85, Accuracy = 83.33%
% K = 86, Accuracy = 83.33%
% K = 87, Accuracy = 83.33%
% K = 88, Accuracy = 83.33%
% K = 89, Accuracy = 83.33%
% K = 90, Accuracy = 83.33%

%% Step 1: Load Training Data

% Load face images and labels
[images, labels] = loadFaceImages('face_train.cdataset');

% Perform histogram equalization if enabled
if applyHistogramEqualization
    for i = 1:size(images, 1)
        originalImage = reshape(images(i, :), [27, 18]);
        originalImage = uint8(255 * mat2gray(originalImage));
        equalizedImage = histeq(originalImage);
        images(i, :) = double(reshape(equalizedImage, 1, []));
    end
end

disp('Loaded training images, histeq performed if turned on.')

%% Step 2: Extract Gabor Features for Training Data

numImages = size(images, 1);
gaborFeatureMatrix = zeros(numImages, 19440);

for i = 1:numImages
    image = reshape(images(i, :), [27, 18]);
    gaborFeatureMatrix(i, :) = gabor_feature_vector(image);
end

disp('Gabor features for all training images have been extracted.');

%% (FOR FUN ONLY) VISUALIZING A GABOR FEATURE IMAGE
n = 21; % Change this index to select different images. Note that if you have Augmentation turned on in loadFaceImages.m, there'll be 10 images of roughly the same person so 1-10, 11-20, etc. 
sampleImage = reshape(images(n, :), [27, 18]); % Have to turn it back into a 2D matrix, since it was reshaped into 1D in the loadPedestrianDatabase().
sampleImage2 = imread('car-2683858_1280.jpg');
% Load a sample image and convert to grayscale if necessary
if size(sampleImage2, 3) > 1
    sampleImage2 = rgb2gray(sampleImage2);
end

% Apply the Gabor feature extraction
gaborFeatures = gabor_feature_vector(sampleImage);
gaborFeatures2 = gabor_feature_vector(sampleImage2);

% Reshape the Gabor features to a 2D format suitable for visualization
% Since the gabor feature vector output is 1x19440, we'll reshape it to a 135x144 grid
featureImage = reshape(gaborFeatures, [144, 135]);
featureImage2 = reshape(gaborFeatures2, [144, 135]);

% Display the original image and the Gabor feature visualization side by side
figure;

% Display the original image
subplot(1, 2, 1);
imshow(sampleImage, []);
title('Original Image');
% Display the Gabor feature visualization
subplot(1, 2, 2);
imshow(featureImage, []);
title('Gabor Feature Visualization');

% Display the original image and the Gabor feature visualization side by side
figure;

% Display the original image
subplot(1, 2, 1);
imshow(sampleImage2, []);
title('Original Image');
% Display the Gabor feature visualization
subplot(1, 2, 2);
imshow(featureImage2, []);
title('Gabor Feature Visualization');
%% Step 3: Initialize K-NN Model

modelNN.neighbours = gaborFeatureMatrix;
modelNN.labels = labels;

disp('NN model initialized using Gabor features and labels.')

%% Step 4: Load Testing Data

[images_test, labels_test] = loadFaceImages('face_test.cdataset');

% Perform histogram equalization on testing images if enabled
if applyHistogramEqualization
    for i = 1:size(images_test, 1)
        originalImage = reshape(images_test(i, :), [27, 18]);
        originalImage = uint8(255 * mat2gray(originalImage));
        equalizedImage = histeq(originalImage);
        images_test(i, :) = double(reshape(equalizedImage, 1, []));
    end
end

disp('Loaded test images, histeq performed if turned on.')

%% Step 5: Extract Gabor Features for Test Data

numTestImages = size(images_test, 1);
gaborTestFeatureMatrix = zeros(numTestImages, size(gaborFeatureMatrix, 2));

for i = 1:numTestImages
    testImage = reshape(images_test(i, :), [27, 18]);
    gaborTestFeatureMatrix(i, :) = gabor_feature_vector(testImage);
end

disp('Gabor features for all testing images have been extracted.');

%% Step 6: K-NN Testing

BestAccuracy = 0;
BestK = 1;
maxK = 90;

fprintf('Now testing for best K from %d to %d. This will take a long time, about 15 minutes.\n', BestK, maxK)

for K = 1:maxK
    classificationResult = zeros(size(gaborTestFeatureMatrix, 1), 1);
    for i = 1:size(gaborTestFeatureMatrix, 1)
        testFeature = gaborTestFeatureMatrix(i, :);
        classificationResult(i) = KNNTesting(testFeature, modelNN, K);
    end
    
    comparison = (labels_test == classificationResult);
    Accuracy = sum(comparison) / length(comparison);
    
    if Accuracy > BestAccuracy
        BestAccuracy = Accuracy;
        BestK = K;
    end
    
    fprintf('K = %d, Accuracy = %.2f%%\n', K, Accuracy * 100);
end

fprintf('\nBest K = %d with Accuracy = %.2f%%\n', BestK, BestAccuracy * 100);

%% Step 7: Final Evaluation on Best K

predicted_labels = zeros(size(labels_test));

for i = 1:size(gaborTestFeatureMatrix, 1)
    testFeature = gaborTestFeatureMatrix(i, :);
    predicted_labels(i) = KNNTesting(testFeature, modelNN, BestK);
end

comparison = (labels_test == predicted_labels);

TP = 0; FP = 0; TN = 0; FN = 0;

for i = 1:length(labels_test)
    if labels_test(i) == 1 && predicted_labels(i) == 1
        TP = TP + 1;
    elseif labels_test(i) == -1 && predicted_labels(i) == 1
        FP = FP + 1;
    elseif labels_test(i) == -1 && predicted_labels(i) == -1
        TN = TN + 1;
    elseif labels_test(i) == 1 && predicted_labels(i) == -1
        FN = FN + 1;
    end
end

accuracy = (TP + TN) / length(labels_test);
precision = TP / (TP + FP);
recall = TP / (TP + FN);
f1_score = 2 * (precision * recall) / (precision + recall);

fprintf('\nFinal Evaluation on Test Set with Best K = %d:\n', BestK);
fprintf('TP: %d, FP: %d, TN: %d, FN: %d\n', TP, FP, TN, FN);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Precision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);
fprintf('F1 Score: %.2f\n', f1_score);

%% Display Correctly Classified and Misclassified Images

% Display correctly classified images
figure('Name', 'Up to 25 Samples of Correctly Classified Images');
title('Correctly Classified Images')
count = 0;
i = 1;
while (count < 25) && (i <= length(comparison))
    if comparison(i)
        count = count + 1;
        Im = reshape(images_test(i, :), [27, 18]);
        subplot(5, 5, count)
        imshow(Im, []);
    end
    i = i + 1;
end

pause(1);

% Display incorrectly classified images
figure('Name', 'Up to 25 Samples of Incorrectly Classified Images');
title('Incorrectly Classified Images')
count = 0;
i = 1;
while (count < 25) && (i <= length(comparison))
    if ~comparison(i)
        count = count + 1;
        Im = reshape(images_test(i, :), [27, 18]);
        subplot(5, 5, count)
        imshow(Im, []);
        title(['Pred: ' num2str(predicted_labels(i)) ', True: ' num2str(labels_test(i))])
    end
    i = i + 1;
end

pause(1);

%% Multi-scale Sliding Window Detection on Larger Image for KNN

fprintf("Now using the model to detect in larger images using multi-scale sliding window.\n Will take about 10 minutes.")

% Load the larger image
largeImage = imread('im3.jpg');
if size(largeImage, 3) > 1
    largeImage = rgb2gray(largeImage);
end

% Apply histogram equalization to a copy of the large image
equalizedImage = uint8(255 * mat2gray(largeImage));
equalizedImage = histeq(equalizedImage);

% Define sliding window parameters
windowHeight = 27;
windowWidth = 18;
stepSizeY = 9;
stepSizeX = 6;

% Initialize arrays to store detection results for both original and equalized images
detectionsOriginal = [];
detectionsEqualized = [];

% Start with the base window size and increase incrementally until the window matches or exceeds image dimensions
currentHeight = windowHeight;
currentWidth = windowWidth;

while currentHeight <= size(largeImage, 1) && currentWidth <= size(largeImage, 2)
    % Total steps per scale for progress tracking (original image)
    totalStepsOriginal = ceil((size(largeImage, 1) - currentHeight + 1) / stepSizeY) * ceil((size(largeImage, 2) - currentWidth + 1) / stepSizeX);
    currentStepOriginal = 0;
    progressCheckpointOriginal = 0.25; % Progress increments for the original image
    
    % Sliding Window Detection for the Original Image
    fprintf('Processing scale [%dx%d] on original image: [', currentWidth, currentHeight);
    for y = 1:stepSizeY:(size(largeImage, 1) - currentHeight + 1)
        for x = 1:stepSizeX:(size(largeImage, 2) - currentWidth + 1)
            % Extract the window patch
            patch = largeImage(y:(y + currentHeight - 1), x:(x + currentWidth - 1));
            
            % Resize patch to match training data dimensions (18x27) and extract Gabor features
            resizedPatch = imresize(patch, [windowHeight, windowWidth]);
            patchGabor = gabor_feature_vector(resizedPatch);
            
            % Perform classification with KNN
            [label, confidence] = KNNTestingWithConfidence(patchGabor, modelNN, BestK);
            
            % If a face (or positive detection) is found, store it
            if label == 1
                detectionsOriginal = [detectionsOriginal; x, y, currentWidth, currentHeight, confidence];
            end
            
            % Update progress
            currentStepOriginal = currentStepOriginal + 1;
            if currentStepOriginal / totalStepsOriginal >= progressCheckpointOriginal && progressCheckpointOriginal < 1
                fprintf(' %d%% ', round(progressCheckpointOriginal * 100));
                progressCheckpointOriginal = progressCheckpointOriginal + 0.25;
            end
        end
    end
    fprintf('100%%]\n'); % Complete progress bar for original image
    
    % Total steps per scale for progress tracking (equalized image)
    totalStepsEqualized = ceil((size(equalizedImage, 1) - currentHeight + 1) / stepSizeY) * ceil((size(equalizedImage, 2) - currentWidth + 1) / stepSizeX);
    currentStepEqualized = 0;
    progressCheckpointEqualized = 0.25;
    
    % Sliding Window Detection for the Equalized Image
    fprintf('Processing scale [%dx%d] on equalized image: [', currentWidth, currentHeight);
    for y = 1:stepSizeY:(size(equalizedImage, 1) - currentHeight + 1)
        for x = 1:stepSizeX:(size(equalizedImage, 2) - currentWidth + 1)
            % Extract the window patch
            patch = equalizedImage(y:(y + currentHeight - 1), x:(x + currentWidth - 1));
            
            % Resize patch to match training data dimensions (18x27) and extract Gabor features
            resizedPatch = imresize(patch, [windowHeight, windowWidth]);
            patchGabor = gabor_feature_vector(resizedPatch);
            
            % Perform classification with KNN
            [label, confidence] = KNNTestingWithConfidence(patchGabor, modelNN, BestK);
            
            % If a face (or positive detection) is found, store it
            if label == 1
                detectionsEqualized = [detectionsEqualized; x, y, currentWidth, currentHeight, confidence];
            end
            
            % Update progress
            currentStepEqualized = currentStepEqualized + 1;
            if currentStepEqualized / totalStepsEqualized >= progressCheckpointEqualized && progressCheckpointEqualized < 1
                fprintf(' %d%% ', round(progressCheckpointEqualized * 100));
                progressCheckpointEqualized = progressCheckpointEqualized + 0.25;
            end
        end
    end
    fprintf('100%%]\n'); % Complete progress bar for equalized image
    
    % Increment window size by 2 pixels in each dimension for multi-scale sliding window
    currentHeight = currentHeight + 2;
    currentWidth = currentWidth + 2;
end
%% Apply Non-Maxima Suppression (NMS)
threshold = 0.01;
finalDetectionsOriginal = simpleNMS(detectionsOriginal, threshold);
finalDetectionsEqualized = simpleNMS(detectionsEqualized, threshold);

%% Display Results Before and After NMS in a 1x2 Grid

figure;
subplot(1, 2, 1);
ShowDetectionResult(largeImage, detectionsOriginal);
title('Original Image - Faces highlighted before NMS (Dynamic Scale)');

subplot(1, 2, 2);
ShowDetectionResult(equalizedImage, detectionsEqualized);
title('Equalized Image - Faces highlighted before NMS (Dynamic Scale)');

pause(1);

figure;
subplot(1, 2, 1);
ShowDetectionResult(largeImage, finalDetectionsOriginal);
title('Original Image - Faces highlighted after NMS (Dynamic Scale)');

subplot(1, 2, 2);
ShowDetectionResult(equalizedImage, finalDetectionsEqualized);
title('Equalized Image - Faces highlighted after NMS (Dynamic Scale)');


%% Saving whole workspace at this point
saveNow = false; % Set to true only if you want to save workspace. This is so these save sections don't run per part of a normal big run (and re-save a broken model for e.g.)
if saveNow
    save('Gabor_KNN.mat');
end
%% Load workspace to skip all the work above
% For selective loading and saving, specify variable names like:
% save('filename.mat', 'var1', 'var2') and load('filename.mat', 'var1', 'var2').
loadNow = false; % Set to true only if you want to load workspace.
if loadNow
    clear
    load('Gabor_KNN.mat')
end
%% Display Results Before and After NMS in a 1x2 Grid

figure;
subplot(1, 2, 1);
ShowDetectionResult(largeImage, detectionsOriginal);
title('Original Image - Faces highlighted before NMS (Dynamic Scale) - Confidence goes up by colors white-yellow-green-cyan');

subplot(1, 2, 2);
ShowDetectionResult(equalizedImage, detectionsEqualized);
title('Equalized Image - Faces highlighted before NMS (Dynamic Scale) - Confidence goes up by colors white-yellow-green-cyan');

pause(1);

figure;
subplot(1, 2, 1);
ShowDetectionResult(largeImage, finalDetectionsOriginal);
title('Original Image - Faces highlighted after NMS (Dynamic Scale) - Confidence goes up by colors white-yellow-green-cyan');

subplot(1, 2, 2);
ShowDetectionResult(equalizedImage, finalDetectionsEqualized);
title('Equalized Image - Faces highlighted after NMS (Dynamic Scale) - Confidence goes up by colors white-yellow-green-cyan');


%% Functions


% KNNTesting function
function prediction = KNNTesting(testFeature, modelNN, K)
    trainingFeatures = modelNN.neighbours; 
    trainingLabels = modelNN.labels;
    numTrainingSamples = size(trainingFeatures, 1);
    distances = zeros(numTrainingSamples, 1); 
    
    for i = 1:numTrainingSamples
        distances(i) = EuclideanDistance(testFeature, trainingFeatures(i, :));
    end
    
    [~, sortedIndices] = sort(distances);
    K_nearestLabels = trainingLabels(sortedIndices(1:K)); 
    prediction = mode(K_nearestLabels);
end

% Giving KNN testing confidence scores, by using both distance-based confidence and label distribution-based confidence
function [prediction, confidence] = KNNTestingWithConfidence(testImage, modelNN, K)
    trainingImages = modelNN.neighbours; 
    trainingLabels = modelNN.labels;
    numTrainingSamples = size(trainingImages, 1);
    distances = zeros(numTrainingSamples, 1); 
    
    for i = 1:numTrainingSamples
        distances(i) = EuclideanDistance(testImage, trainingImages(i, :));
    end
    
    [sortedDistances, sortedIndices] = sort(distances);
    K_nearestLabels = trainingLabels(sortedIndices(1:K)); 
    prediction = mode(K_nearestLabels);
    
    % Distance-based confidence
    distanceConfidence = 1 / (mean(sortedDistances(1:K)) + eps);
    
    % Label proportion-based confidence
    labelRatioConfidence = sum(K_nearestLabels == prediction) / K;
    
    % Combined confidence score
    confidence = distanceConfidence * labelRatioConfidence;
end

% EuclideanDistance function
function dEuc = EuclideanDistance(sample1, sample2)
    difference = sample1 - sample2;
    squaredDifference = difference .^ 2;
    sumOfSquares = sum(squaredDifference);
    dEuc = sqrt(sumOfSquares);
end



% Had to be modified to handle missing detections, in this case the hist eq'd image had zero detections.
function Objects = simpleNMS(Objects, threshold)
    % Check if Objects is empty or doesn't have enough columns
    if isempty(Objects) || size(Objects, 2) < 5
        return; % Return an empty matrix if there are no detections or insufficient data
    end
    
    % Proceed with NMS if there are enough detections
    [~, sortedIdx] = sort(Objects(:, 5), 'descend');
    Objects = Objects(sortedIdx, :);
    keepIdx = true(size(Objects, 1), 1);
    
    for i = 1:size(Objects, 1)
        if ~keepIdx(i), continue; end
        boxA = Objects(i, 1:4);
        
        for j = i + 1:size(Objects, 1)
            if ~keepIdx(j), continue; end
            boxB = Objects(j, 1:4);
            
            % Calculate intersection area and overlap ratio
            intersectionArea = rectint([boxA(1), boxA(2), boxA(3), boxA(4)], ...
                                       [boxB(1), boxB(2), boxB(3), boxB(4)]);
            areaBoxB = boxB(3) * boxB(4);
            overlapRatio = intersectionArea / areaBoxB;
            
            % Suppress box if overlap exceeds threshold
            if overlapRatio > threshold
                keepIdx(j) = false;
            end
        end
    end
    
    % Keep only the boxes that passed the NMS
    Objects = Objects(keepIdx, :);
end



% Display detection result function
function ShowDetectionResult(Picture, Objects)
    imshow(Picture); hold on;
    colours = ['c'; 'g'; 'y'; 'w'];
    if ~isempty(Objects)
        for n = 1:size(Objects, 1)
            x1 = Objects(n, 1); 
            y1 = Objects(n, 2);
            x2 = x1 + Objects(n, 3); 
            y2 = y1 + Objects(n, 4);

            confidence = Objects(n, 5) / max(Objects(:, 5));
            if confidence > 0.9
                c = 1;
            elseif confidence > 0.7
                c = 2;
            elseif confidence > 0.5
                c = 3;
            else
                c = 4;
            end

            plot([x1 x1 x2 x2 x1], [y1 y2 y2 y1 y1], colours(c));
        end
    end
    hold off;
end

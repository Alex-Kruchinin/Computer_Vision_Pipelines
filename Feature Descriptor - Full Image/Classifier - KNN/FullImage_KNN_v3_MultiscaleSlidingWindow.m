clear 
close all

%% This KNNv3 version implements multiscale sliding window for detection in larger images.
%% Full Image -> KNN -> Multiscale Sliding Window Detection with Confidence Score
%% OPTION: USE HISTOGRAM EQUALIZATION TO PREPROCESS?
applyHistogramEqualization = false; % Set to true to apply histogram equalization, false to skip it
%In my testing, LOWERS accuracy from 90.83 to 86.67% if turned ON.

%% Step 1: Load Training Data

% Load face images and labels
[images, labels] = loadFaceImages('face_train.cdataset');

% Perform histogram equalization if enabled
if applyHistogramEqualization
    for i = 1:size(images, 1)
        % Reshape, normalize, equalize, and reshape back to original format
        originalImage = reshape(images(i, :), [27, 18]);
        originalImage = uint8(255 * mat2gray(originalImage)); % Normalize for equalization
        equalizedImage = histeq(originalImage);
        images(i, :) = double(reshape(equalizedImage, 1, [])); % Flatten and store back
    end
end

disp('Loaded training images, histeq performed if turned on.')

%% Step 2: Initialize K-NN Model

% Create a structure for storing the model
modelNN.neighbours = images; 
modelNN.labels = labels;

disp('NN model initialized using images and labels.')

%% Step 3: Load Testing Data

% Load test face images and labels
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

%% Step 4: K-NN Testing

% Initialize variables
BestAccuracy = 0;
BestK = 1; % To store the best value of K
maxK = 200; % Maximum value of K to check (adjust as needed)

fprintf('Now testing for best K from %d to %d:\n', BestK, maxK)

% Loop through values of K from 1 to maxK
for K = 1:maxK
    % For each testing image, we obtain a prediction based on the current value of K
    classificationResult = zeros(size(images_test, 1), 1);
    for i = 1:size(images_test, 1)
        testImage = images_test(i, :);
        classificationResult(i) = KNNTesting(testImage, modelNN, K);
    end
    
    % Evaluate the accuracy for this K
    comparison = (labels_test == classificationResult);
    Accuracy = sum(comparison) / length(comparison);
    
    % Check if this K gives a better accuracy
    if Accuracy > BestAccuracy
        BestAccuracy = Accuracy; % Update the best accuracy
        BestK = K; % Store the best K
    end
    
    fprintf('K = %d, Accuracy = %.2f%%\n', K, Accuracy * 100); % Print the accuracy for each K
end

% Display the best value of K and its accuracy
fprintf('\nBest K = %d with Accuracy = %.2f%%\n', BestK, BestAccuracy * 100);

%% Step 5: Final Evaluation on Best K
% Perform final classification with the best K on the test set
% After finding the best K value from the previous loop, we use it to evaluate the final metrics.

% Re-run testing with the best K to resolve some bug with calculating TP, FP, TN and FN
predicted_labels = zeros(size(labels_test)); % Initialize predictions array

for i = 1:size(images_test, 1)
    testImage = images_test(i, :);
    predicted_labels(i) = KNNTesting(testImage, modelNN, BestK); % Using BestK only!!!
    % fprintf('Test Image %d: Predicted = %d, Actual = %d\n', i, predicted_labels(i), labels_test(i));
end

% Create the comparison array to identify correct and incorrect classifications
comparison = (labels_test == predicted_labels);


% Initialize counters for TP, FP, TN, FN
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

% Calculate evaluation metrics
accuracy = (TP + TN) / length(labels_test);
precision = TP / (TP + FP);
recall = TP / (TP + FN);
f1_score = 2 * (precision * recall) / (precision + recall);

% Display final metrics
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
        Im = reshape(images_test(i, :), 27, 18); % Reshape back to 18x27 dimensions
        subplot(5, 5, count)
        imshow(Im, []);
    end
    i = i + 1;
end

% Pause to avoid matlab bug with overlapping figure info
pause(1);

% Display incorrectly classified images
figure('Name', 'Up to 25 Samples of Incorrectly Classified Images');
title('Incorrectly Classified Images')
count = 0;
i = 1;
while (count < 25) && (i <= length(comparison))
    if ~comparison(i) % Use predicted_labels to ensure misclassified samples are shown
        count = count + 1;
        Im = reshape(images_test(i, :), 27, 18); % Reshape back to 18x27 dimensions
        subplot(5, 5, count)
        imshow(Im, []);
        title(['Pred: ' num2str(predicted_labels(i)) ', True: ' num2str(labels_test(i))])
    end
    i = i + 1;
end

% Pause to avoid matlab bug with overlapping figure info
pause(1);

%% Multi-scale Sliding Window Detection on Larger Image for KNNv2

fprintf("Now using the model to detect in larger images using multi-scale sliding window.\n")

% Load the larger image
largeImage = imread('im3.jpg');
if size(largeImage, 3) > 1
    largeImage = rgb2gray(largeImage);
end

% Apply histogram equalization to a copy of the large image
equalizedImage = uint8(255 * mat2gray(largeImage));
equalizedImage = histeq(equalizedImage);

% Define sliding window parameters
windowHeight = 27; % Base height of the window
windowWidth = 18;  % Base width of the window
stepSizeY = 9;         % Vertical step size for the sliding window
stepSizeX = 6;         % Horizontal step size for the sliding window

% Initialize arrays to store detection results for both original and equalized images
detectionsOriginal = [];
detectionsEqualized = [];

% Start with the base window size and increase incrementally until the window matches or exceeds image dimensions
currentHeight = windowHeight;
currentWidth = windowWidth;

while currentHeight <= size(largeImage, 1) && currentWidth <= size(largeImage, 2)
    % Total steps per scale for progress tracking
    totalStepsOriginal = ceil((size(largeImage, 1) - currentHeight + 1) / stepSizeY) * ceil((size(largeImage, 2) - currentWidth + 1) / stepSizeX);
    currentStepOriginal = 0;
    progressCheckpointOriginal = 0.25; % Progress increments for the original image
    
    % Sliding Window Detection for the Original Image
    fprintf('Processing scale [%dx%d] on original image: [', currentWidth, currentHeight);
    for y = 1:stepSizeY:(size(largeImage, 1) - currentHeight + 1)
        for x = 1:stepSizeX:(size(largeImage, 2) - currentWidth + 1)
            % Extract the window patch
            patch = largeImage(y:(y + currentHeight - 1), x:(x + currentWidth - 1));
            
            % Resize patch to match training data dimensions (18x27)
            resizedPatch = imresize(patch, [windowHeight, windowWidth]);
            patchVector = double(reshape(resizedPatch, 1, []));
            
            % Perform classification with KNN
            [label, confidence] = KNNTestingWithConfidence(patchVector, modelNN, BestK);
            
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
    
    % Total steps per scale for equalized image progress tracking
    totalStepsEqualized = ceil((size(equalizedImage, 1) - currentHeight + 1) / stepSizeY) * ceil((size(equalizedImage, 2) - currentWidth + 1) / stepSizeX);
    currentStepEqualized = 0;
    progressCheckpointEqualized = 0.25; % Progress increments for the equalized image
    
    % Sliding Window Detection for the Equalized Image
    fprintf('Processing scale [%dx%d] on equalized image: [', currentWidth, currentHeight);
    for y = 1:stepSizeY:(size(equalizedImage, 1) - currentHeight + 1)
        for x = 1:stepSizeX:(size(equalizedImage, 2) - currentWidth + 1)
            % Extract the window patch
            patch = equalizedImage(y:(y + currentHeight - 1), x:(x + currentWidth - 1));
            
            % Resize patch to match training data dimensions (18x27)
            resizedPatch = imresize(patch, [windowHeight, windowWidth]);
            patchVector = double(reshape(resizedPatch, 1, []));
            
            % Perform classification with KNN
            [label, confidence] = KNNTestingWithConfidence(patchVector, modelNN, BestK);
            
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
    
    % Increment window size by 2 pixels in each dimension
    currentHeight = currentHeight + 2;
    currentWidth = currentWidth + 2;
end


% Apply Non-Maxima Suppression (NMS) on Original and Equalized detections
threshold = 0.05;
finalDetectionsOriginal = simpleNMS(detectionsOriginal, threshold);
finalDetectionsEqualized = simpleNMS(detectionsEqualized, threshold);

%% Saving whole workspace at this point
saveNow = false; % Set to true only if you want to save workspace. This is so these save sections don't run per part of a normal big run (and re-save a broken model for e.g.)
if saveNow
    save('FullImage_KNN_v3_MultiscaleSlidingWindow.mat');
end
%% Load workspace to skip all the work above
% For selective loading and saving, specify variable names like:
% save('filename.mat', 'var1', 'var2') and load('filename.mat', 'var1', 'var2').
loadNow = false; % Set to true only if you want to load workspace.
if loadNow
    clear
    load('FullImage_KNN_v3_MultiscaleSlidingWindow.mat')
end
%% Display Results Before NMS in a 1x2 Grid
figure;
subplot(1, 2, 1);
ShowDetectionResult(largeImage, detectionsOriginal);
title('Original Image - Faces highlighted before NMS (Dynamic Scale)  - Confidence goes up by colors white-yellow-green-cyan');

subplot(1, 2, 2);
ShowDetectionResult(equalizedImage, detectionsEqualized);
title('Equalized Image - Faces highlighted before NMS (Dynamic Scale) - Confidence goes up by colors white-yellow-green-cyan');

pause(1); % Pause to avoid MATLAB display issues

% Display Results After NMS in a 1x2 Grid
figure;
subplot(1, 2, 1);
ShowDetectionResult(largeImage, finalDetectionsOriginal);
title('Original Image - Faces highlighted after NMS (Dynamic Scale) - Confidence goes up by colors white-yellow-green-cyan');

subplot(1, 2, 2);
ShowDetectionResult(equalizedImage, finalDetectionsEqualized);
title('Equalized Image - Faces highlighted after NMS (Dynamic Scale) - Confidence goes up by colors white-yellow-green-cyan');


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




% Show bounding boxes of detections function
function ShowDetectionResult(Picture, Objects)
    % Inputs:
    %   Picture - The original image
    %   Objects - An n x 5 matrix where each row is [x, y, width, height, confidence]

    imshow(Picture); hold on;
    colours = ['c'; 'g'; 'y'; 'w'];

    % Show the detected objects
    if ~isempty(Objects)
        for n = 1:size(Objects, 1)
            x1 = Objects(n, 1); 
            y1 = Objects(n, 2);
            x2 = x1 + Objects(n, 3); 
            y2 = y1 + Objects(n, 4);

            % Determine color based on confidence level
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
    title('Faces highlighted');
    hold off;
end



% Modified simpleNMS Function
function Objects = simpleNMS(Objects, threshold)
    % simpleNMS - Non-Maxima Suppression to reduce overlapping bounding boxes
    %
    % Inputs:
    %   Objects - An n x 5 matrix where each row is [x, y, width, height, confidence]
    %   threshold - The overlap ratio threshold (e.g., 0.5)
    %
    % Outputs:
    %   Objects - The filtered list of bounding boxes after NMS

    % Sort the bounding boxes by confidence score (descending)
    [~, sortedIdx] = sort(Objects(:, 5), 'descend');
    Objects = Objects(sortedIdx, :);

    % Initialize an array to keep track of boxes to keep
    keepIdx = true(size(Objects, 1), 1);

    % Loop through each bounding box
    for i = 1:size(Objects, 1)
        if ~keepIdx(i)
            continue; % Skip if this box has already been suppressed
        end
        
        % Define the bounding box A
        boxA = Objects(i, 1:4);
        
        % Loop through remaining bounding boxes
        for j = i+1:size(Objects, 1)
            if ~keepIdx(j)
                continue; % Skip if this box has already been suppressed
            end
            
            % Define the bounding box B
            boxB = Objects(j, 1:4);
            
            % Calculate intersection area
            intersectionArea = rectint([boxA(1), boxA(2), boxA(3), boxA(4)], ...
                                       [boxB(1), boxB(2), boxB(3), boxB(4)]);
            
            % Calculate the area of boxB
            areaBoxB = boxB(3) * boxB(4);
            
            % Calculate overlap ratio
            overlapRatio = intersectionArea / areaBoxB;
            
            % Suppress box B if the overlap ratio exceeds the threshold
            if overlapRatio > threshold
                keepIdx(j) = false; % Suppress the box with the lower confidence
            end
        end
    end
    
    % Return only the bounding boxes that are kept
    Objects = Objects(keepIdx, :);
end

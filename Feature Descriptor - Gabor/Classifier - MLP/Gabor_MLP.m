clear;
close all;

%% Gabor -> MLP -> Multiscale Sliding Window Detection with Confidence Scores
%% OPTION: USE HISTOGRAM EQUALIZATION TO PREPROCESS?
applyHistogramEqualization = true;

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

disp('Loaded training images, histeq performed if turned on.');

%% Step 2: Extract Gabor Features for Training Data

numImages = size(images, 1);
gaborFeatureMatrix = zeros(numImages, 19440); % Adjusted for Gabor features

for i = 1:numImages
    image = reshape(images(i, :), [27, 18]);
    gaborFeatureMatrix(i, :) = gabor_feature_vector(image);
end

disp('Gabor features for all training images have been extracted.');

%% Step 3: Define and Train the Multilayer Perceptron (MLP) Model

% Convert labels to categorical for MLP
labelsCategorical = categorical(labels);

% Define MLP architecture
inputSize = [1, 19440, 1]; % Match Gabor feature dimensions
numClasses = 2; % Binary classification
layers = [
    imageInputLayer(inputSize, 'Normalization', 'none')
    fullyConnectedLayer(10000) % Hidden layer (adjust the number of neurons)
    reluLayer
    fullyConnectedLayer(5000) % Another hidden layer (adjust the number of neurons)
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

% Set training options
options = trainingOptions('adam', ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 32, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% Train the MLP
mlpModel = trainNetwork(reshape(gaborFeatureMatrix', [1, size(gaborFeatureMatrix, 2), 1, size(gaborFeatureMatrix, 1)]), labelsCategorical, layers, options);

disp('MLP model trained using Gabor features.');

%% Step 4: Load Testing Data

[images_test, labels_test] = loadFaceImages('face_test.cdataset');

% Perform histogram equalization if enabled
if applyHistogramEqualization
    for i = 1:size(images_test, 1)
        originalImage = reshape(images_test(i, :), [27, 18]);
        originalImage = uint8(255 * mat2gray(originalImage));
        equalizedImage = histeq(originalImage);
        images_test(i, :) = double(reshape(equalizedImage, 1, []));
    end
end

disp('Loaded test images, histeq performed if turned on.');

%% Step 5: Extract Gabor Features for Test Data

numTestImages = size(images_test, 1);
gaborTestFeatureMatrix = zeros(numTestImages, 19440); % Adjusted for Gabor features

for i = 1:numTestImages
    testImage = reshape(images_test(i, :), [27, 18]);
    gaborTestFeatureMatrix(i, :) = gabor_feature_vector(testImage);
end

disp('Gabor features for all testing images have been extracted.');

%% Step 6: MLP Testing

% Predict on test data
predictedLabelsCategorical = classify(mlpModel, reshape(gaborTestFeatureMatrix', [1, size(gaborTestFeatureMatrix, 2), 1, size(gaborTestFeatureMatrix, 1)]));
predicted_labels = double(predictedLabelsCategorical);
predicted_labels(predictedLabelsCategorical == categorical(1)) = 1;
predicted_labels(predictedLabelsCategorical == categorical(-1)) = -1;

comparison = (labels_test == predicted_labels);

% Calculate performance metrics
TP = sum((labels_test == 1) & (predicted_labels == 1));
FP = sum((labels_test == -1) & (predicted_labels == 1));
TN = sum((labels_test == -1) & (predicted_labels == -1));
FN = sum((labels_test == 1) & (predicted_labels == -1));

accuracy = (TP + TN) / length(labels_test);
precision = TP / (TP + FP);
recall = TP / (TP + FN);
f1_score = 2 * (precision * recall) / (precision + recall);

fprintf('\nFinal Evaluation on Test Set:\n');
fprintf('TP: %d, FP: %d, TN: %d, FN: %d\n', TP, FP, TN, FN);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Precision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);
fprintf('F1 Score: %.2f\n', f1_score);

%% Display Correctly and Incorrectly Classified Images

% Display correctly classified images
figure('Name', 'Up to 25 Samples of Correctly Classified Images');
title('Correctly Classified Images');
count = 0; i = 1;
while (count < 25) && (i <= length(comparison))
    if comparison(i)
        count = count + 1;
        Im = reshape(images_test(i, :), [27, 18]);
        subplot(5, 5, count);
        imshow(Im, []);
    end
    i = i + 1;
end

pause(1);

% Display incorrectly classified images
figure('Name', 'Up to 25 Samples of Incorrectly Classified Images');
title('Incorrectly Classified Images');
count = 0; i = 1;
while (count < 25) && (i <= length(comparison))
    if ~comparison(i)
        count = count + 1;
        Im = reshape(images_test(i, :), [27, 18]);
        subplot(5, 5, count);
        imshow(Im, []);
        title(['Pred: ' num2str(predicted_labels(i)) ', True: ' num2str(labels_test(i))]);
    end
    i = i + 1;
end

pause(1);

%% Multi-scale Sliding Window Detection on Larger Image for MLP

fprintf("Now using the model to detect in larger images using multi-scale sliding window.\n Will take about 10 minutes. \n")

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
            
            % Resize patch to match training data dimensions (27x18) and extract Gabor features
            resizedPatch = imresize(patch, [windowHeight, windowWidth]);
            patchGabor = gabor_feature_vector(resizedPatch);

            % Predict using the MLP model
            [label, scores] = classify(mlpModel, reshape(patchGabor, [1, 19440, 1, 1]));
            confidence = max(scores);

            % Convert categorical label to numeric for comparison
            labelNumeric = double(label);

            % If a face (or positive detection) is found, store it
            if labelNumeric == 1
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
            
            % Resize patch to match training data dimensions (27x18) and extract Gabor features
            resizedPatch = imresize(patch, [windowHeight, windowWidth]);
            patchGabor = gabor_feature_vector(resizedPatch);
            
            % Predict using the MLP model
            [label, scores] = classify(mlpModel, reshape(patchGabor, [1, 19440, 1, 1]));
            confidence = max(scores);

            % Convert categorical label to numeric for comparison
            labelNumeric = double(label);

            % If a face (or positive detection) is found, store it
            if labelNumeric == 1
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
threshold = 0.05;
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
    save('Gabor_MLP_15000Neurons_30Epochs.mat');
end
%% Load workspace to skip all the work above
% For selective loading and saving, specify variable names like:
% save('filename.mat', 'var1', 'var2') and load('filename.mat', 'var1', 'var2').
loadNow = false; % Set to true only if you want to load workspace.
if loadNow
    clear
    load('Gabor_MLP_15000Neurons_30Epochs.mat')
end
%% Functions

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

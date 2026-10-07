clear;
close all;

%% Gabor -> Random Forest -> Multiscale Sliding Window Detection
%% OPTION: USE HISTOGRAM EQUALIZATION TO PREPROCESS?
applyHistogramEqualization = false; % Set to true to apply histogram equalization, false to skip it

%% Step 1: Load Training Data
% Load face images and labels
[images, labels] = loadFaceImages('face_train.cdataset');

% Apply histogram equalization to training images if enabled
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
imageHeight = 27;
imageWidth = 18;
numImages = size(images, 1);

% Initialize matrix to store Gabor features for each image
gaborFeatureMatrix = zeros(numImages, length(gabor_feature_vector(reshape(images(1, :), [imageHeight, imageWidth]))));

for i = 1:numImages
    image = reshape(images(i, :), [imageHeight, imageWidth]);
    gaborFeatureMatrix(i, :) = gabor_feature_vector(image);
end

disp('Gabor features for all training images have been extracted.');

%% Step 3: Train Random Forest Model
numTrees = 100; % Number of trees in the Random Forest
modelRF = TreeBagger(numTrees, gaborFeatureMatrix, labels, 'OOBPrediction', 'On', 'Method', 'classification');
disp('Random Forest model created with Gabor features.');

%% Step 4: Load and Process Testing Data
[images_test, labels_test] = loadFaceImages('face_test.cdataset');

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
gaborTestFeatureMatrix = zeros(numTestImages, size(gaborFeatureMatrix, 2));

for i = 1:numTestImages
    testImage = reshape(images_test(i, :), [imageHeight, imageWidth]);
    gaborTestFeatureMatrix(i, :) = gabor_feature_vector(testImage);
end

disp('Gabor features for all testing images have been extracted.');

%% Step 6: Random Forest Testing on Test Data
disp('Now testing on the Random Forest model using test data.');

[predicted_labels, scores] = predict(modelRF, gaborTestFeatureMatrix);
predicted_labels = str2double(predicted_labels); % Convert labels from cell array to numeric array
confidences = max(scores, [], 2); % Confidence is the maximum score

% Calculate evaluation metrics
comparison = (labels_test == predicted_labels);
accuracy = sum(comparison) / length(labels_test);
TP = sum((labels_test == 1) & (predicted_labels == 1));
FP = sum((labels_test == -1) & (predicted_labels == 1));
TN = sum((labels_test == -1) & (predicted_labels == -1));
FN = sum((labels_test == 1) & (predicted_labels == -1));
precision = TP / (TP + FP);
recall = TP / (TP + FN);
f1_score = 2 * (precision * recall) / (precision + recall);

fprintf('\nFinal Evaluation on Test Set:\n');
fprintf('TP: %d, FP: %d, TN: %d, FN: %d\n', TP, FP, TN, FN);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Precision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);
fprintf('F1 Score: %.2f\n', f1_score);

% Display correctly and incorrectly classified images
figure('Name', 'Up to 25 Samples of Correctly Classified Images');
title('Correctly Classified Images');
count = 0; i = 1;
while (count < 25) && (i <= length(comparison))
    if comparison(i)
        count = count + 1;
        Im = reshape(images_test(i, :), [imageHeight, imageWidth]);
        subplot(5, 5, count);
        imshow(Im, []);
    end
    i = i + 1;
end

pause(1);

figure('Name', 'Up to 25 Samples of Incorrectly Classified Images');
title('Incorrectly Classified Images');
count = 0; i = 1;
while (count < 25) && (i <= length(comparison))
    if ~comparison(i)
        count = count + 1;
        Im = reshape(images_test(i, :), [imageHeight, imageWidth]);
        subplot(5, 5, count);
        imshow(Im, []);
        title(['Pred: ' num2str(predicted_labels(i)) ', True: ' num2str(labels_test(i))]);
    end
    i = i + 1;
end

pause(1);

%% Multi-scale Sliding Window Detection on Larger Image

fprintf("Now using the model to detect in larger images using multi-scale sliding window.\n");

largeImage = imread('im3.jpg');
if size(largeImage, 3) > 1
    largeImage = rgb2gray(largeImage);
end
equalizedImage = uint8(255 * mat2gray(largeImage));
equalizedImage = histeq(equalizedImage);

windowHeight = 27;
windowWidth = 18;
stepSizeY = 9;
stepSizeX = 6;

detectionsOriginal = [];
detectionsEqualized = [];

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
            patch = largeImage(y:(y + currentHeight - 1), x:(x + currentWidth - 1));
            resizedPatch = imresize(patch, [windowHeight, windowWidth]);
            patchGabor = gabor_feature_vector(resizedPatch);
            [label, scores] = predict(modelRF, patchGabor);
            confidence = max(scores);
            if str2double(label) == 1
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
    fprintf('100%%]\n');

    % Total steps per scale for equalized image progress tracking
    totalStepsEqualized = ceil((size(equalizedImage, 1) - currentHeight + 1) / stepSizeY) * ceil((size(equalizedImage, 2) - currentWidth + 1) / stepSizeX);
    currentStepEqualized = 0;
    progressCheckpointEqualized = 0.25; % Progress increments for the equalized image

    % Sliding Window Detection for the Equalized Image
    fprintf('Processing scale [%dx%d] on equalized image: [', currentWidth, currentHeight);
    for y = 1:stepSizeY:(size(equalizedImage, 1) - currentHeight + 1)
        for x = 1:stepSizeX:(size(equalizedImage, 2) - currentWidth + 1)
            patch = equalizedImage(y:(y + currentHeight - 1), x:(x + currentWidth - 1));
            resizedPatch = imresize(patch, [windowHeight, windowWidth]);
            patchGabor = gabor_feature_vector(resizedPatch);
            [label, scores] = predict(modelRF, patchGabor);
            confidence = max(scores);
            if str2double(label) == 1
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
    fprintf('100%%]\n');
    
    currentHeight = currentHeight + 2;
    currentWidth = currentWidth + 2;
end

%% Apply Non-Maxima Suppression (NMS)
threshold = 0.1;
finalDetectionsOriginal = simpleNMS(detectionsOriginal, threshold);
finalDetectionsEqualized = simpleNMS(detectionsEqualized, threshold);

%% Display Results Before and After NMS
figure;
subplot(1, 2, 1);
ShowDetectionResult(largeImage, detectionsOriginal);
title('Original Image - Faces highlighted before NMS - Confidence goes up by colors white-yellow-green-cyan');

subplot(1, 2, 2);
ShowDetectionResult(equalizedImage, detectionsEqualized);
title('Equalized Image - Faces highlighted before NMS - Confidence goes up by colors white-yellow-green-cyan');

pause(1);

figure;
subplot(1, 2, 1);
ShowDetectionResult(largeImage, finalDetectionsOriginal);
title('Original Image - Faces highlighted after NMS - Confidence goes up by colors white-yellow-green-cyan');

subplot(1, 2, 2);
ShowDetectionResult(equalizedImage, finalDetectionsEqualized);
title('Equalized Image - Faces highlighted after NMS - Confidence goes up by colors white-yellow-green-cyan');

%% Saving whole workspace at this point
saveNow = false; % Set to true only if you want to save workspace. This is so these save sections don't run per part of a normal big run (and re-save a broken model for e.g.)
if saveNow
    save('Gabor_RF_BROKEN.mat');
end
%% Load workspace to skip all the work above
% For selective loading and saving, specify variable names like:
% save('filename.mat', 'var1', 'var2') and load('filename.mat', 'var1', 'var2').
loadNow = false; % Set to true only if you want to load workspace.
if loadNow
    clear
    load('Gabor_RF_BROKEN.mat')
end
%% Functions


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

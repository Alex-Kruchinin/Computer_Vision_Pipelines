clear 
close all

%% LBP -> MLP -> Multiscale Sliding Window Detection with Confidence Score
%% OPTION: USE HISTOGRAM EQUALIZATION TO PREPROCESS?
applyHistogramEqualization = false;

%% Step 1: Load Training Data and Extract LBP Features

% Load face images and labels
[images, labels] = loadFaceImages('face_train.cdataset');

% Extract LBP features for training data
lbpFeaturesTrain = zeros(size(images, 1), 256); % Assuming 256 bins for LBP histogram
for i = 1:size(images, 1)
    originalImage = reshape(images(i, :), [27, 18]); % Reshape to original dimensions
    
    % Perform histogram equalization if enabled
    if applyHistogramEqualization
        originalImage = uint8(255 * mat2gray(originalImage)); % Normalize for equalization
        originalImage = histeq(originalImage);
    end
    
    % Compute LBP features
    lbpImage = computeLBP(originalImage); % Call the LBP function
    lbpHistogram = histcounts(lbpImage(:), 0:256); % Compute histogram (256 bins)
    lbpHistogram = lbpHistogram / sum(lbpHistogram); % Normalize histogram
    lbpFeaturesTrain(i, :) = lbpHistogram;
end

disp('Training images loaded and LBP features extracted.')

% Reshape LBP features for the MLP
lbpFeaturesTrain = reshape(lbpFeaturesTrain', [16, 16, 1, size(lbpFeaturesTrain, 1)]); % Match expected input size

%% Step 2: Define and Train the MLP Model

% Convert labels to categorical for training
labelsCategorical = categorical(labels);

% Define MLP architecture
inputSize = [16, 16, 1]; % Match reshaped LBP input size
numClasses = 2; % Binary classification (face vs. non-face)
layers = [
    imageInputLayer(inputSize, 'Normalization', 'zscore')
    fullyConnectedLayer(10000) % Hidden layer (adjust the number of neurons)
    reluLayer
    fullyConnectedLayer(5000) % Another hidden layer (adjust the number of neurons)
    reluLayer
    fullyConnectedLayer(numClasses) % Final classification layer
    softmaxLayer
    classificationLayer];

% Set training options
options = trainingOptions('adam', ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 32, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% Train the MLP
mlpModel = trainNetwork(lbpFeaturesTrain, labelsCategorical, layers, options);

disp('MLP model trained using LBP features.')

%% Step 3: Load Testing Data and Extract LBP Features

% Load test face images and labels
[images_test, labels_test] = loadFaceImages('face_test.cdataset');

% Extract LBP features for testing data
lbpFeaturesTest = zeros(size(images_test, 1), 256); % Assuming 256 bins for LBP histogram
for i = 1:size(images_test, 1)
    originalImage = reshape(images_test(i, :), [27, 18]); % Reshape to original dimensions
    
    % Perform histogram equalization if enabled
    if applyHistogramEqualization
        originalImage = uint8(255 * mat2gray(originalImage));
        originalImage = histeq(originalImage);
    end
    
    % Compute LBP features
    lbpImage = computeLBP(originalImage); % Call the LBP function
    lbpHistogram = histcounts(lbpImage(:), 0:256); % Compute histogram (256 bins)
    lbpHistogram = lbpHistogram / sum(lbpHistogram); % Normalize histogram
    lbpFeaturesTest(i, :) = lbpHistogram;
end

% Reshape LBP features for the MLP
lbpFeaturesTest = reshape(lbpFeaturesTest', [16, 16, 1, size(lbpFeaturesTest, 1)]);

disp('Test images loaded and LBP features extracted.')

%% Step 4: MLP Testing

% Predict on test data
predictedLabelsCategorical = classify(mlpModel, lbpFeaturesTest);
predicted_labels = double(predictedLabelsCategorical);
predicted_labels(predictedLabelsCategorical == categorical(1)) = 1;
predicted_labels(predictedLabelsCategorical == categorical(-1)) = -1;

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

%% Step 5: Display Correctly and Incorrectly Classified Images
figure('Name', 'Up to 25 Samples of Correctly Classified Images');
title('Correctly Classified Images');
count = 0; i = 1;
while (count < 25) && (i <= length(comparison))
    if comparison(i)
        count = count + 1;
        Im = reshape(images_test(i, :), 27, 18);
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
        Im = reshape(images_test(i, :), 27, 18);
        subplot(5, 5, count);
        imshow(Im, []);
        title(['Pred: ' num2str(predicted_labels(i)) ', True: ' num2str(labels_test(i))]);
    end
    i = i + 1;
end

pause(1);

%% Multi-scale Sliding Window Detection on Larger Image for MLP

fprintf("Now using the model to detect in larger images using multi-scale sliding window.\n");

% Load and preprocess the larger image
largeImage = imread('im3.jpg');
if size(largeImage, 3) > 1
    largeImage = rgb2gray(largeImage);
end

% Apply histogram equalization to the larger image if desired
equalizedImage = uint8(255 * mat2gray(largeImage));
equalizedImage = histeq(equalizedImage);

% Define sliding window parameters
windowHeight = 27; % Base height of the window
windowWidth = 18;  % Base width of the window
stepSizeY = 9;     % Vertical step size for the sliding window
stepSizeX = 6;     % Horizontal step size for the sliding window

% Initialize detection results
detectionsOriginal = [];
detectionsEqualized = [];

% Start with the base window size and increase incrementally
currentHeight = windowHeight;
currentWidth = windowWidth;

while currentHeight <= size(largeImage, 1) && currentWidth <= size(largeImage, 2)
    % Total steps per scale for progress tracking
    totalStepsOriginal = ceil((size(largeImage, 1) - currentHeight + 1) / stepSizeY) * ceil((size(largeImage, 2) - currentWidth + 1) / stepSizeX);
    currentStepOriginal = 0;
    progressCheckpointOriginal = 0.25;

    % Sliding Window Detection for the Original Image
    fprintf('Processing scale [%dx%d] on original image: [', currentWidth, currentHeight);
    for y = 1:stepSizeY:(size(largeImage, 1) - currentHeight + 1)
        for x = 1:stepSizeX:(size(largeImage, 2) - currentWidth + 1)
            % Extract and resize the patch to match training dimensions
            patch = largeImage(y:(y + currentHeight - 1), x:(x + currentWidth - 1));
            resizedPatch = imresize(patch, [16, 16]);
            resizedPatch = reshape(resizedPatch, [16, 16, 1]); % Reshape to match MLP input dimensions
            
            % Predict using the MLP
            label = classify(mlpModel, resizedPatch);
            confidence = max(predict(mlpModel, resizedPatch));

            % If detection is positive, store the results
            if label == categorical(1)
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

    % Sliding Window Detection for the Equalized Image
    totalStepsEqualized = ceil((size(equalizedImage, 1) - currentHeight + 1) / stepSizeY) * ceil((size(equalizedImage, 2) - currentWidth + 1) / stepSizeX);
    currentStepEqualized = 0;
    progressCheckpointEqualized = 0.25;

    fprintf('Processing scale [%dx%d] on equalized image: [', currentWidth, currentHeight);
    for y = 1:stepSizeY:(size(equalizedImage, 1) - currentHeight + 1)
        for x = 1:stepSizeX:(size(equalizedImage, 2) - currentWidth + 1)
            % Extract and resize the patch to match training dimensions
            patch = equalizedImage(y:(y + currentHeight - 1), x:(x + currentWidth - 1));
            resizedPatch = imresize(patch, [16, 16]);
            resizedPatch = reshape(resizedPatch, [16, 16, 1]); % Reshape to match MLP input dimensions
            
            % Predict using the MLP
            label = classify(mlpModel, resizedPatch);
            confidence = max(predict(mlpModel, resizedPatch));

            % If detection is positive, store the results
            if label == categorical(1)
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

    % Increment the window size for the next scale
    currentHeight = currentHeight + 2;
    currentWidth = currentWidth + 2;
end

%% Apply Non-Maxima Suppression (NMS)
threshold = 0.05;
finalDetectionsOriginal = simpleNMS(detectionsOriginal, threshold);
finalDetectionsEqualized = simpleNMS(detectionsEqualized, threshold);

%% Display Results
% Display detection results before and after NMS
figure;
subplot(1, 2, 1);
ShowDetectionResult(largeImage, detectionsOriginal);
title('Original Image - Detections Before NMS');

subplot(1, 2, 2);
ShowDetectionResult(equalizedImage, detectionsEqualized);
title('Equalized Image - Detections Before NMS');

pause(1);

figure;
subplot(1, 2, 1);
ShowDetectionResult(largeImage, finalDetectionsOriginal);
title('Original Image - Detections After NMS');

subplot(1, 2, 2);
ShowDetectionResult(equalizedImage, finalDetectionsEqualized);
title('Equalized Image - Detections After NMS');

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
    save('LBP_MLP_15000Neurons.mat');
end
%% Load workspace to skip all the work above
% For selective loading and saving, specify variable names like:
% save('filename.mat', 'var1', 'var2') and load('filename.mat', 'var1', 'var2').
loadNow = false; % Set to true only if you want to load workspace.
if loadNow
    clear
    load('LBP_MLP_15000Neurons.mat')
end

%% Functions

% LBP Function
function lbpImage = computeLBP(grayImage)
    % Ensure the input image is grayscale
    if size(grayImage, 3) == 3
        grayImage = rgb2gray(grayImage);
    end
    
    % Get the size of the image
    [rows, cols] = size(grayImage);
    
    % Initialize the LBP image
    lbpImage = zeros(rows, cols, 'uint8');
    
    % Pad the image with zeros for boundary processing
    paddedImage = padarray(grayImage, [1, 1], 'replicate');
    
    % Define neighbor offsets for 3x3 window
    offsets = [
        -1 -1; -1 0; -1 1;
         0 -1;        0 1;
         1 -1;  1 0;  1 1
    ];
    
    % Process each pixel in the original image
    for i = 2:rows+1
        for j = 2:cols+1
            % Get the center pixel value
            centerPixel = paddedImage(i, j);
            
            % Initialize the binary pattern
            binaryPattern = zeros(1, 8);
            
            % Loop through all neighbors
            for k = 1:8
                % Get the neighbor pixel value
                neighborPixel = paddedImage(i + offsets(k, 1), j + offsets(k, 2));
                
                % Compare and assign binary value
                binaryPattern(k) = neighborPixel >= centerPixel;
            end
            
            % Convert binary pattern to decimal
            lbpValue = sum(binaryPattern .* (2.^(0:7)));
            
            % Assign the LBP value to the output image
            lbpImage(i-1, j-1) = lbpValue;
        end
    end
end


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

function Objects = simpleNMS(Objects, threshold)
    [~, sortedIdx] = sort(Objects(:, 5), 'descend');
    Objects = Objects(sortedIdx, :);
    keepIdx = true(size(Objects, 1), 1);
    for i = 1:size(Objects, 1)
        if ~keepIdx(i), continue; end
        boxA = Objects(i, 1:4);
        for j = i+1:size(Objects, 1)
            if ~keepIdx(j), continue; end
            boxB = Objects(j, 1:4);
            intersectionArea = rectint([boxA(1), boxA(2), boxA(3), boxA(4)], ...
                                       [boxB(1), boxB(2), boxB(3), boxB(4)]);
            areaBoxB = boxB(3) * boxB(4);
            overlapRatio = intersectionArea / areaBoxB;
            if overlapRatio > threshold
                keepIdx(j) = false;
            end
        end
    end
    Objects = Objects(keepIdx, :);
end
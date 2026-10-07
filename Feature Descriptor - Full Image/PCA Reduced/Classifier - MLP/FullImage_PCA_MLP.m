clear;
close all;

%% Full Image -> PCA -> MLP -> Multiscale Sliding Window Detection with Confidence Score
%% OPTION: USE HISTOGRAM EQUALIZATION TO PREPROCESS?
applyHistogramEqualization = false; % Set to true to apply histogram equalization, false to skip it

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

%% Step 2: Apply PCA for Dimensionality Reduction

% Define the number of dimensions to reduce to with PCA
nReducedDimensions = 200; 
[U, ~, imagesPCA] = pca(images, nReducedDimensions);

disp(['Applied PCA to reduce dimensions to ', num2str(nReducedDimensions)])

%% Step 3: Train MLP Model with PCA-Reduced Data

% Reshape PCA data for MLP input
imagesPCA_MLP = reshape(imagesPCA', [nReducedDimensions, 1, 1, size(imagesPCA, 1)]);

% Convert labels to categorical for training
labelsCategorical = categorical(labels);

% Define MLP architecture
inputSize = [nReducedDimensions, 1, 1]; % Match PCA-reduced size
numClasses = 2; % Binary classification (face vs. non-face)
layers = [
    imageInputLayer(inputSize, 'Normalization', 'none')
    fullyConnectedLayer(1000) % Hidden layer (adjust the number of neurons)
    reluLayer
    fullyConnectedLayer(500) % Another hidden layer (adjust the number of neurons)
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

% Set training options
options = trainingOptions('adam', ...
    'MaxEpochs', 3, ...
    'MiniBatchSize', 32, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% Train the MLP
mlpModel = trainNetwork(imagesPCA_MLP, labelsCategorical, layers, options);

disp('MLP model trained using PCA-reduced data.')

%% Step 4: Load Testing Data

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

% Apply PCA to test data using the trained PCA eigenvectors (U)
images_testPCA = images_test * U(:, 1:nReducedDimensions);

% Reshape PCA data for MLP input
images_testPCA_MLP = reshape(images_testPCA', [nReducedDimensions, 1, 1, size(images_testPCA, 1)]);

disp('Loaded and PCA-transformed test images, histeq performed if turned on.')

%% Step 5: MLP Testing

% Predict on test data
predictedLabelsCategorical = classify(mlpModel, images_testPCA_MLP);
predicted_labels = double(predictedLabelsCategorical);

% Fix broken data
predicted_labels(predictedLabelsCategorical == categorical(-1)) = -1;
predicted_labels(predictedLabelsCategorical == categorical(1)) = 1;


% Create the comparison array to identify correct and incorrect classifications
comparison = (labels_test == predicted_labels);

% Initialize counters for TP, FP, TN, FN
TP = sum((labels_test == 1) & (predicted_labels == 1));
FP = sum((labels_test == -1) & (predicted_labels == 1));
TN = sum((labels_test == -1) & (predicted_labels == -1));
FN = sum((labels_test == 1) & (predicted_labels == -1));

% Calculate evaluation metrics
accuracy = (TP + TN) / length(labels_test);
precision = TP / (TP + FP);
recall = TP / (TP + FN);
f1_score = 2 * (precision * recall) / (precision + recall);

% Display final metrics
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
count = 0;
i = 1;
while (count < 25) && (i <= length(comparison))
    if comparison(i)
        count = count + 1;
        Im = reshape(images_test(i, :), 27, 18); % Reshape back to original dimensions
        subplot(5, 5, count)
        imshow(Im, []);
    end
    i = i + 1;
end

pause(1);

% Display incorrectly classified images
figure('Name', 'Up to 25 Samples of Incorrectly Classified Images');
title('Incorrectly Classified Images');
count = 0;
i = 1;
while (count < 25) && (i <= length(comparison))
    if ~comparison(i) % Show misclassified samples
        count = count + 1;
        Im = reshape(images_test(i, :), 27, 18); % Reshape back to original dimensions
        subplot(5, 5, count)
        imshow(Im, []);
        title(['Pred: ' num2str(predicted_labels(i)) ', True: ' num2str(labels_test(i))])
    end
    i = i + 1;
end

pause(1);

%% Multi-scale Sliding Window Detection on Larger Image for MLP with PCA

fprintf("Now using the model to detect in larger images using multi-scale sliding window.\n");

% Load the larger image
largeImage = imread('im3.jpg');
if size(largeImage, 3) > 1
    largeImage = rgb2gray(largeImage);
end

% Apply histogram equalization to a copy of the large image
equalizedImage = uint8(255 * mat2gray(largeImage));
equalizedImage = histeq(equalizedImage);

% Initialize sliding window parameters
windowHeight = 27; % Base height of the window
windowWidth = 18;  % Base width of the window
stepSizeY = 9;     % Vertical step size for the sliding window
stepSizeX = 6;     % Horizontal step size for the sliding window

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
            
            % Apply PCA to reduce the patch's dimensions
            patchVectorPCA = patchVector * U(:, 1:nReducedDimensions);
            patchVectorPCA_MLP = reshape(patchVectorPCA, [nReducedDimensions, 1, 1]);
            
            % Predict using MLP
            [label, scores] = classify(mlpModel, patchVectorPCA_MLP);
            confidence = max(scores);
            
            % If a face (or positive detection) is found, store it
            if double(label) == 1
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
    
    % Sliding Window Detection for the Equalized Image
    totalStepsEqualized = ceil((size(equalizedImage, 1) - currentHeight + 1) / stepSizeY) * ceil((size(equalizedImage, 2) - currentWidth + 1) / stepSizeX);
    currentStepEqualized = 0;
    progressCheckpointEqualized = 0.25;
    
    fprintf('Processing scale [%dx%d] on equalized image: [', currentWidth, currentHeight);
    for y = 1:stepSizeY:(size(equalizedImage, 1) - currentHeight + 1)
        for x = 1:stepSizeX:(size(equalizedImage, 2) - currentWidth + 1)
            % Extract the window patch
            patch = equalizedImage(y:(y + currentHeight - 1), x:(x + currentWidth - 1));
            
            % Resize patch to match training data dimensions (18x27)
            resizedPatch = imresize(patch, [windowHeight, windowWidth]);
            patchVector = double(reshape(resizedPatch, 1, []));
            
            % Apply PCA to reduce the patch's dimensions
            patchVectorPCA = patchVector * U(:, 1:nReducedDimensions);
            patchVectorPCA_MLP = reshape(patchVectorPCA, [nReducedDimensions, 1, 1]);
            
            % Predict using MLP
            [label, scores] = classify(mlpModel, patchVectorPCA_MLP);
            confidence = max(scores);
            
            % If a face (or positive detection) is found, store it
            if double(label) == 1
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

%% Apply Non-Maxima Suppression (NMS) on Original and Equalized detections
threshold = 0.05;
finalDetectionsOriginal = simpleNMS(detectionsOriginal, threshold);
finalDetectionsEqualized = simpleNMS(detectionsEqualized, threshold);

%% Display Results Before and After NMS

figure;
subplot(1, 2, 1);
ShowDetectionResult(largeImage, detectionsOriginal);
title('Original Image - Faces highlighted before NMS (Dynamic Scale) - Confidence goes up by colors white-yellow-green-cyan');

subplot(1, 2, 2);
ShowDetectionResult(equalizedImage, detectionsEqualized);
title('Equalized Image - Faces highlighted before NMS (Dynamic Scale) - Confidence goes up by colors white-yellow-green-cyan');

pause(1); % Pause to avoid MATLAB display issues

% Display Results After NMS
figure;
subplot(1, 2, 1);
ShowDetectionResult(largeImage, finalDetectionsOriginal);
title('Original Image - Faces highlighted after NMS (Dynamic Scale) - Confidence goes up by colors white-yellow-green-cyan');

subplot(1, 2, 2);
ShowDetectionResult(equalizedImage, finalDetectionsEqualized);
title('Equalized Image - Faces highlighted after NMS (Dynamic Scale) - Confidence goes up by colors white-yellow-green-cyan');

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
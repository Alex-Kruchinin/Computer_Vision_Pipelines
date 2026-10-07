clear 
close all

%% Full Image -> PCA -> Decision Tree -> Multiscale Sliding Window Detection with Confidence Score
%% OPTION: USE HISTOGRAM EQUALIZATION TO PREPROCESS?
applyHistogramEqualization = false; % Set to true to apply histogram equalization, false to skip it

% OFF, 50 Dimensions:
%TP: 154, FP: 16, TN: 24, FN: 46 | Accuracy: 74.17% | Precision: 0.91 | Recall: 0.77 | F1 Score: 0.83

% OFF, 100 Dimensions:
% TP: 148, FP: 16, TN: 24, FN: 52 | Accuracy: 71.67% | Precision: 0.90 | Recall: 0.74 | F1 Score: 0.81

% OFF, 200 Dimensions:
% TP: 148, FP: 16, TN: 24, FN: 52 | Accuracy: 71.67% | Precision: 0.90 | Recall: 0.74 | F1 Score: 0.81

%% Step 1: Load Training Data
% Load face images and labels
[images, labels] = loadFaceImages('face_train.cdataset');

% Apply histogram equalization to training images if enabled
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

%% Step 2: Apply PCA for Dimensionality Reduction
nReducedDimensions = 50; % Define the number of dimensions to reduce to with PCA
[U, S, imagesPCA] = pca(images, nReducedDimensions);

disp(['Applied PCA to reduce dimensions to ', num2str(nReducedDimensions)])

%% Step 3: Train Decision Tree Model with PCA-reduced Data
modelTree = fitctree(imagesPCA, labels, 'SplitCriterion', 'gdi', 'MaxNumSplits', 100); 
disp('Standalone Decision Tree trained with PCA-reduced training data.')

%% Step 4: Load Testing Data
[images_test, labels_test] = loadFaceImages('face_test.cdataset');

% Apply histogram equalization to test images if enabled
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

disp('Loaded and PCA-transformed test images, histeq performed if turned on.')

%% Step 5: Decision Tree Testing on Test Data with Confidence Calculation
disp('Now testing on the Decision Tree model using PCA-reduced test data.')

predicted_labels = zeros(size(labels_test));
confidences = zeros(size(labels_test));

for i = 1:size(images_testPCA, 1)
    testImage = images_testPCA(i, :);
    [predicted_label, nodeScores] = predict(modelTree, testImage);
    predicted_labels(i) = predicted_label; 
    confidences(i) = max(nodeScores); % Confidence is max probability of prediction
end

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
    if ~comparison(i)
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

%% Multi-scale Sliding Window Detection on Larger Image for Decision Tree with PCA

fprintf("Now using the model to detect in larger images using multi-scale sliding window.\n")

% Load and equalize large image
largeImage = imread('im3.jpg');
if size(largeImage, 3) > 1
    largeImage = rgb2gray(largeImage);
end
equalizedImage = uint8(255 * mat2gray(largeImage));
equalizedImage = histeq(equalizedImage);

% Define sliding window parameters
windowHeight = 27;
windowWidth = 18;
stepSizeY = 9;
stepSizeX = 6;

% Initialize detection results
detectionsOriginal = [];
detectionsEqualized = [];

% Start with base window size, increase incrementally until window matches/exceeds image dimensions
currentHeight = windowHeight;
currentWidth = windowWidth;

while currentHeight <= size(largeImage, 1) && currentWidth <= size(largeImage, 2)
    % Progress tracking setup for original image
    totalStepsOriginal = ceil((size(largeImage, 1) - currentHeight + 1) / stepSizeY) * ceil((size(largeImage, 2) - currentWidth + 1) / stepSizeX);
    currentStepOriginal = 0;
    progressCheckpointOriginal = 0.25;

    % Sliding Window Detection on Original Image
    fprintf('Processing scale [%dx%d] on original image: [', currentWidth, currentHeight);
    for y = 1:stepSizeY:(size(largeImage, 1) - currentHeight + 1)
        for x = 1:stepSizeX:(size(largeImage, 2) - currentWidth + 1)
            patch = largeImage(y:(y + currentHeight - 1), x:(x + currentWidth - 1));
            resizedPatch = imresize(patch, [windowHeight, windowWidth]);
            patchVector = double(reshape(resizedPatch, 1, []));
            
            % Apply PCA to reduce the patch's dimensions
            patchVectorPCA = patchVector * U(:, 1:nReducedDimensions);
            
            % Perform classification with Decision Tree
            [label, nodeScores] = predict(modelTree, patchVectorPCA);
            confidence = max(nodeScores);
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
    fprintf('100%%]\n');

    % Sliding Window Detection on Equalized Image
    totalStepsEqualized = ceil((size(equalizedImage, 1) - currentHeight + 1) / stepSizeY) * ceil((size(equalizedImage, 2) - currentWidth + 1) / stepSizeX);
    currentStepEqualized = 0;
    progressCheckpointEqualized = 0.25;
    
    fprintf('Processing scale [%dx%d] on equalized image: [', currentWidth, currentHeight);
    for y = 1:stepSizeY:(size(equalizedImage, 1) - currentHeight + 1)
        for x = 1:stepSizeX:(size(equalizedImage, 2) - currentWidth + 1)
            patch = equalizedImage(y:(y + currentHeight - 1), x:(x + currentWidth - 1));
            resizedPatch = imresize(patch, [windowHeight, windowWidth]);
            patchVector = double(reshape(resizedPatch, 1, []));
            
            % Apply PCA to reduce the patch's dimensions
            patchVectorPCA = patchVector * U(:, 1:nReducedDimensions);
            
            % Perform classification with Decision Tree
            [label, nodeScores] = predict(modelTree, patchVectorPCA);
            confidence = max(nodeScores);
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
    fprintf('100%%]\n');
    
    % Increment window size by 2 pixels in each dimension
    currentHeight = currentHeight + 2;
    currentWidth = currentWidth + 2;
end

%% Apply Non-Maxima Suppression
threshold = 0.01;
finalDetectionsOriginal = simpleNMS(detectionsOriginal, threshold);
finalDetectionsEqualized = simpleNMS(detectionsEqualized, threshold);

%% Display Results Before NMS in a 1x2 Grid
figure;
subplot(1, 2, 1);
ShowDetectionResult(largeImage, detectionsOriginal);
title('Original Image - Faces highlighted before NMS (Dynamic Scale) - Confidence goes up by colors white-yellow-green-cyan');

subplot(1, 2, 2);
ShowDetectionResult(equalizedImage, detectionsEqualized);
title('Equalized Image - Faces highlighted before NMS (Dynamic Scale) - Confidence goes up by colors white-yellow-green-cyan');

pause(1);

% Display Results After NMS in a 1x2 Grid
figure;
subplot(1, 2, 1);
ShowDetectionResult(largeImage, finalDetectionsOriginal);
title('Original Image - Faces highlighted after NMS (Dynamic Scale) - Confidence goes up by colors white-yellow-green-cyan');

subplot(1, 2, 2);
ShowDetectionResult(equalizedImage, finalDetectionsEqualized);
title('Equalized Image - Faces highlighted after NMS (Dynamic Scale) - Confidence goes up by colors white-yellow-green-cyan');



%% Functions

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

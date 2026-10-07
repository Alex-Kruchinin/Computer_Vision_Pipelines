clear 
close all

%% Full Image -> Random Forests -> Multiscale Sliding Window Detection with Confidence Score
%% OPTION: USE HISTOGRAM EQUALIZATION TO PREPROCESS?
applyHistogramEqualization = false; % Set to true to apply histogram equalization, false to skip it
% In my testing, at 100 Trees, both had the same the accuracy of 75%. But OFF had better Recall and F1 Score.
% ON:OFF Precision 0.95:0.89   |   Recall 0.74:0.8    |   F1 Score 0.83:0.84
% In terms of face detection, and depending on what you want, OFF might be
% the better model, due to the higher recall. A higher recall at the cost
% of precision means you get less FN at the cost of more FP.
% In other words, you might get more results of non-faces, but you'll
% capture more faces that you would've lost otherwise, which is important
% for example if you want to make sure you catch a missing criminal and 
% you don't mind sifting through the extra FP results. It really
% depends on the application.

% At 10 Trees, Off was definitely better.:
% On: Accuracy: 69.58% | Precision: 0.97 | Recall: 0.66 | F1 Score: 0.78
% Off: Accuracy: 72.08% | Precision: 0.92 | Recall: 0.72 | F1 Score: 0.81
% Off: Accuracy: 74.58% | Precision: 0.90 | Recall: 0.79 | F1 Score: 0.84
% Off: Accuracy: 72.50% | Precision: 0.90 | Recall: 0.75 | F1 Score: 0.82
% OFF: Accuracy: 77.50% | Precision: 0.90 | Recall: 0.82 | F1 Score: 0.86

% At 200 Trees, Off was definitely better.
% On: Accuracy: 76.67% | Precision: 0.99 | Recall: 0.73 | F1 Score: 0.84
% Off: Accuracy: 77.08% | Precision: 0.91 | Recall: 0.81 | F1 Score: 0.85
% Off: Accuracy: 76.25% | Precision: 0.89 | Recall: 0.81 | F1 Score: 0.85
% Off: Accuracy: 75.83% | Precision: 0.89 | Recall: 0.81 | F1 Score: 0.85

% 300 Trees, Off better:
% On: Accuracy: 76.67% | Precision: 0.97 | Recall: 0.74 | F1 Score: 0.84
% Off: Accuracy: 77.50% | Precision: 0.91 | Recall: 0.81 | F1 Score: 0.86

% 500 Trees:

% Off: Accuracy: 75.42% | Precision: 0.89 | Recall: 0.80 | F1 Score: 0.84

% 1000 Trees:
% Off: Accuracy: 76.25% | Precision: 0.90 | Recall: 0.81 | F1 Score: 0.85

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

disp('Loaded training images, histeq performed if turned on.')

%% Step 2: Train Random Forest Model
numTrees = 100; % Number of trees in the Random Forest
modelRF = TreeBagger(numTrees, images, labels, 'Method', 'classification');

fprintf('Generated a Random Forest model with %d trees.\n',numTrees)

%% Step 3: Load Testing Data
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

disp('Loaded test images, histeq performed if turned on.')

%% Step 4: Random Forest Testing on Test Data with Confidence Calculation
predicted_labels = zeros(size(labels_test));
confidences = zeros(size(labels_test));

disp('Now testing on the random forest model using the test data...')

for i = 1:size(images_test, 1)
    testImage = images_test(i, :);
    [predicted_label_cell, scores] = predict(modelRF, testImage);
    predicted_labels(i) = str2double(predicted_label_cell); % Convert cell to numeric
    confidences(i) = max(scores); % Confidence is max probability of prediction
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

%% Multi-scale Sliding Window Detection on Larger Image
% This portion of the code takes an EXTREMELY long time depending on num of
% trees (About 10 minutes for 100 trees.
% I've hence saved the workspace to a .mat and you can load it instead of redoing it.)
% Refer to section right after this one for save/load.

fprintf("Now using the model to detect in larger images using multi-scale sliding window...\n Will take about %d minutes because of %d trees.\n",numTrees/10, numTrees);

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
            [label, scores] = predict(modelRF, patchVector);
            label = str2double(label);
            confidence = max(scores);
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

    % Repeat above for equalized image
    totalStepsEqualized = ceil((size(equalizedImage, 1) - currentHeight + 1) / stepSizeY) * ceil((size(equalizedImage, 2) - currentWidth + 1) / stepSizeX);
    currentStepEqualized = 0;
    progressCheckpointEqualized = 0.25;
    fprintf('Processing scale [%dx%d] on equalized image: [', currentWidth, currentHeight);
    for y = 1:stepSizeY:(size(equalizedImage, 1) - currentHeight + 1)
        for x = 1:stepSizeX:(size(equalizedImage, 2) - currentWidth + 1)
            patch = equalizedImage(y:(y + currentHeight - 1), x:(x + currentWidth - 1));
            resizedPatch = imresize(patch, [windowHeight, windowWidth]);
            patchVector = double(reshape(resizedPatch, 1, []));
            [label, scores] = predict(modelRF, patchVector);
            label = str2double(label);
            confidence = max(scores);
            if label == 1
                detectionsEqualized = [detectionsEqualized; x, y, currentWidth, currentHeight, confidence];
            end
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

% Apply Non-Maxima Suppression
threshold = 0.05;
finalDetectionsOriginal = simpleNMS(detectionsOriginal, threshold);
finalDetectionsEqualized = simpleNMS(detectionsEqualized, threshold);

%% Saving whole workspace at this point
saveNow = false; % Set to true only if you want to save workspace. This is so these save sections don't run per part of a normal big run (and re-save a broken model for e.g.)
if saveNow
    save(sprintf('FullImage_RF_%dTrees.mat', numTrees));
end
%% Load workspace to skip all the work above
% For selective loading and saving, specify variable names like:
% save('filename.mat', 'var1', 'var2') and load('filename.mat', 'var1', 'var2').
loadNow = false; % Set to true only if you want to load workspace.
if loadNow
    clear
    load('FullImage_RF_200Trees.mat')
end
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


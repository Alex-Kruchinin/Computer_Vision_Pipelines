clear 
close all
%% Full Image -> SVM -> Fixed Sliding Window Detection
%% OPTION: USE HISTOGRAM EQUALIZATION TO PREPROCESS?
applyHistogramEqualization = true; % Set to true to apply histogram equalization, false to skip it
%In my testing, INCREASES accuracy from 70.83 to 72.5% if turned ON.

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

%% Step 2: Train SVM Model
modelSVM = SVMTraining(images, labels);

disp('SVM model created.')

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

disp('Loaded test image, histeq performed if turned on.')

%% Step 4: SVM Testing on Test Data

disp('Now testing on the SVM model using test data.')

predicted_labels = zeros(size(labels_test));
confidences = zeros(size(labels_test));

for i = 1:size(images_test, 1)
    testImage = images_test(i, :);
    [predicted_labels(i), confidences(i)] = SVMTesting(testImage, modelSVM);
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

% Display correctly and incorrectly classified images
figure('Name', 'Up to 25 Samples of Correctly Classified Images');
title('Correctly Classified Images')
count = 0; i = 1;
while (count < 25) && (i <= length(comparison))
    if comparison(i)
        count = count + 1;
        Im = reshape(images_test(i, :), 27, 18);
        subplot(5, 5, count)
        imshow(Im, []);
    end
    i = i + 1;
end

% Pause to avoid matlab bug with overlapping figure info
pause(1);

figure('Name', 'Up to 25 Samples of Incorrectly Classified Images');
title('Incorrectly Classified Images')
count = 0; i = 1;
while (count < 25) && (i <= length(comparison))
    if ~comparison(i)
        count = count + 1;
        Im = reshape(images_test(i, :), 27, 18);
        subplot(5, 5, count)
        imshow(Im, []);
        title(['Pred: ' num2str(predicted_labels(i)) ', True: ' num2str(labels_test(i))])
    end
    i = i + 1;
end

% Pause to avoid matlab bug with overlapping figure info
pause(3);

%% Sliding Window Detection on Larger Image

fprintf("Now using the model to detect in larger images using fixed-scale sliding window.\n")

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
stepSizeY = 9;     % Vertical step size for the sliding window
stepSizeX = 6;     % Horizontal step size for the sliding window

% Detection on Original Image

% Initialize an array to store detection results for the original image
detectionsOriginal = [];

% Sliding window detection loop on the original image
for y = 1:stepSizeY:(size(largeImage, 1) - windowHeight + 1)
    for x = 1:stepSizeX:(size(largeImage, 2) - windowWidth + 1)
        % Extract the window patch
        patch = largeImage(y:(y + windowHeight - 1), x:(x + windowWidth - 1));
        
        % Reshape to 1D vector and classify with SVM
        patchVector = double(reshape(patch, 1, []));
        [label, confidence] = SVMTesting(patchVector, modelSVM);
        
        % If a face (or positive detection) is found, store it
        if label == 1
            detectionsOriginal = [detectionsOriginal; x, y, windowWidth, windowHeight, confidence];
        end
    end
end

% Apply Non-Maxima Suppression (NMS) on detections for the original image
threshold = 0.1;
finalDetectionsOriginal = simpleNMS(detectionsOriginal, threshold);

% Detection on Histogram Equalized Image

% Initialize an array to store detection results for the equalized image
detectionsEqualized = [];

% Sliding window detection loop on the histogram equalized image
for y = 1:stepSizeY:(size(equalizedImage, 1) - windowHeight + 1)
    for x = 1:stepSizeX:(size(equalizedImage, 2) - windowWidth + 1)
        % Extract the window patch
        patch = equalizedImage(y:(y + windowHeight - 1), x:(x + windowWidth - 1));
        
        % Reshape to 1D vector and classify with SVM
        patchVector = double(reshape(patch, 1, []));
        [label, confidence] = SVMTesting(patchVector, modelSVM);
        
        % If a face (or positive detection) is found, store it
        if label == 1
            detectionsEqualized = [detectionsEqualized; x, y, windowWidth, windowHeight, confidence];
        end
    end
end

% Apply Non-Maxima Suppression (NMS) on detections for the equalized image
finalDetectionsEqualized = simpleNMS(detectionsEqualized, 0.1);

% Display Results Before NMS in a 1x2 Grid

figure;
subplot(1, 2, 1);
ShowDetectionResult(largeImage, detectionsOriginal);
title('Original Image - Faces highlighted before NMS - Confidence goes up by colors white-yellow-green-cyan');

subplot(1, 2, 2);
ShowDetectionResult(equalizedImage, detectionsEqualized);
title('Equalized Image - Faces highlighted before NMS - Confidence goes up by colors white-yellow-green-cyan');

pause(1); % Pause to avoid MATLAB display issues

% Display Results After NMS in a 1x2 Grid

figure;
subplot(1, 2, 1);
ShowDetectionResult(largeImage, finalDetectionsOriginal);
title('Original Image - Faces highlighted after NMS - Confidence goes up by colors white-yellow-green-cyan');

subplot(1, 2, 2);
ShowDetectionResult(equalizedImage, finalDetectionsEqualized);
title('Equalized Image - Faces highlighted after NMS - Confidence goes up by colors white-yellow-green-cyan');


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

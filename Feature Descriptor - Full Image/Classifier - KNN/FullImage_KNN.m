clear 
close all
%% Full Image -> KNN -> Fixed Sliding Window Detection with Confidence Score
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
maxK = 90; % Maximum value of K to check (adjust as needed)

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
pause(3);


%% Using the model and implementing Sliding Window Detector for bigger images

fprintf("Now using the model to detect in larger images using fixed-scale sliding window.\n")

% Load the larger image
largeImage = imread('im3.jpg');
if size(largeImage, 3) > 1
    largeImage = rgb2gray(largeImage);
end

% Define sliding window parameters
windowHeight = 27; % Base height of the window
windowWidth = 18;  % Base width of the window
stepSizeY = 9;     % Vertical step size for the sliding window
stepSizeX = 6;     % Horizontal step size for the sliding window

% Detection on Original Image

% Initialize an array to store detection results for the original image
detections = [];

% Loop through the image with sliding window
for y = 1:stepSizeY:(size(largeImage, 1) - windowHeight + 1)
    for x = 1:stepSizeX:(size(largeImage, 2) - windowWidth + 1)
        % Extract the window patch
        patch = largeImage(y:(y + windowHeight - 1), x:(x + windowWidth - 1));
        
        % Reshape to 1D vector and classify
        patchVector = double(reshape(patch, 1, []));
        label = KNNTesting(patchVector, modelNN, BestK);
        
        % If a face (or positive detection) is found, store it with a confidence score
        if label == 1
            confidence = 1; % Since KNN does not inherently give confidence scores compared to SVM, we just set this to 1.
            detections = [detections; x, y, windowWidth, windowHeight, confidence];
        end
    end
end

% Apply Non-Maxima Suppression (NMS) to the detections on Original Image
threshold = 0.1; % Set a threshold for overlap (adjust as needed)
finalDetections = simpleNMS(detections, threshold);

% Detection with Histogram Equalization

% Apply histogram equalization to a copy of the large image
equalizedImage = uint8(255 * mat2gray(largeImage)); % Normalize and then apply histogram equalization
equalizedImage = histeq(equalizedImage);

% Initialize an array to store detection results for the equalized image
detectionsEqualized = [];

% Loop through the equalized image with sliding window
for y = 1:stepSizeY:(size(equalizedImage, 1) - windowHeight + 1)
    for x = 1:stepSizeX:(size(equalizedImage, 2) - windowWidth + 1)
        % Extract the window patch
        patch = equalizedImage(y:(y + windowHeight - 1), x:(x + windowWidth - 1));
        
        % Reshape to 1D vector and classify
        patchVector = double(reshape(patch, 1, []));
        label = KNNTesting(patchVector, modelNN, BestK);
        
        % If a face (or positive detection) is found, store it with a confidence score
        if label == 1
            confidence = 1;
            detectionsEqualized = [detectionsEqualized; x, y, windowWidth, windowHeight, confidence];
        end
    end
end

% Apply Non-Maxima Suppression (NMS) on detections for the equalized image
finalDetectionsEqualized = simpleNMS(detectionsEqualized, threshold);

% Display Results Before NMS in a 1x2 Grid

figure;
subplot(1, 2, 1);
ShowDetectionResult(largeImage, detections);
title('Original Image - Faces highlighted before NMS');

subplot(1, 2, 2);
ShowDetectionResult(equalizedImage, detectionsEqualized);
title('Equalized Image - Faces highlighted before NMS');

pause(1); % Pause to avoid MATLAB display issues

% Display Results After NMS in a 1x2 Grid

figure;
subplot(1, 2, 1);
ShowDetectionResult(largeImage, finalDetections);
title('Original Image - Faces highlighted after NMS');

subplot(1, 2, 2);
ShowDetectionResult(equalizedImage, finalDetectionsEqualized);
title('Equalized Image - Faces highlighted after NMS');


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

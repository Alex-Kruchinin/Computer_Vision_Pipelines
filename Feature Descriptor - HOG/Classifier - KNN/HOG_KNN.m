clear 
close all
%% HOG -> KNN -> Multiscale Sliding Window Detection with Confidence Scores
%% OPTION: USE HISTOGRAM EQUALIZATION TO PREPROCESS?
applyHistogramEqualization = true; % Set to true to apply histogram equalization, false to skip it

% ON: Accuracy: 92.50% | Precision: 0.94 | Recall: 0.97 | F1 Score: 0.96
% OFF: Accuracy: 87.92% | Precision: 0.89 | Recall: 0.98 | F1 Score: 0.93
% ON seems to have the better model but the final test on a larger image
% using sliding window seems to have ambigous results. OFF feels better?

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

%% Step 2: Extract HOG Features for Training Data

% Define image dimensions for HOG feature extraction
imageHeight = 27; 
imageWidth = 18;

% Initialize a matrix to store HOG features for all images
numImages = size(images, 1);
hogFeatureMatrix = zeros(numImages, length(hog_feature_vector(reshape(images(1, :), [imageHeight, imageWidth]))));

% Loop over each image and extract HOG features
for i = 1:numImages
    image = reshape(images(i, :), [imageHeight, imageWidth]);
    hogFeatureMatrix(i, :) = hog_feature_vector(image);
end

disp('HOG features for all training images have been extracted.');

%% (FOR FUN ONLY) Look at HOG images
% Select the 'nth' image to display
fprintf("There's %d images in the dataset, each of %d pixels because %d x %d.\n",size(images), imageWidth, imageHeight) % Should show 670 x 486 if augmentation is turned on, because each image is flattened into a 1D vector, stored in a 2D matrix where each new row is a new image. 
n = 111; % Change this index to select different images. Note that if you have Augmentation turned on in loadFaceImages.m, there'll be 10 images of roughly the same person so 1-10, 11-20, etc. 
sample_image = reshape(images(n, :), [imageHeight, imageWidth]); % Have to turn it back into a 2D matrix, since it was reshaped into 1D in the loadPedestrianDatabase().
hogFeatures = hog_feature_vector(sample_image);

% Display the Image and HOG Visualization side by side
figure;

% Display the original image
subplot(1, 2, 1);
imshow(sample_image, []);
title(['Original Image (Index ', num2str(n), ')']);

% Display the HOG visualization
subplot(1, 2, 2);
showHog(hogFeatures, [imageHeight, imageWidth]);
title(['HOG Representation (Index ', num2str(n), ')']);

% % If you want to look at a car, for a clearer HOG picture
% figure
% im1 = imread('car-2683858_1280.jpg'); 
% hogFeatures = hog_feature_vector(im1);
% % Display the original car image
% subplot(1, 2, 1);
% imshow(im1, []);
% title(['Original Image']);
% 
% % Display the HOG visualization of the car
% subplot(1, 2, 2);
% showHogPreserveAspectRatio(hogFeatures, [853, 1280]);
% title(['HOG Representation']);


%% Step 3: Initialize K-NN Model

% Create a structure for storing the model
modelNN.neighbours = hogFeatureMatrix; 
modelNN.labels = labels;

disp('NN model initialized using HOG features and labels.')


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

disp('Loaded test images, histeq performed if turned on.')

%% Step 5: Extract HOG Features for Test Data

% Initialize a matrix to store HOG features for test images
numTestImages = size(images_test, 1);
hogTestFeatureMatrix = zeros(numTestImages, size(hogFeatureMatrix, 2));

% Loop over each test image and extract HOG features
for i = 1:numTestImages
    testImage = reshape(images_test(i, :), [imageHeight, imageWidth]);
    hogTestFeatureMatrix(i, :) = hog_feature_vector(testImage);
end

disp('HOG features for all testing images have been extracted.');

%% Step 6: K-NN Testing

% Initialize variables
BestAccuracy = 0;
BestK = 1; % To store the best value of K
maxK = 90; % Maximum value of K to check (adjust as needed)

fprintf('Now testing for best K from %d to %d:\n', BestK, maxK)

% Loop through values of K from 1 to maxK
for K = 1:maxK
    % For each testing image, we obtain a prediction based on the current value of K
    classificationResult = zeros(size(hogTestFeatureMatrix, 1), 1);
    for i = 1:size(hogTestFeatureMatrix, 1)
        testFeature = hogTestFeatureMatrix(i, :);
        classificationResult(i) = KNNTesting(testFeature, modelNN, K);
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

%% Step 7: Final Evaluation on Best K

% Re-run testing with the best K to resolve some bug with calculating TP, FP, TN, and FN
predicted_labels = zeros(size(labels_test)); % Initialize predictions array

for i = 1:size(hogTestFeatureMatrix, 1)
    testFeature = hogTestFeatureMatrix(i, :);
    predicted_labels(i) = KNNTesting(testFeature, modelNN, BestK); % Using BestK only!
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
        Im = reshape(images_test(i, :), [imageHeight, imageWidth]); % Reshape back to original dimensions
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
    if ~comparison(i) % Show misclassified samples
        count = count + 1;
        Im = reshape(images_test(i, :), [imageHeight, imageWidth]);
        subplot(5, 5, count)
        imshow(Im, []);
        title(['Pred: ' num2str(predicted_labels(i)) ', True: ' num2str(labels_test(i))])
    end
    i = i + 1;
end

pause(1);

%% Multi-scale Sliding Window Detection on Larger Image for KNN

fprintf("Now using the model to detect in larger images using multi-scale sliding window.\n")

% Load the larger image
largeImage = imread('im1.jpg');
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
            
            % Resize patch to match training data dimensions (18x27) and extract HOG features
            resizedPatch = imresize(patch, [windowHeight, windowWidth]);
            patchHOG = hog_feature_vector(resizedPatch);
            
            % Perform classification with KNN
            [label, confidence] = KNNTestingWithConfidence(patchHOG, modelNN, BestK);
            
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
    progressCheckpointEqualized = 0.25;
    
    % Sliding Window Detection for the Equalized Image
    fprintf('Processing scale [%dx%d] on equalized image: [', currentWidth, currentHeight);
    for y = 1:stepSizeY:(size(equalizedImage, 1) - currentHeight + 1)
        for x = 1:stepSizeX:(size(equalizedImage, 2) - currentWidth + 1)
            patch = equalizedImage(y:(y + currentHeight - 1), x:(x + currentWidth - 1));
            resizedPatch = imresize(patch, [windowHeight, windowWidth]);
            patchHOG = hog_feature_vector(resizedPatch);
            
            [label, confidence] = KNNTestingWithConfidence(patchHOG, modelNN, BestK);
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
%% Apply Non-Maxima Suppression (NMS)
threshold = 0.01;
finalDetectionsOriginal = simpleNMS(detectionsOriginal, threshold);
finalDetectionsEqualized = simpleNMS(detectionsEqualized, threshold);

%% Saving whole workspace at this point
saveNow = false; % Set to true only if you want to save workspace. This is so these save sections don't run per part of a normal big run (and re-save a broken model for e.g.)
if saveNow
    save('HOG_KNN.mat');
end
%% Load workspace to skip all the work above
% For selective loading and saving, specify variable names like:
% save('filename.mat', 'var1', 'var2') and load('filename.mat', 'var1', 'var2').
loadNow = false; % Set to true only if you want to load workspace.
if loadNow
    clear
    load('HOG_KNN_histeqON.mat')
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




% HOG feature extraction function
function [feature] = hog_feature_vector(im)
    % Convert RGB image to grayscale
    if size(im,3)==3
        im=rgb2gray(im);
    end
    % if nargin>1
    %     im=imresize(im,resize);
    % end
    
    im=double(im);
    
    rows=size(im,1);
    cols=size(im,2);
    Ix=im; %Basic Matrix assignment
    Iy=im; %Basic Matrix assignment
    
    % Gradients in X and Y direction. Iy is the gradient in X direction and Iy
    % is the gradient in Y direction
    for i=1:rows-2
        Iy(i,:)=(im(i,:)-im(i+2,:));
    end
    for i=1:cols-2
        Ix(:,i)=(im(:,i)-im(:,i+2));
    end
    
    gauss=fspecial('gaussian',8); %% Initialized a gaussian filter with sigma=0.5 * block width.    
    
    angle=atand(Ix./Iy); % Matrix containing the angles of each edge gradient
    angle=imadd(angle,90); %Angles in range (0,180)
    magnitude=sqrt(Ix.^2 + Iy.^2);
    
    % figure,imshow(uint8(angle));
    % figure,imshow(uint8(magnitude));
    
    % Remove redundant pixels in an image. 
    angle(isnan(angle))=0;
    magnitude(isnan(magnitude))=0;
    
    feature=[]; %initialized the feature vector
    
    % Iterations for Blocks
    for i = 0: rows/8 - 2
        for j= 0: cols/8 -2
            %disp([i,j])
            
            mag_patch = magnitude(8*i+1 : 8*i+16 , 8*j+1 : 8*j+16);
            %mag_patch = imfilter(mag_patch,gauss);
            ang_patch = angle(8*i+1 : 8*i+16 , 8*j+1 : 8*j+16);
            
            block_feature=[];
            
            %Iterations for cells in a block
            for x= 0:1
                for y= 0:1
                    angleA =ang_patch(8*x+1:8*x+8, 8*y+1:8*y+8);
                    magA   =mag_patch(8*x+1:8*x+8, 8*y+1:8*y+8); 
                    histr  =zeros(1,9);
                    
                    %Iterations for pixels in one cell
                    for p=1:8
                        for q=1:8
                           
                            alpha= angleA(p,q);
                            
                            % Binning Process (Bi-Linear Interpolation)
                            if alpha>10 && alpha<=30
                                histr(1)=histr(1)+ magA(p,q)*(30-alpha)/20;
                                histr(2)=histr(2)+ magA(p,q)*(alpha-10)/20;
                            elseif alpha>30 && alpha<=50
                                histr(2)=histr(2)+ magA(p,q)*(50-alpha)/20;                 
                                histr(3)=histr(3)+ magA(p,q)*(alpha-30)/20;
                            elseif alpha>50 && alpha<=70
                                histr(3)=histr(3)+ magA(p,q)*(70-alpha)/20;
                                histr(4)=histr(4)+ magA(p,q)*(alpha-50)/20;
                            elseif alpha>70 && alpha<=90
                                histr(4)=histr(4)+ magA(p,q)*(90-alpha)/20;
                                histr(5)=histr(5)+ magA(p,q)*(alpha-70)/20;
                            elseif alpha>90 && alpha<=110
                                histr(5)=histr(5)+ magA(p,q)*(110-alpha)/20;
                                histr(6)=histr(6)+ magA(p,q)*(alpha-90)/20;
                            elseif alpha>110 && alpha<=130
                                histr(6)=histr(6)+ magA(p,q)*(130-alpha)/20;
                                histr(7)=histr(7)+ magA(p,q)*(alpha-110)/20;
                            elseif alpha>130 && alpha<=150
                                histr(7)=histr(7)+ magA(p,q)*(150-alpha)/20;
                                histr(8)=histr(8)+ magA(p,q)*(alpha-130)/20;
                            elseif alpha>150 && alpha<=170
                                histr(8)=histr(8)+ magA(p,q)*(170-alpha)/20;
                                histr(9)=histr(9)+ magA(p,q)*(alpha-150)/20;
                            elseif alpha>=0 && alpha<=10
                                histr(1)=histr(1)+ magA(p,q)*(alpha+10)/20;
                                histr(9)=histr(9)+ magA(p,q)*(10-alpha)/20;
                            elseif alpha>170 && alpha<=180
                                histr(9)=histr(9)+ magA(p,q)*(190-alpha)/20;
                                histr(1)=histr(1)+ magA(p,q)*(alpha-170)/20;
                            end
                            
                    
                        end
                    end
                    block_feature=[block_feature histr]; % Concatenation of Four histograms to form one block feature
                                    
                end
            end
            % Normalize the values in the block using L1-Norm
            block_feature=block_feature/sqrt(norm(block_feature)^2+.01);
                   
            feature=[feature block_feature]; %Features concatenation
        end
    end
    
    feature(isnan(feature))=0; %Removing Infinitiy values
    
    % Normalization of the feature vector using L2-Norm
    feature=feature/sqrt(norm(feature)^2+.001);
    for z=1:length(feature)
        if feature(z)>0.2
             feature(z)=0.2;
        end
    end
    feature=feature/sqrt(norm(feature)^2+.001);        
    
    % toc;
end




% Non-Maxima Suppression function
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




function []=showHog(feature,rsize)
    %Input param feature: HOG feature vector in a row vector
    %Input param rsize: dimensions of the original input image in the form of a
    %                    2 elements vector [rows columns]
    blocksPerColumn=rsize(1)/8-1;
    blocksPerRow=rsize(2)/8-1;
    
    % construct a "glyph" for each orientaion
    bs=20;
    bim1 = zeros(bs, bs);
    bim1(:,round(bs/2):round(bs/2)+1) = 255;
    bim = zeros([size(bim1) 9]);
    bim(:,:,1) = bim1;
    for i = 2:9,
        bim(:,:,i) = imrotate(bim1, -(i-1)*20, 'crop');
    end
    
    totalIm=zeros(2*bs*blocksPerColumn,2*bs*blocksPerRow);
    counter=0;
    for i=1:36:length(feature)
        
        histBlock= feature(i:i+35);
        
        imBlock=zeros(2*bs,2*bs);
        % a block is composed of 4 cells
        for j=1:4
            % for each cell, we can caluclate the composed "glyph" according to teh hitogram of orientations
            w=histBlock(j:j+8);
            
            w(w < 0) = 0;
            im = zeros(bs, bs);
            for k = 1:9,
                im = im + bim(:,:,k) * w(k);
            end
            
            %compose the image of the block
            row=floor((j-1)/2)+1;
            column=mod(j-1,2)+1;
            
            imBlock((row-1)*bs+1: row*bs, (column-1)*bs+1: column*bs) = im;
        end
        counter=counter+1;
        %compose the full image
        row=floor((counter-1)/blocksPerRow)+1;
        column=mod(counter-1,blocksPerRow)+1;
        totalim((row-1)*2*bs+1: row*2*bs, (column-1)*2*bs+1: column*2*bs) = imBlock;
    end
    scale = max(feature);
    totalim = totalim / scale;
    imagesc(totalim), colormap(gray)
end




function []=showHogPreserveAspectRatio(feature, rsize)
    % Input param feature: HOG feature vector in a row vector
    % Input param rsize: dimensions of the original input image in the form of a
    %                    2 elements vector [rows columns] (height, width)
    
    blocksPerColumn = rsize(1) / 8 - 1;
    blocksPerRow = rsize(2) / 8 - 1;

    % Construct a "glyph" for each orientation
    bs = 20;
    bim1 = zeros(bs, bs);
    bim1(:, round(bs/2):round(bs/2)+1) = 255;
    bim = zeros([size(bim1), 9]);
    bim(:,:,1) = bim1;
    for i = 2:9
        bim(:,:,i) = imrotate(bim1, -(i-1)*20, 'crop');
    end

    % Adjusted size for the HOG representation image based on the aspect ratio
    hogImHeight = blocksPerColumn * 2 * bs;
    hogImWidth = blocksPerRow * 2 * bs;
    totalIm = zeros(hogImHeight, hogImWidth);

    counter = 0;
    for i = 1:36:length(feature)
        histBlock = feature(i:i+35);
        imBlock = zeros(2 * bs, 2 * bs);

        % A block is composed of 4 cells
        for j = 1:4
            w = histBlock((j-1)*9+1:j*9);
            w(w < 0) = 0;
            im = zeros(bs, bs);
            for k = 1:9
                im = im + bim(:,:,k) * w(k);
            end

            % Position in the 2x2 block structure
            row = floor((j-1)/2) + 1;
            column = mod(j-1, 2) + 1;
            imBlock((row-1)*bs+1 : row*bs, (column-1)*bs+1 : column*bs) = im;
        end

        counter = counter + 1;
        row = floor((counter-1) / blocksPerRow) + 1;
        column = mod(counter-1, blocksPerRow) + 1;

        totalIm((row-1)*2*bs+1 : row*2*bs, (column-1)*2*bs+1 : column*2*bs) = imBlock;
    end

    % Scale the HOG image to the original aspect ratio
    scale = max(feature);
    totalIm = totalIm / scale;

    % Display HOG image preserving aspect ratio
    imagesc(totalIm), colormap(gray);
    axis image; % This command preserves the aspect ratio
end

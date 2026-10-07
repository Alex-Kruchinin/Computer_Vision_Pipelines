clear
close all

%% HOG -> KNN with Leave-One-Out Cross Validation
%% OPTION: USE HISTOGRAM EQUALIZATION TO PREPROCESS?
applyHistogramEqualization = true; % Set to true to apply histogram equalization, false to skip it

%% Step 1: Load All Data (Combine face and non-face images)
[images, labels] = loadFaceImages('face_all.cdataset'); % Load all samples for LOOCV

% Perform histogram equalization if enabled
if applyHistogramEqualization
    for i = 1:size(images, 1)
        originalImage = reshape(images(i, :), [27, 18]);
        originalImage = uint8(255 * mat2gray(originalImage)); % Normalize for equalization
        equalizedImage = histeq(originalImage);
        images(i, :) = double(reshape(equalizedImage, 1, [])); % Flatten and store back
    end
end

disp('Loaded all images, histeq performed if turned on.')

%% Step 2: Extract HOG Features for All Data

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

disp('HOG features for all images have been extracted.')

%% Step 3: LOOCV for K-NN with Tuning K

maxK = 100; % Maximum value of K to check (adjust as needed)
totalSamples = size(hogFeatureMatrix, 1);
accuracyPerK = zeros(maxK, 1); % To store average accuracy for each K

fprintf('Running LOOCV for K values from 1 to %d:\n', maxK);

% For each K within the selected range
for K = 1:maxK
    correctPredictions = 0; % Increment the number of correct predictions for this K if it predicts correctly
    
    % Leave-One-Out Cross Validation
    for testIndex = 1:totalSamples
        % Leave the current sample out as the test instance
        testFeature = hogFeatureMatrix(testIndex, :);
        testLabel = labels(testIndex);
        
        % Use remaining samples as the training set
        trainFeatures = hogFeatureMatrix([1:testIndex-1, testIndex+1:end], :);
        trainLabels = labels([1:testIndex-1, testIndex+1:end]);
        
        % Initialize KNN model with current training set
        modelNN.neighbours = trainFeatures;
        modelNN.labels = trainLabels;
        
        % Perform KNN classification on the test instance with current K
        predictedLabel = KNNTesting(testFeature, modelNN, K);
        
        % Check if the prediction is correct
        if predictedLabel == testLabel
            correctPredictions = correctPredictions + 1;
        end
    end
    
    % Calculate and store average accuracy for this K
    accuracyPerK(K) = correctPredictions / totalSamples;
    fprintf('K = %d, Average LOOCV Accuracy = %.2f%%\n', K, accuracyPerK(K) * 100);
end

%% Step 4: Determine the Best K

[bestAccuracy, bestK] = max(accuracyPerK);
fprintf('\nBest K = %d with Average LOOCV Accuracy = %.2f%%\n', bestK, bestAccuracy * 100);

%% Functions
% HOG Feature Extraction Function
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


% EuclideanDistance Function
function dEuc = EuclideanDistance(sample1, sample2)
    difference = sample1 - sample2;
    squaredDifference = difference .^ 2;
    sumOfSquares = sum(squaredDifference);
    dEuc = sqrt(sumOfSquares);
end

% KNNTesting Function
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

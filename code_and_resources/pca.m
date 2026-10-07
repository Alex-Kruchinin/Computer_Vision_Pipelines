% This function implements pca and 
% returns  U:eigenvectors,S:eigenvalues & X_reduce: dataset with n dimensions
% here X:dataset with each instance as a row , n: reduced dimesions size
function [ U,S,X_reduce ] = pca(X,n)
    % 1. Input Parameters:
    % X: The dataset where each row is a data instance (for example, each MNIST image vector). 
    %    It is assumed to have a size of m x d, where m is the number of data instances (samples),
    %    and d is the number of features (dimensions), which is 784 for MNIST (28x28 images).
    % n: The number of reduced dimensions. If this parameter is not provided, the function will default to 50 dimensions unless the original dataset has fewer than 50 dimensions.
    
    if nargin < 2 AND size(X,2)> 50
        n = 50;
    elseif size(X,2)<50
        fprintf('very few dimensions.. maybe you dont need pca at all')
    end
    
    % 2. Covariance Matrix Calculation:
    m = size(X,1); % m is the number of data samples (rows in X).
    sigma = (1/m)*(X'*X); %The covariance matrix captures the relationships (correlations) between the
                          % different features (pixels in the case of MNIST). It helps you understand 
                          % how much the features vary with each other. The goal of PCA is to find the
                          % principal directions (or components) where the data varies the most, and 
                          % project the data onto those directions to reduce dimensionality.
    
    % 3. Singular Value Decomposition (SVD):
    [U S] = svd(sigma); %Here, SVD (Singular Value Decomposition) is applied to the covariance matrix sigma. SVD decomposes the covariance matrix into three matrices:
                        %   U: A matrix of eigenvectors (the principal components).
                        %   S: A diagonal matrix of eigenvalues (which represent the variance along each principal component).
                        %The eigenvectors (columns of U) correspond to the principal components of the data. These are the directions in which the data varies the most.
                        %The eigenvalues in S tell you how much variance (information) is captured by each principal component.
    
    X_reduce = zeros(size(X, 1), n); 
    
    % 4. Selecting Top n Principal Components:
    U_reduce = U(:,1:n);      %The function selects the first n eigenvectors (principal components) from U. 
                              %These are the top n directions where the data varies the most.
                              % If n = 50, the top 50 principal components are selected.
    
    % 5. Projecting the Data onto the Reduced Subspace:
    for i=1:m
        for j=1:n
            x= X(i,:)';            
            X_reduce(i,j) = x'*U_reduce(:,j);
        end
    end


end


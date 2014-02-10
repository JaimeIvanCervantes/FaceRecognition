
%AUTHOR: Jaime Ivan Cervantes
%LAST UPDATED: December 13, 2011
%DESCRIPTION: This code implements PCA, Fisher LDA, and SVM to find matching faces.

% Clear memory and console
close all
clear
clc

% Choose simple, pose variation, or illumination variation dataset 
dataset = input('Please select your dataset: \n(1) simple.m \n(2) pose.m\n');

if dataset == 1
     load '../data/data.mat'
     %Define variables
     k = 200;% Number of classes
     n = 3;% Number of images per class 
elseif dataset == 2
     load '../pose/pose.mat'
     face = pose;
    %Define variables
     k = 68;% Number of classes
     n = 13;% Number of images per class
else
    display('Input not valid');
end

testNum = input('Enter the number of the face for which you want to find the match: ');
classType = input('Choose (1) for SVM or (2) for Euclidean Distance: ');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Principal Component Analysis (PCA)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

display('Loading please wait...')

% Read images in T matrix
[nRow nCol M] = size(face);

% T is a matrix containing the reshaped vectors for each image
T = reshape(face,[nRow*nCol M]);

% mTot is the mean of the entire set of training images
mTot = mean(T,2);

% substract mean
A = T-repmat(mTot,1,M);

% Obtaining eigenvalues and eigenvectors of A'A
[V,D] = eig(A'*A);

% Obtaning more relevant eigenvalues and eigenvectors
eval = diag(D);

peval = [];
pevec = [];

for i = M:-1:k+1
    peval = [peval eval(i)];
    pevec = [pevec V(:,i)];
end

% Obtaining the eigenvectors
U = A * pevec; 

% Obtaining PCA weights
 Wpca = U'*A;
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fisher's Linear Discriminant Analysis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
 
% Obtaining Sb and Sw
cMean = zeros(M-k,M-k);
Sb = zeros(M-k,M-k);
Sw = zeros(M-k,M-k);

pcaMean = mean(Wpca,2);

for i = 1:k
    cMean = mean(Wpca(:,n*i-(n-1):n*i),2);
    Sb = Sb + (cMean-pcaMean)*(cMean-pcaMean)';
end

Sb = n*Sb;

for i = 1:k
    cMean = mean(Wpca(:,n*i-(n-1):n*i),2);
    for j = n*i-(n-1):n*i
         Sw = Sw + (Wpca(:,j)-cMean)*(Wpca(:,j)-cMean)';
    end
end

% Obtaining Fisher eigenvectors and eigenvalues
[Vf, Df] = eig(Sb,Sw);

% Calculating weights
 Df = fliplr(diag(Df));
 Vf = fliplr(Vf);

% Calculating fisher weights
Wf = Vf'*Wpca;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Support Vector Machine (LIBSCM Dependency)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if classType == 1
    
    % testNum = 13*20;

    % Reshape the selected face
    Tr = reshape(face(:,:,testNum),[nRow*nCol 1]);
    Ar = Tr-mTot;

    % Obtain the weights of the normalized selected face
    Wrec = Vf'*U'*Ar;

    % SVM parameters. The kernel is a polynomial
    c=1e9;
    params=[' -t ' int2str(1) ' -c ' int2str(c)];

    % prevArray starts with an array containing each class. Winner array is the
    %classes that are selected in the binary tree
    prevArr = [1:k]; 
    winnerArr = [];

    % This section of the code computes a binary SVM tree, by solving a 2 class
    % problem with SVM for every 2 classes (1 and 2, 3 and 4, 5 and 6 etc). The
    % that you selected is classified according to each 2-class problem and the
    % class selected goes on to compete with the other classes selected.
    for max = 1:1000
        winnerArr = []; 
        for winRep = 1:2:length(prevArr) 

            % Selects the two classes to train the SVM
            if winRep >= length(prevArr)
                i = prevArr(winRep) ;       
                j = prevArr(winRep-1) ;   
            else
                i = prevArr(winRep)  ;      
                j = prevArr(winRep+1)  ; 
            end

            % Selects the features of the 2 classes
            feature = [Wf(1:2,n*i-(n-1):n*i),Wf(1:2,n*j-(n-1):n*j)]';

            % Assigns the labels for each class
            for m1 = 1:n
               label(m1) = 1; 
            end

            for n1 = n+1:2*n
               label(n1) = -1; 
            end

            % The SVM is trained
            model=svmtrain(label', feature, params);

            % The face that the user selected is classified to any of the two
            % classes
            guessLab(1) = 1;
            predLabel=svmpredict(guessLab',[Wrec(1) Wrec(2)],model);

            predLabel;

            % A winner class is selected
            if predLabel == 1 
                winnerArr = [winnerArr i];
            elseif predLabel == -1 
                winnerArr = [winnerArr j];
            end

            if winnerArr > 1
                for c1 = 2:length(winnerArr)
                    if winnerArr(c1) == winnerArr(c1-1)
                       winnerArr(c1) = []; 
                    end
                end
            end

        end

        prevArr = winnerArr;

       if length(winnerArr) < 2
           winnerArr;
            break
        end
    end

    % This is the class that was selected
    CLASS = winnerArr;

    display('The class that matches your face is:')
    display(CLASS)

    % Plot the selected face
    figure(1)
    imagesc(reshape(Tr,nRow,nCol));
    colormap gray;
    title('Face selected')

    % Plot the faces of the matching class
    for i = 1:n
        figure(i+1)
        imagesc(reshape(T(:,winnerArr*n-i+1),nRow,nCol));
        colormap gray;
        title(strcat('Matching face number: ',num2str(winnerArr*n-i+1)));
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SELECT FACE BASED ON THE SHORTEST EUCLIDEAN DISTANCE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if classType == 2
    
    % Calculate euclidean distance
    %testNum = 134;

    % Normalize selected image
    Tr = reshape(face(:,:,testNum),[nRow*nCol 1]);
    Ar = Tr-mTot;

    % Obtain weights of the selected face
    Wrec = Vf'*U'*Ar;

    temp = 0;

    % Obtaining an array of euclidean distances to each face
    eDist = [];
    for i = 1:M
        eDist = [eDist sqrt(( norm( Wrec - Wf(:,i)) )^2)]; 
    end

    % Find minimum distance and the corresponding index
    minDis = 999999;
    minIndex = 0;

    for i = 1:length(eDist)
       if minDis > eDist(i) && i ~= testNum
           minDis = eDist(i);
           minIndex = i;
       end
    end

    Matching_index = minIndex;
    
    % Matching index
    display(Matching_index);

    % Plot selected face
    figure(1)
    imagesc(reshape(Tr,nRow,nCol));
    colormap gray;
    title('Face selected')
    
    % Plot best match
    figure(2)
    imagesc(reshape(T(:,minIndex),nRow,nCol));
    colormap gray;
    title('Best match')
end






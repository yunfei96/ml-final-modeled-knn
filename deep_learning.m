%get the input data
f1 = importdata("f1.data");
f2 = importdata("f2.data");
f3 = importdata("f3.data");
f4 = importdata("f4.data");
f5 = importdata("f5.data");
%implement 5 fold
%data = [f1;f2;f3;f4;f5];
%data = importdata("australian.dat"); %1-14, 15
data = importdata("page-blocks.data");
% load fisheriris
% X = meas;
% Y = species;
% Y = grp2idx(Y)
% data = [X, Y]
folds = 5;
interval = floor(length(data) / folds);
accuracy_list = [];
for j=1:folds
    left = (j - 1) * interval + 1;
    if j == folds
        right = length(data);
    else
        right = left + interval - 1;
    end
    test = data(left : right, :);
    train = [data(1 : left - 1, :); data(right + 1 : length(data), :)];
%set up test data
test_data = test(:,1:10);
num_test_data = length(test_data);
test_label = test(:,11);
%set  up train data
train_data = train(:,1:10);
num_train_data = length(train_data);
train_label = train(:,11);
%---------------------

XT = cell(num_train_data,1);
for i=1:num_train_data
    XT(i) = {train_data(i,:)'};
end

YT = categorical(train_label);

inputSize = 10;
numHiddenUnits = 100;
numClasses = 5;

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]

maxEpochs = 100;
miniBatchSize = 27;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(XT,YT,layers,options);
%%

XTest = cell(num_test_data,1);
for i=1:num_test_data
    XTest(i) = {test_data(i,:)'};
end
YTest = categorical(test_label);
YPred = classify(net,XTest,'MiniBatchSize',miniBatchSize);
acc = sum(YPred == YTest)./numel(YTest)
accuracy_list(length(accuracy_list) + 1) = acc;
end
disp("--------------------")
accuracy_list
disp(mean(accuracy_list))
%get the input data
f1 = importdata("f1.data");
f2 = importdata("f2.data");
f3 = importdata("f3.data");
f4 = importdata("f4.data");
f5 = importdata("f5.data");
%implement 5 fold
train = [f1;f2;f3;f4];
test = f5;
%set up test data
test_data = test(:,2:10);
num_test_data = length(test_data);
test_label = test(:,11);
%set  up train data
train_data = train(:,2:10);
num_train_data = length(train_data);
train_label = train(:,11);
%---------------------

XT = cell(171,1);
for i=1:num_train_data
    XT(i) = {train_data(i,:)'};
end

YT = categorical(train_label);

inputSize = 9;
numHiddenUnits = 100;
numClasses = 6;

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

XTest = cell(43,1);
for i=1:num_test_data
    XTest(i) = {test_data(i,:)'};
end
YTest = categorical(test_label);
YPred = classify(net,XTest,'MiniBatchSize',miniBatchSize);
acc = sum(YPred == YTest)./numel(YTest)
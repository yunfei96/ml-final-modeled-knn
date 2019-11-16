%get the input data
f1 = importdata("f1.data");
f2 = importdata("f2.data");
f3 = importdata("f3.data");
f4 = importdata("f4.data");
f5 = importdata("f5.data");
%implement 5 fold
%train = [f1;f2;f3;f4];
%test = f5;
data = [f1;f2;f3;f4;f5];
folds = 5;
interval = floor(length(data) / folds);
final_list = []
knn_accuracy = []
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
test_data = test(:,2:10);
num_test_data = length(test_data);
test_label = test(:,11);
%set  up train data
train_data = train(:,2:10);
num_train_data = length(train_data);
train_label = train(:,11);
visit = zeros(num_train_data,1);
model = [];

%loop through each data
for i = 1:num_train_data
    if visit(i) ~= 1
        point = train_data(i,:);
        cat = train_label(i);
        [number, center, range, visit] = find_largest_range(point, cat , train_data, train_label, visit);
        model = [model; [number, center, range, cat]];
    end
    
end

result_model = [];
%eliminate small point
for i = 1:length(model)
    if model(i) > 2
        result_model = [result_model; model(i,:)];
    end 
end

%prediction
predict_label = [];


for i=1:num_test_data
    %find in range and push into
    result = [];
    closest_dis = intmax;
    closest_cat  = [];
    for j = 1: size(result_model)
        center = result_model(j,2:10);
        cat = result_model(j,12);
        range = result_model(j,11);
    
        %find the cloest
        if pdist2(center, test_data(i,:))-range < closest_dis
            closest_dis = pdist2(center, test_data(i,:))-range;
            closest_cat = cat;
        end
    
        %find anything in range
        if pdist2(center, test_data(i,:)) <= range
            result = [result; result_model(j,:)];
        end
    end
    %if inrange == 0 or inrange >= 1
    [l, r] = size(result);
    if l==0
        predict_label = [predict_label;closest_cat];
    else
        max = 0;
        max_cat = 0;
        for k = 1:l
            if result(k,1) > max
                max = result(k,1);
                max_cat = result(k,12);
            end
        end
        predict_label = [predict_label; max_cat];
    end

end



cm = confusionmat(test_label,predict_label)
accuracy = sum(diag (cm))/ sum(sum(cm))


Idx = knnsearch(train_data,test_data);
pl = train_label(Idx);

cm = confusionmat(test_label,pl)
accuracy = sum(diag (cm))/ sum(sum(cm))
final_list(length(final_list) + 1) = accuracy
Mdl = fitcknn(train_data, train_label,'NumNeighbors',2);
pre = predict(Mdl,test_data)
count = 0;
for i=1:length(pre)
      if(pre(i)==test_label(i))
          count = count + 1;
      end
end
knn_accuracy(length(knn_accuracy) + 1) = count / length(test_label);
end

disp("--------------------")
final_list
mean(final_list)
knn_accuracy
mean(knn_accuracy)

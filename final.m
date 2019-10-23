%get the input data
raw_data = importdata("glass.data");
%
data = raw_data(:,2:10);
num_data = length(data);
label = raw_data(:,11);
visit = zeros(num_data,1);
model = [];

%loop through each data
for i = 1:num_data
    if visit(i) ~= 1
        point = data(i,:);
        cat = label(i);
        [number, center, range, visit] = find_largest_range(point, cat , data, label, visit);
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

%test data should be there 
test = [];

%prediction
%find in range and push into
result = [];

closest = intmax;

for i = 1:length(result_model)
    center = result_model(i,2:10);
    cat = result_model(i,12);
    range = result_model(i,11);
    
    %find the cloest
    if pdist2(center, test) < closest
        
    end
    
    %find anything in range
    if pdist2(center, test) <= range
         result = [result; result_model(i,:)];
    end
end

%if inrange == 0 

%if inrange == 1

%if inrange >= 1


function [number, center, range, label, visit] = find_largest_range(point, cat , data, label, visit)
range = 10000000000;
%for each point in data, find the min distance between point and 
for i = 1:len(data)
    %if different cat from the point to the given label
    if cat ~= label(i)
       dis = pdist2(data(i),point);
       if dis < range 
           range = dis;
           center = point(i);
       end
    end

end


number = 0;
%loop again to find which point is located in that region and mark visit
for i = i:len(dagta)
    if pdist2(data(i), point) < range
        %mark visit
        visit(i)  = 1;
        number = number +1;
    end
        
end
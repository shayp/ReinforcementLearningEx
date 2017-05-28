function num = Softmax(Grades)
%EPSGREEDY Softmax policy
% Grades    - The critic grades for each possible action
% num       - The chosen action's index

% TODO: Implement the softmax policy

temperature = 0.2;
Grades = exp(Grades / temperature);
sumOfProbs = sum(Grades);
Grades = Grades / sumOfProbs;
Grades = cumsum(Grades);
randProb = rand();
indexes = find (Grades < randProb);
if length(indexes) ~= 0
num = indexes(1);
else
    [~, num] = max(Grades);
end

end

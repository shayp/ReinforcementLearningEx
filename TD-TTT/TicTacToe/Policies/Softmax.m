function num = Softmax(Grades)
%EPSGREEDY Softmax policy
% Grades    - The critic grades for each possible action
% num       - The chosen action's index

% Set temperature
temperature = 0.5;

% Calculate exp of grades divided by temperature
Grades = exp(Grades / temperature);

sumOfProbs = sum(Grades);

% Run softmax on grades
Grades = Grades / sumOfProbs;

% Calculate the Cumulative distribution function
Grades = cumsum(Grades);

% Get random number and find the first value in the Cumulative distribution
% function which is bigger then the random number. The first index that
% fits will be the choosen move
randProb = rand();
indexes = find (Grades > randProb);
if length(indexes) ~= 0
num = indexes(1);
else
num = Grades(1);
end

end

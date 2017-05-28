function TDTrainForward(S, r,eta)
% Train the TD system
% S - The states' history for a given trial
% r - the reward given for each state in the given trial

% Load the network's data if needed
global Net;
LoadNet();

% Number of layers in the network
L = length(Net.W);

% Define params for the learning rule
% eta = 0.005;
Gamma = 1.0;
lambda = 0.8;

% Get game length for later use
gameLength = size(S,2);

% We define R, the predicted reward
eTrace = zeros(1,gameLength);
eTrace = cell(1, L);
for i = 1:L
    eTrace{i} = zeros(size(Net.W{i}));
end

% We run on all steps in the game except the last one
for i=2:gameLength
    [currentReward, Gradient] = TDEvaluate(S(:,i));
    % We Get V(t), V'(t)
    [currentReward, Gradient] = TDEvaluate(S(:,i));
    if i == gameLength
        deltaT = r(i) - currentReward;
    else
        nextReward = TDEvaluate(S(:,i + 1));
        deltaT = r(i) + Gamma * nextReward  - currentReward;
    end
    
    
    for layer=0:L - 1
        l = L - layer;
        eTrace{l} = Gradient{l} + Gamma * lambda * eTrace{l};
        % W = W + eta * (R(t) - V(t)) * V'(t)
        Net.W{l}(:,:) =  Net.W{l}(:,:) + eta * deltaT * eTrace{l};
    end
end
    
end


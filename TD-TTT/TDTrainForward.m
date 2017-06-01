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
Gamma = 1.00;
lambda = 0.5;

% Get game length for later use
gameLength = size(S,2);

% We define R, the predicted reward
eTrace = cell(1, L);

% Set eTrace to zero
for i = 1:L
    eTrace{i} = zeros(size(Net.W{i}));
end

% We run on all steps in the game
for i=1:gameLength
    
    % We Get V(St), V'(St)
    [currentReward, Gradient] = TDEvaluate(S(:,i));
    
    if i == gameLength
        % delta(t) = r(t) - V(St)
        deltaT = r(i) - currentReward;
    else
        nextReward = TDEvaluate(S(:,i + 1));
        % delta(t) = r(t)+ gamma * V(S(t + 1)) - V(St)
        deltaT = r(i) + Gamma * nextReward  - currentReward;
    end
    
    % We update from the last layer to the first(don't realy matter in this
    % case)
    for layer=0:L - 1
        l = L - layer;
        
        % eTrace(t) = V'(St)  + gamma * lambda * eTrace(t - 1)
        eTrace{l} = Gradient{l} + Gamma * lambda * eTrace{l};
        % W = W + eta * (R(t) - V(St)) * V'(St)
        Net.W{l}(:,:) =  Net.W{l}(:,:) + eta * deltaT * eTrace{l};
    end
end
    
end


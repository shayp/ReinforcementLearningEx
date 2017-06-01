function TDTrainBackward(S, r,eta)
% Train the TD system
% S - The states' history for a given trial
% r - the reward given for each state in the given trial
% eta - learning rate
% Load the network's data if needed
global Net;
LoadNet();

% Number of layers in the network
L = length(Net.W);

% Define params for the learning rule
Gamma = 1.0;
lambda = 0.8;

% Get game length for later use
gameLength = size(S,2);

% We define R, the predicted reward
R = zeros(1,gameLength);

% We set the last reward to the actual reward
R(gameLength) = r(end);

% We run on all steps in the game except the last one
for i=0:gameLength - 1
    % Calculate current step
    currentStep = gameLength - i;
    
    % We Calculate Expected reward: 
    if currentStep ~= gameLength
        %R(t) = r(t) + Gamma * ((1 - lambda) * V(S(t + 1)) + lambda * R(t + 1))
        R(currentStep) = r(currentStep) + Gamma * ((1 - lambda) * TDEvaluate(S(:,currentStep + 1)) + lambda * R(currentStep + 1));
    end
    % We Get V(St), V'(St)
    [currentReward, Gradient] = TDEvaluate(S(:,currentStep));
    for layer=0:L - 1
        l = L - layer;
        % W = W + eta * (R(t) - V(St)) * V'(St)
        Net.W{l}(:,:) =  Net.W{l}(:,:) + eta * (R(currentStep) - currentReward) * Gradient{l}(:,:);
    end
end
    
end


eta = 0.05;

expLength = 200;
trialSeries = 1:expLength;
u = ones(expLength ,1);
constantReward = [ones(expLength / 2,1); zeros(expLength / 2,1)];
randomReward = randi(2, expLength, 1) - 1;

wConstant = zeros(expLength,1);
wRandom = zeros(expLength,1);


for i = 1:expLength - 1
    deltaWConstant = eta * (constantReward(i) - wConstant(i) * u(i))*u(i);
    deltaWRandom = eta * (randomReward(i) - wRandom(i) * u(i))*u(i);

    wConstant(i + 1) = wConstant(i) + deltaWConstant;
    wRandom(i + 1) = wRandom(i) + deltaWRandom;
end

figure();
plot(trialSeries, wConstant, trialSeries, wRandom)
title('Rescorla-Wagner rule');
xlabel('Trial');
ylabel('Weight');
legend('Constant reward','Random Reward')
text(expLength * 0.1,wConstant(expLength * 0.1),'acquisition');
text(expLength * 0.6, wConstant(expLength * 0.6),'extinction');
%RUNTRAINING Creates a new network and run a full TD training process
clear all;
close all;
clc;

mpath = strrep(which(mfilename),[mfilename '.m'],'');
addpath([mpath 'TDNet']);
addpath([mpath 'TicTacToe']);

%% Initialize the network

InitializeNet();

%% Train by playing vs. random opponent
TdMethods = {'Backward', 'Forward'};
TDMethodUse = TdMethods{1};
GetSetPolicy(@EpsGreedy);
n_games = 500000;
train_res = Train(@TDChooseSquare, @RandomChooseSquare, n_games,TDMethodUse);

%% Plot the results of the training process

PlotTrainingRes(train_res);

%% Test

n_games = 10000;
Test(n_games);

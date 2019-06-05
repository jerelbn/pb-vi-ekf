% Prepare Variables for Plotting
clear, close all
format compact
set(0,'DefaultFigureWindowStyle','docked')
addpath('yamlmatlab')
param = ReadYaml('../param/param.yaml');

% bag_name = 'MocapCalCollect1';
bag_name = 'MocapCalCollect2';
% bag_name = 'MocapCalCollect3';
% bag_name = 'MocapCalCollect4';

plot_ekf(param, true, false, false, bag_name)

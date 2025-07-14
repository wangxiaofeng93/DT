clc
clear
% 设置参数
% rpm 参数输入的时候直接计算

fs0 = 97656; %基线条件
fs2 = 48828; %内外圈故障

%查看插值多少合适
% 每圈时长（秒）
T_rev = 1 / 25; %60/rpm  1/Hz;
% 每圈采样点数
L_rev = round(fs2 * T_rev);

%每圈角度采样点数，如1024
pointsPerRev = 1024;  
% 每个样本由几圈组成
k = 1; 


%% 正常状态  
label = 0;

normal_0_files = {'baseline_1.mat','baseline_2.mat','baseline_3.mat'}; 
[data_Nornal, label_Nornal] = generateSamplesFromDataset(normal_0_files, label, fs0, 25*60, k, pointsPerRev);


%% 内圈故障  
label = 1;

ir_0_files = {'InnerRaceFault_vload_5.mat','InnerRaceFault_vload_6.mat','InnerRaceFault_vload_7.mat'}; 
[data_IR, label_IR] = generateSamplesFromDataset(ir_0_files, label, fs2, 25*60, k, pointsPerRev);


%% 外圈故障  
label = 2;

or_0_files = {'OuterRaceFault_vload_5.mat','OuterRaceFault_vload_6.mat','OuterRaceFault_vload_7.mat'}; 
[data_OR, label_OR] = generateSamplesFromDataset(or_0_files, label, fs2, 25*60, k, pointsPerRev);


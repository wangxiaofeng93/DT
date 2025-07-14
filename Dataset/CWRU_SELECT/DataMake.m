clc
clear
% 设置参数
rpm0 = 1797;
rpm1 = 1772;
rpm2 = 1750;
rpm3 = 1730;
fs = 12000;  
sigField = 'DE_time';

%查看插值多少合适
% 每圈时长（秒）
T_rev = 60 / rpm1; %60/rpm;
% 每圈采样点数
L_rev = round(fs * T_rev);

%每圈角度采样点数，如1024
pointsPerRev = 1024;  
% 每个样本由几圈组成
k = 1; 

%% normal
label = 0;
%负载0，
normal_0_files = {'normal_0_97.mat'}; 
[data00, label00] = generateSamplesFromDataset(normal_0_files, sigField, label, fs, rpm0, k, pointsPerRev);

%负载1
normal_1_files = {'normal_1_98.mat'}; 
[data01, label01] = generateSamplesFromDataset(normal_1_files, sigField, label, fs, rpm1, k, pointsPerRev);

%负载2
normal_1_files = {'normal_2_99.mat'}; 
[data02, label02] = generateSamplesFromDataset(normal_1_files, sigField, label, fs, rpm2, k, pointsPerRev);

%负载3
normal_1_files = {'normal_3_100.mat'}; 
[data03, label03] = generateSamplesFromDataset(normal_1_files, sigField, label, fs, rpm3, k, pointsPerRev);

% 合并这些子集
data_Normal = cat(1, data00, data01, data02, data03);
label0 = cat(1, label00, label01, label02, label03);

clear data00 data01 data02 data03 label00 label01 label02 label03

%% IR
label = 1;
%负载0，
IR_0_files = {'12k_Drive_End_IR007_0_105.mat','12k_Drive_End_IR021_0_209.mat'}; 
[data00, label00] = generateSamplesFromDataset(IR_0_files, sigField, label, fs, rpm0, k, pointsPerRev);

%负载1
IR_1_files = {'12k_Drive_End_IR007_1_106.mat','12k_Drive_End_IR021_1_210.mat'}; 
[data01, label01] = generateSamplesFromDataset(IR_1_files, sigField, label, fs, rpm1, k, pointsPerRev);

%负载2
IR_2_files = {'12k_Drive_End_IR007_2_107.mat','12k_Drive_End_IR021_2_211.mat'}; 
[data02, label02] = generateSamplesFromDataset(IR_2_files, sigField, label, fs, rpm2, k, pointsPerRev);

%负载3
IR_3_files = {'12k_Drive_End_IR007_3_108.mat','12k_Drive_End_IR021_3_212.mat'}; 
[data03, label03] = generateSamplesFromDataset(IR_3_files, sigField, label, fs, rpm3, k, pointsPerRev);

% 合并这些子集
data_IR = cat(1, data00, data01, data02, data03);
label1 = cat(1, label00, label01, label02, label03);

clear data00 data01 data02 data03 label00 label01 label02 label03

%% OR
label = 2;
%负载0，
OR_0_files = {'12k_Drive_End_OR007@6_0_130.mat','12k_Drive_End_OR021@6_0_234.mat'}; 
[data00, label00] = generateSamplesFromDataset(OR_0_files, sigField, label, fs, rpm0, k, pointsPerRev);

%负载1
OR_1_files = {'12k_Drive_End_OR007@6_1_131.mat','12k_Drive_End_OR021@6_1_235.mat'}; 
[data01, label01] = generateSamplesFromDataset(OR_1_files, sigField, label, fs, rpm1, k, pointsPerRev);

%负载2
OR_2_files = {'12k_Drive_End_OR007@6_2_132.mat','12k_Drive_End_OR021@6_2_236.mat'}; 
[data02, label02] = generateSamplesFromDataset(OR_2_files, sigField, label, fs, rpm2, k, pointsPerRev);

%负载3
OR_3_files = {'12k_Drive_End_OR007@6_3_133.mat','12k_Drive_End_OR021@6_3_237.mat'}; 
[data03, label03] = generateSamplesFromDataset(OR_3_files, sigField, label, fs, rpm3, k, pointsPerRev);

% 合并这些子集
data_OR = cat(1, data00, data01, data02, data03);
label2 = cat(1, label00, label01, label02, label03);

clear data00 data01 data02 data03 label00 label01 label02 label03

clearvars -except data_Normal data_IR data_OR
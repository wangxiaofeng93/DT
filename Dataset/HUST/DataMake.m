clc
clear
% 设置参数
% rpm 参数输入的时候直接计算

fs = 25.6*1000;  
%查看插值多少合适
% 每圈时长（秒）
T_rev = 1/70; %60/rpm 1/Hz;
% 每圈采样点数
L_rev = round(fs * T_rev);

%每圈角度采样点数，如1024
pointsPerRev = 1024;  
% 每个样本由几圈组成
k = 1; 


%% 正常状态  
label = 0;

normal_0_files = {'H_60Hz.xlsx'}; 
[data00, label00] = generateSamplesFromDataset(normal_0_files, label, fs, 60*60, k, pointsPerRev);

normal_1_files = {'H_65Hz.xlsx'};
[data01, label01] = generateSamplesFromDataset(normal_1_files, label, fs, 65*60, k, pointsPerRev);

normal_2_files = {'H_70Hz.xlsx'};
[data02, label02] = generateSamplesFromDataset(normal_2_files, label, fs, 70*60, k, pointsPerRev);

normal_3_files = {'H_75Hz.xlsx'};
[data03, label03] = generateSamplesFromDataset(normal_3_files, label, fs, 75*60, k, pointsPerRev);

normal_4_files = {'H_80Hz.xlsx'};
[data04, label04] = generateSamplesFromDataset(normal_4_files, label, fs, 80*60, k, pointsPerRev);

% 合并
data_Normal = cat(1, data00, data01, data02, data03, data04);
label_Normal = cat(1, label00, label01, label02, label03, label04);

clear label00 label01 label02  label03 label04  data00 data01 data02  data03  data04 label
%% 内圈故障  
label = 1;

ir_0_files = {'0.5X_I_60Hz.xlsx','I_60Hz.xlsx'}; 
[data00, label00] = generateSamplesFromDataset(ir_0_files, label, fs, 60*60, k, pointsPerRev);

ir_1_files = {'0.5X_I_65Hz.xlsx','I_65Hz.xlsx'}; 
[data01, label01] = generateSamplesFromDataset(ir_1_files, label, fs, 65*60, k, pointsPerRev);

ir_2_files = {'0.5X_I_70Hz.xlsx','I_70Hz.xlsx'}; 
[data02, label02] = generateSamplesFromDataset(ir_2_files, label, fs, 70*60, k, pointsPerRev);

ir_3_files = {'0.5X_I_75Hz.xlsx','I_75Hz.xlsx'}; 
[data03, label03] = generateSamplesFromDataset(ir_3_files, label, fs, 75*60, k, pointsPerRev);

ir_4_files = {'0.5X_I_80Hz.xlsx','I_80Hz.xlsx'}; 
[data04, label04] = generateSamplesFromDataset(ir_4_files, label, fs, 80*60, k, pointsPerRev);



% 合并
data_IR = cat(1, data00, data01, data02, data03, data04);
label_IR = cat(1, label00, label01, label02, label03, label04);

clear label00 label01 label02  label03 label04  data00 data01 data02  data03  data04 label

%% 外圈故障  
label = 2;

Or_0_files = {'0.5X_O_60Hz.xlsx','O_60Hz.xlsx'}; 
[data00, label00] = generateSamplesFromDataset(Or_0_files, label, fs, 60*60, k, pointsPerRev);

Or_1_files = {'0.5X_O_65Hz.xlsx','O_65Hz.xlsx'}; 
[data01, label01] = generateSamplesFromDataset(Or_1_files, label, fs, 65*60, k, pointsPerRev);

Or_2_files = {'0.5X_O_70Hz.xlsx','O_70Hz.xlsx'}; 
[data02, label02] = generateSamplesFromDataset(Or_2_files, label, fs, 70*60, k, pointsPerRev);

Or_3_files = {'0.5X_O_75Hz.xlsx','O_75Hz.xlsx'}; 
[data03, label03] = generateSamplesFromDataset(Or_3_files, label, fs, 75*60, k, pointsPerRev);

Or_4_files = {'0.5X_O_80Hz.xlsx','O_80Hz.xlsx'}; 
[data04, label04] = generateSamplesFromDataset(Or_4_files, label, fs, 80*60, k, pointsPerRev);



% 合并
data_OR = cat(1, data00, data01, data02, data03, data04);
label_OR = cat(1, label00, label01, label02, label03, label04);


clear label00 label01 label02  label03 label04  data00 data01 data02  data03  data04 label
clearvars -except data_Normal data_IR data_OR
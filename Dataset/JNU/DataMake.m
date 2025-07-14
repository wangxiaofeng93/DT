clc
clear
% 设置参数
% rpm 参数输入的时候直接计算

fs = 50000;  
%查看插值多少合适
% 每圈时长（秒）
T_rev = 60 / 800; %60/rpm;
% 每圈采样点数
L_rev = round(fs * T_rev);

%每圈角度采样点数，如1024
pointsPerRev = 1024;  
% 每个样本由几圈组成
k = 1; 


%% 正常状态  
label = 0;

normal_0_files = {'n600_3_2.csv'}; 
[data00, label00] = generateSamplesFromDataset(normal_0_files, label, fs, 600, k, pointsPerRev);

normal_1_files = {'n800_3_2.csv'};
[data01, label01] = generateSamplesFromDataset(normal_1_files, label, fs, 800, k, pointsPerRev);

normal_2_files = {'n1000_3_2.csv'};
[data02, label02] = generateSamplesFromDataset(normal_2_files, label, fs, 1000, k, pointsPerRev);


% 合并
data_Normal = cat(1, data00, data01, data02);
label_Normal = cat(1, label00, label01, label02);

clear label00 label01 label02 data00 data01 data02 label
%% 内圈故障  
label = 1;

ir_0_files = {'ib600_2.csv'}; 
[data00, label00] = generateSamplesFromDataset(ir_0_files, label, fs, 600, k, pointsPerRev);

ir_1_files = {'ib800_2.csv'};
[data01, label01] = generateSamplesFromDataset(ir_1_files, label, fs, 800, k, pointsPerRev);

ir_2_files = {'ib1000_2.csv'};
[data02, label02] = generateSamplesFromDataset(ir_2_files, label, fs, 1000, k, pointsPerRev);



% 合并
data_IR = cat(1, data00, data01, data02);
label_IR = cat(1, label00, label01, label02);

clear label00 label01 label02 data00 data01 data02 label
%% 外圈故障  
label = 2;

or_0_files = {'ob600_2.csv'}; 
[data00, label00] = generateSamplesFromDataset(or_0_files, label, fs, 600, k, pointsPerRev);

or_1_files = {'ob800_2.csv'};
[data01, label01] = generateSamplesFromDataset(or_1_files, label, fs, 800, k, pointsPerRev);

or_2_files = {'ob1000_2.csv'};
[data02, label02] = generateSamplesFromDataset(or_2_files, label, fs, 1000, k, pointsPerRev);

% 合并
data_OR = cat(1, data00, data01, data02);
label_OR = cat(1, label00, label01, label02);

clear label00 label01 label02 data00 data01 data02 label
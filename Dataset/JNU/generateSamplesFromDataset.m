function [data, labels] = generateSamplesFromDataset(fileList, label, fs, rpm, k, pointsPerRev)
% generateSamplesFromDataset 从Excel文件提取振动样本（按角度重采样）
%
% 参数说明：
%   fileList        - Excel文件路径的cell数组
%   ❌sigField❌   - JNU不使用此参数，为兼容性保留）
%   label           - 故障类型标签
%   fs              - 采样率(Hz)
%   rpm             - 转速(RPM)
%   k               - 每个样本由几圈组成
%   pointsPerRev    - 每圈角度采样点数

allSegments = {};
for i = 1:length(fileList)
    file = fileList{i};
    % 读取Excel数据
    % 创建导入选项
    opts = delimitedTextImportOptions("NumVariables", 1);  % 只导入 1 列
    opts.DataLines = [1, Inf];
    opts.Delimiter = ",";
    opts.VariableNames = "X";
    opts.VariableTypes = "double";

    % 读取数据、转换为列向量
    tbl = readtable(file, opts);
    signal = tbl.X;

    
    % 确保列向量
    if isrow(signal); signal = signal'; end
    
    % 角度重采样
    resampledSegments = angleResampleByRevolution(signal, fs, rpm, pointsPerRev, 'spline');
    allSegments = [allSegments, resampledSegments];
end

% 拼接k圈为单个样本
totalRevs = length(allSegments);
numSamples = floor(totalRevs / k);
data = zeros(numSamples, k * pointsPerRev);

for i = 1:numSamples
    segs = allSegments((i-1)*k + 1 : i*k);
    data(i, :) = [segs{:}]; % 水平拼接
end

labels = label * ones(numSamples, 1);
end
function [data, labels] = generateSamplesFromDataset(fileList, sigField, label, fs, rpm, k, pointsPerRev)
% generateSamplesFromDataset 按角度重采样提取样本
%
% 参数说明：
%   fileList        - 包含.mat文件路径的cell数组
%   sigField        - 要提取的变量名，例如 'DE_time' 或 'vibration'
%   label           - 整数类别标签（如 0, 1, 2）
%   fs              - 采样率 (Hz)
%   rpm             - 转速 (RPM)
%   k               - 每个样本由几圈组成
%   pointsPerRev    - 每圈角度采样点数，如1024

    allSegments = {};

    for i = 1:length(fileList)
        file = fileList{i};
        matData = load(file);
        % 获取所有字段名
        fieldNames = fieldnames(matData);
        matchIdx = find(contains(fieldNames, sigField));
        
        if isempty(matchIdx)
            error(['文件中未找到以 ', sigField, '结尾的字段: ', file]);
        end
        
        sigFieldMatched = fieldNames{matchIdx(1)};
        signal = matData.(sigFieldMatched);

        % 一维列向量保证
        if isrow(signal)
            signal = signal';
        end

        % 重采样为角度域段
        resampledSegments = angleResampleByRevolution(signal, fs, rpm, pointsPerRev, 'spline');

        allSegments = [allSegments, resampledSegments];
    end

    % 构造样本：每k圈拼接为一个样本
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

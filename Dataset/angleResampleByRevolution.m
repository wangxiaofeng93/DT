function resampledSegments = angleResampleByRevolution(signal, samplingRate, rpm, pointsPerRevolution, interpMethod)
% angleResampleByRevolution 按整圈切割信号并进行角度重采样
%
% 输入参数：
%   signal              - 一维时域信号向量
%   samplingRate        - 采样率 (Hz)
%   rpm                 - 转速 (转/分钟)
%   pointsPerRevolution - 每圈目标角度采样点数（默认360）
%   interpMethod        - 插值方法（'linear', 'spline', 'pchip' 等，默认 'spline'）
%
% 输出：
%   resampledSegments   - 包含每圈角度域信号的 cell 数组，每个 cell 是 1×N 的向量

    if nargin < 4
        pointsPerRevolution = 360;
    end
    if nargin < 5
        interpMethod = 'spline'; % 对应 Python 的 'cubic'
    end

    % 每圈时长（秒）
    T_rev = 60 / rpm;

    % 每圈采样点数
    L_rev = round(samplingRate * T_rev);

    % 可切割圈数
    numRevs = floor(length(signal) / L_rev);
    if numRevs < 1
        error('输入信号太短，无法切割出完整的圈');
    end

    % 预定义输出
    resampledSegments = cell(1, numRevs);

    % 对每一圈分段并进行角度重采样
    for i = 1:numRevs
        seg = signal((i-1)*L_rev + 1 : i*L_rev);  % 当前圈段
        L = length(seg);

        theta_orig = linspace(0, 360, L);         % 原始角度位置
        theta_target = linspace(0, 360, pointsPerRevolution);  % 目标角度点

        % 插值重采样
        seg_interp = interp1(theta_orig, seg, theta_target, interpMethod);

        % 存入 cell 数组
        resampledSegments{i} = seg_interp;
    end
end

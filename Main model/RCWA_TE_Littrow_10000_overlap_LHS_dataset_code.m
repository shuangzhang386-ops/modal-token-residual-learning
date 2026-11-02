%% ============================================================
%  TE 偏振、利特罗入射、一维矩形深刻蚀石英光栅
%  RCWA 扫参生成数据集
%
%  数据用途：
%  1. RCWA 生成严格标签 eta0_RCWA, etam1_RCWA
%  2. 简化模理论生成 modal features:
%     neff0, neff1, phi0, phi1, eta0_SMM, etam1_SMM
%  3. 后续训练神经网络学习：
%     几何参数 + modal token -> RCWA效率
%     或者 modal token -> RCWA - SMM 残差
%
%  光栅模型：
%  - 一维矩形石英光栅
%  - 入射介质为空气
%  - 光栅脊为 SiO2
%  - 槽区为空气
%  - 基底为 SiO2
%  - TE 偏振
%  - 利特罗入射
%
%  作者建议：
%  先 TEST_MODE = true 跑小数据检查能量守恒；
%  确认无误后再 TEST_MODE = false 生成 10000 条数据。
%  本版本：LHS 拉丁超立方采样。
%% ============================================================

clear; clc; close all;

%% ===================== 0. 是否测试模式 =====================
TEST_MODE = false;   % true: 先跑小数据；false: 跑完整 10000 条

%% ===================== 1. 基本光学参数 =====================
lambda0 = 1064e-9;          % 工作波长，单位 m
n_inc   = 1.0;              % 入射介质：空气
n_air   = 1.0;              % 光栅槽区：空气
n_sio2  = 1.44964;          % 1064 nm 附近 SiO2 折射率
n_sub   = n_sio2;           % 基底：SiO2

N_harm = 15;                % RCWA 傅里叶谐波阶数，实际阶数为 2*N_harm+1
                            % 建议后续做收敛性验证：11, 15, 21, 31

N_overlap = 801;             % 重叠积分的 x 方向采样点数；越大越精确，计算越慢


%% ===================== 2. LHS 采样范围 =====================
% 本版本将原来的均匀网格采样改为拉丁超立方采样（Latin Hypercube Sampling, LHS）。
% 独立采样变量：Lambda, f, h。
% 利特罗入射角 thetaL 不单独采样，而是由当前 Lambda 自动计算：
% theta_L = asin(lambda0 / (2*n_inc*Lambda))
%
% 主数据集范围：
% Lambda/lambda = 0.60 ~ 1.20
% f             = 0.35 ~ 0.75
% h             = 0.5  ~ 4.5 um
%
% 注意：这里的 LHS 使用自定义实现，不依赖 lhsdesign 工具箱。

rng(2026);   % 固定随机种子，保证数据集可复现

Lambda_ratio_min = 0.60;
Lambda_ratio_max = 1.20;
f_min = 0.35;
f_max = 0.75;
h_min = 0.5e-6;
h_max = 4.5e-6;

if TEST_MODE
    num_samples = 500;
else
    num_samples = 10000;
end

X_lhs = lhs_unit_custom(num_samples, 3);

Lambda_samples = (Lambda_ratio_min + (Lambda_ratio_max - Lambda_ratio_min) * X_lhs(:,1)) * lambda0;
f_samples      =  f_min + (f_max - f_min) * X_lhs(:,2);
h_samples      =  h_min + (h_max - h_min) * X_lhs(:,3);

fprintf('预计样本数：%d\n', num_samples);
fprintf('采样方式：LHS 拉丁超立方采样\n');
fprintf('lambda0 = %.3f nm\n', lambda0*1e9);
fprintf('period range = %.3f ~ %.3f nm\n', min(Lambda_samples)*1e9, max(Lambda_samples)*1e9);
fprintf('period/lambda range = %.6f ~ %.6f\n', min(Lambda_samples)/lambda0, max(Lambda_samples)/lambda0);
fprintf('f range = %.3f ~ %.3f\n', min(f_samples), max(f_samples));
fprintf('h range = %.3f ~ %.3f um\n', min(h_samples)*1e6, max(h_samples)*1e6);
fprintf('N_harm = %d\n\n', N_harm);

%% ===================== 3. 预分配结果 =====================
Lambda_col       = zeros(num_samples,1);
f_col            = zeros(num_samples,1);
b_col            = zeros(num_samples,1);
h_col            = zeros(num_samples,1);
lambda_col       = zeros(num_samples,1);
theta_rad_col    = zeros(num_samples,1);
theta_deg_col    = zeros(num_samples,1);

eta0_RCWA_col    = zeros(num_samples,1);
etam1_RCWA_col   = zeros(num_samples,1);
Tsum_col         = zeros(num_samples,1);
Rsum_col         = zeros(num_samples,1);
energy_col       = zeros(num_samples,1);

neff0_col        = zeros(num_samples,1);
neff1_col        = zeros(num_samples,1);
delta_neff_col   = zeros(num_samples,1);

phi0_col         = zeros(num_samples,1);
phi1_col         = zeros(num_samples,1);
delta_phi_col    = zeros(num_samples,1);

sin_phi0_col     = zeros(num_samples,1);
cos_phi0_col     = zeros(num_samples,1);
sin_phi1_col     = zeros(num_samples,1);
cos_phi1_col     = zeros(num_samples,1);

sin_dphi_col     = zeros(num_samples,1);
cos_dphi_col     = zeros(num_samples,1);

eta0_SMM_col     = zeros(num_samples,1);
etam1_SMM_col    = zeros(num_samples,1);
res_eta0_col     = zeros(num_samples,1);
res_etam1_col    = zeros(num_samples,1);

% 重叠积分：mode q 与 diffraction order p 的耦合强度
% 公式参考：|int E_p^*(x) u_q(x) dx|^2 / [int |E_p|^2 dx · int |u_q|^2 dx]
overlap_m0_o0_col   = zeros(num_samples,1);   % mode 0 <-> 0th order
overlap_m0_om1_col  = zeros(num_samples,1);   % mode 0 <-> -1st order
overlap_m1_o0_col   = zeros(num_samples,1);   % mode 1 <-> 0th order
overlap_m1_om1_col  = zeros(num_samples,1);   % mode 1 <-> -1st order

overlap_m0_o1_col   = zeros(num_samples,1);   % 附加检查：mode 0 <-> +1st order
overlap_m1_o1_col   = zeros(num_samples,1);   % 附加检查：mode 1 <-> +1st order

status_col       = repmat({''}, num_samples, 1);

%% ===================== 4. 主循环扫参 =====================
tic;

for count = 1:num_samples

    Lambda = Lambda_samples(count);
    f      = f_samples(count);
    h      = h_samples(count);
    b      = f * Lambda;

    % 利特罗角：由当前 Lambda 自动决定，不单独采样
    arg = lambda0 / (2*n_inc*Lambda);

    if abs(arg) > 1
        Lambda_col(count)     = Lambda;
        f_col(count)          = f;
        b_col(count)          = b;
        h_col(count)          = h;
        lambda_col(count)     = lambda0;
        theta_rad_col(count)  = NaN;
        theta_deg_col(count)  = NaN;

        eta0_RCWA_col(count)  = NaN;
        etam1_RCWA_col(count) = NaN;
        Tsum_col(count)       = NaN;
        Rsum_col(count)       = NaN;
        energy_col(count)     = NaN;

        neff0_col(count)      = NaN;
        neff1_col(count)      = NaN;
        delta_neff_col(count) = NaN;

        phi0_col(count)       = NaN;
        phi1_col(count)       = NaN;
        delta_phi_col(count)  = NaN;

        sin_phi0_col(count)   = NaN;
        cos_phi0_col(count)   = NaN;
        sin_phi1_col(count)   = NaN;
        cos_phi1_col(count)   = NaN;

        sin_dphi_col(count)   = NaN;
        cos_dphi_col(count)   = NaN;

        eta0_SMM_col(count)   = NaN;
        etam1_SMM_col(count)  = NaN;

        res_eta0_col(count)   = NaN;
        res_etam1_col(count)  = NaN;

        overlap_m0_o0_col(count)   = NaN;
        overlap_m0_om1_col(count)  = NaN;
        overlap_m1_o0_col(count)   = NaN;
        overlap_m1_om1_col(count)  = NaN;
        overlap_m0_o1_col(count)   = NaN;
        overlap_m1_o1_col(count)   = NaN;

        status_col{count} = 'ERROR: Lambda does not satisfy Littrow condition';
        continue;
    end

    thetaL = asin(arg);

    try
        %% ---------- 4.1 RCWA 严格计算 ----------
        out_rcwa = rcwa_lamellar_TE_1layer( ...
            Lambda, f, h, lambda0, thetaL, ...
            n_inc, n_sub, n_sio2, n_air, N_harm);

        eta0_RCWA  = out_rcwa.eta0;
        etam1_RCWA = out_rcwa.eta_m1;

        %% ---------- 4.2 简化模理论计算 neff 与重叠积分 ----------
        out_smm = smm_littrow_TE_two_modes( ...
            Lambda, f, h, lambda0, thetaL, ...
            n_air, n_sio2, N_overlap);

        neff0      = out_smm.neff0;
        neff1      = out_smm.neff1;
        delta_neff = neff0 - neff1;

        phi0       = out_smm.phi0;
        phi1       = out_smm.phi1;
        delta_phi  = out_smm.delta_phi;

        eta0_SMM   = out_smm.eta0_SMM;
        etam1_SMM  = out_smm.etam1_SMM;

        overlap_m0_o0  = out_smm.overlap_m0_o0;
        overlap_m0_om1 = out_smm.overlap_m0_om1;
        overlap_m1_o0  = out_smm.overlap_m1_o0;
        overlap_m1_om1 = out_smm.overlap_m1_om1;
        overlap_m0_o1  = out_smm.overlap_m0_o1;
        overlap_m1_o1  = out_smm.overlap_m1_o1;

        %% ---------- 4.3 保存数据 ----------
        Lambda_col(count)     = Lambda;
        f_col(count)          = f;
        b_col(count)          = b;
        h_col(count)          = h;
        lambda_col(count)     = lambda0;
        theta_rad_col(count)  = thetaL;
        theta_deg_col(count)  = thetaL * 180/pi;

        eta0_RCWA_col(count)  = eta0_RCWA;
        etam1_RCWA_col(count) = etam1_RCWA;
        Tsum_col(count)       = out_rcwa.Tsum;
        Rsum_col(count)       = out_rcwa.Rsum;
        energy_col(count)     = out_rcwa.energy_sum;

        neff0_col(count)      = neff0;
        neff1_col(count)      = neff1;
        delta_neff_col(count) = delta_neff;

        phi0_col(count)       = phi0;
        phi1_col(count)       = phi1;
        delta_phi_col(count)  = delta_phi;

        sin_phi0_col(count)   = sin(phi0);
        cos_phi0_col(count)   = cos(phi0);
        sin_phi1_col(count)   = sin(phi1);
        cos_phi1_col(count)   = cos(phi1);

        sin_dphi_col(count)   = sin(delta_phi);
        cos_dphi_col(count)   = cos(delta_phi);

        eta0_SMM_col(count)   = eta0_SMM;
        etam1_SMM_col(count)  = etam1_SMM;

        res_eta0_col(count)   = eta0_RCWA  - eta0_SMM;
        res_etam1_col(count)  = etam1_RCWA - etam1_SMM;

        overlap_m0_o0_col(count)   = overlap_m0_o0;
        overlap_m0_om1_col(count)  = overlap_m0_om1;
        overlap_m1_o0_col(count)   = overlap_m1_o0;
        overlap_m1_om1_col(count)  = overlap_m1_om1;
        overlap_m0_o1_col(count)   = overlap_m0_o1;
        overlap_m1_o1_col(count)   = overlap_m1_o1;

        status_col{count}     = 'OK';

    catch ME

        Lambda_col(count)     = Lambda;
        f_col(count)          = f;
        b_col(count)          = b;
        h_col(count)          = h;
        lambda_col(count)     = lambda0;
        theta_rad_col(count)  = thetaL;
        theta_deg_col(count)  = thetaL * 180/pi;

        eta0_RCWA_col(count)  = NaN;
        etam1_RCWA_col(count) = NaN;
        Tsum_col(count)       = NaN;
        Rsum_col(count)       = NaN;
        energy_col(count)     = NaN;

        neff0_col(count)      = NaN;
        neff1_col(count)      = NaN;
        delta_neff_col(count) = NaN;

        phi0_col(count)       = NaN;
        phi1_col(count)       = NaN;
        delta_phi_col(count)  = NaN;

        sin_phi0_col(count)   = NaN;
        cos_phi0_col(count)   = NaN;
        sin_phi1_col(count)   = NaN;
        cos_phi1_col(count)   = NaN;

        sin_dphi_col(count)   = NaN;
        cos_dphi_col(count)   = NaN;

        eta0_SMM_col(count)   = NaN;
        etam1_SMM_col(count)  = NaN;

        res_eta0_col(count)   = NaN;
        res_etam1_col(count)  = NaN;

        overlap_m0_o0_col(count)   = NaN;
        overlap_m0_om1_col(count)  = NaN;
        overlap_m1_o0_col(count)   = NaN;
        overlap_m1_om1_col(count)  = NaN;
        overlap_m0_o1_col(count)   = NaN;
        overlap_m1_o1_col(count)   = NaN;

        status_col{count}     = ['ERROR: ', ME.message];
    end

    if mod(count,100) == 0
        fprintf('已完成 %d / %d，用时 %.1f s\n', count, num_samples, toc);
    end
end

%% ===================== 5. 整理表格并保存 =====================
valid_rows = 1:num_samples;

T = table;

T.Lambda_m       = Lambda_col(valid_rows);
T.f              = f_col(valid_rows);
T.b_m            = b_col(valid_rows);
T.h_m            = h_col(valid_rows);
T.lambda_m       = lambda_col(valid_rows);
T.theta_rad      = theta_rad_col(valid_rows);
T.theta_deg      = theta_deg_col(valid_rows);

T.eta0_RCWA_TE   = eta0_RCWA_col(valid_rows);
T.etam1_RCWA_TE  = etam1_RCWA_col(valid_rows);
T.Tsum           = Tsum_col(valid_rows);
T.Rsum           = Rsum_col(valid_rows);
T.energy_sum     = energy_col(valid_rows);

T.neff0_TE       = neff0_col(valid_rows);
T.neff1_TE       = neff1_col(valid_rows);
T.delta_neff_TE  = delta_neff_col(valid_rows);

T.phi0_TE        = phi0_col(valid_rows);
T.phi1_TE        = phi1_col(valid_rows);
T.delta_phi_TE   = delta_phi_col(valid_rows);

T.sin_phi0_TE    = sin_phi0_col(valid_rows);
T.cos_phi0_TE    = cos_phi0_col(valid_rows);
T.sin_phi1_TE    = sin_phi1_col(valid_rows);
T.cos_phi1_TE    = cos_phi1_col(valid_rows);

T.sin_dphi_TE    = sin_dphi_col(valid_rows);
T.cos_dphi_TE    = cos_dphi_col(valid_rows);

T.eta0_SMM_TE    = eta0_SMM_col(valid_rows);
T.etam1_SMM_TE   = etam1_SMM_col(valid_rows);

T.res_eta0_TE    = res_eta0_col(valid_rows);
T.res_etam1_TE   = res_etam1_col(valid_rows);

T.overlap_mode0_order0_TE   = overlap_m0_o0_col(valid_rows);
T.overlap_mode0_orderm1_TE  = overlap_m0_om1_col(valid_rows);
T.overlap_mode1_order0_TE   = overlap_m1_o0_col(valid_rows);
T.overlap_mode1_orderm1_TE  = overlap_m1_om1_col(valid_rows);
T.overlap_mode0_order1_TE   = overlap_m0_o1_col(valid_rows);
T.overlap_mode1_order1_TE   = overlap_m1_o1_col(valid_rows);

T.status         = status_col(valid_rows);

if TEST_MODE
    filename = 'RCWA_TE_Littrow_TEST_LHS_dataset.csv';
else
    filename = 'RCWA_TE_Littrow_10000_overlap_LHS_dataset.csv';
end

writetable(T, filename);

fprintf('\n========== 扫参完成 ==========\n');
fprintf('保存文件：%s\n', filename);
fprintf('总样本数：%d\n', height(T));

ok_idx = strcmp(T.status, 'OK');
fprintf('成功样本数：%d\n', sum(ok_idx));
fprintf('失败样本数：%d\n', sum(~ok_idx));

fprintf('\n能量守恒检查，仅统计 OK 样本：\n');
fprintf('mean(R+T) = %.6f\n', mean(T.energy_sum(ok_idx), 'omitnan'));
fprintf('min(R+T)  = %.6f\n', min(T.energy_sum(ok_idx), [], 'omitnan'));
fprintf('max(R+T)  = %.6f\n', max(T.energy_sum(ok_idx), [], 'omitnan'));

fprintf('\nRCWA 与 SMM 初步误差，仅统计 OK 样本：\n');
fprintf('mean abs residual eta0   = %.6f\n', mean(abs(T.res_eta0_TE(ok_idx)), 'omitnan'));
fprintf('mean abs residual eta-1  = %.6f\n', mean(abs(T.res_etam1_TE(ok_idx)), 'omitnan'));

fprintf('\n重叠积分检查，仅统计 OK 样本：\n');
fprintf('mean overlap mode0-order0  = %.6f\n', mean(T.overlap_mode0_order0_TE(ok_idx), 'omitnan'));
fprintf('mean overlap mode0-order-1 = %.6f\n', mean(T.overlap_mode0_orderm1_TE(ok_idx), 'omitnan'));
fprintf('mean overlap mode1-order0  = %.6f\n', mean(T.overlap_mode1_order0_TE(ok_idx), 'omitnan'));
fprintf('mean overlap mode1-order-1 = %.6f\n', mean(T.overlap_mode1_orderm1_TE(ok_idx), 'omitnan'));

%% ===================== 6. 简单画图检查 =====================
figure;
scatter(T.eta0_SMM_TE(ok_idx), T.eta0_RCWA_TE(ok_idx), 15, 'filled');
xlabel('\eta_0^{SMM}');
ylabel('\eta_0^{RCWA}');
title('0级效率：SMM vs RCWA');
grid on;
axis equal;

figure;
scatter(T.etam1_SMM_TE(ok_idx), T.etam1_RCWA_TE(ok_idx), 15, 'filled');
xlabel('\eta_{-1}^{SMM}');
ylabel('\eta_{-1}^{RCWA}');
title('-1级效率：SMM vs RCWA');
grid on;
axis equal;

figure;
histogram(T.energy_sum(ok_idx), 50);
xlabel('R + T');
ylabel('count');
title('能量守恒分布');
grid on;

%% ============================================================
%  局部函数 1：一维矩形光栅 TE-RCWA
%% ============================================================
function out = rcwa_lamellar_TE_1layer(Lambda, f, h, lambda0, theta, ...
                                       n_inc, n_sub, n_ridge, n_groove, N_harm)

    k0 = 2*pi/lambda0;

    orders = (-N_harm:N_harm).';
    Nh = numel(orders);

    % 归一化横向波矢 Kx = kx/k0
    % 透射/反射级次：
    % Kx_m = n_inc*sin(theta) + m*lambda/Lambda
    Kx = n_inc*sin(theta) + orders * lambda0/Lambda;

    % 上下均匀区域的归一化 kz
    q_inc = physical_sqrt(n_inc^2 - Kx.^2);
    q_sub = physical_sqrt(n_sub^2 - Kx.^2);

    q0_inc = n_inc*cos(theta);

    %% ---------- 介电常数傅里叶卷积矩阵 ----------
    eps_r = n_ridge^2;
    eps_g = n_groove^2;
    deps  = eps_r - eps_g;

    p_all = (-2*N_harm:2*N_harm).';
    eps_fourier = zeros(numel(p_all),1);

    for idx = 1:numel(p_all)
        p = p_all(idx);

        if p == 0
            eps_fourier(idx) = eps_g + deps*f;
        else
            % ridge 放在周期中心，傅里叶系数为实数
            eps_fourier(idx) = deps * sin(pi*p*f)/(pi*p);
        end
    end

    E = zeros(Nh, Nh);

    for row = 1:Nh
        for col = 1:Nh
            p = orders(row) - orders(col);
            idx = p + 2*N_harm + 1;
            E(row,col) = eps_fourier(idx);
        end
    end

    %% ---------- TE 本征问题 ----------
    % TE 标量 RCWA:
    % d2Ey/dz2 + k0^2*(E - Kx^2)*Ey = 0
    A = E - diag(Kx.^2);

    [W, D] = eig(A);
    q_layer = physical_sqrt(diag(D));
    Q = diag(q_layer);

    %% ---------- 单层传播 ----------
    X_forward  = diag(exp(1i*k0*q_layer*h));
    X_backward = diag(exp(-1i*k0*q_layer*h));

    Ivec = zeros(Nh,1);
    idx0 = find(orders == 0, 1);
    Ivec(idx0) = 1;

    Qinc = diag(q_inc);
    Qsub = diag(q_sub);

    Z  = zeros(Nh,Nh);
    Id = eye(Nh);

    %% ---------- 边界条件线性方程 ----------
    % 未知量：
    % R: 反射级次系数
    % a: 光栅层正向本征模系数
    % b: 光栅层反向本征模系数
    % T: 透射级次系数
    %
    % z=0:
    % I + R = W(a+b)
    % Qinc(I-R) = WQ(a-b)
    %
    % z=h:
    % W(Xa+Xb) = T
    % WQ(Xa-Xb) = Qsub*T

    M = [ ...
         Id,     -W,              -W,              Z;
        -Qinc,   -W*Q,             W*Q,            Z;
         Z,       W*X_forward,     W*X_backward,  -Id;
         Z,       W*Q*X_forward,  -W*Q*X_backward,-Qsub ...
        ];

    rhs = [ ...
        -Ivec;
        -Qinc*Ivec;
         zeros(Nh,1);
         zeros(Nh,1) ...
        ];

    sol = M \ rhs;

    R = sol(1:Nh);
    T = sol(3*Nh+1:4*Nh);

    %% ---------- 衍射效率 ----------
    DE_R = zeros(Nh,1);
    DE_T = zeros(Nh,1);

    for m = 1:Nh
        if real(q_inc(m)) > 1e-12
            DE_R(m) = real(q_inc(m)/q0_inc) * abs(R(m))^2;
        else
            DE_R(m) = 0;
        end

        if real(q_sub(m)) > 1e-12
            DE_T(m) = real(q_sub(m)/q0_inc) * abs(T(m))^2;
        else
            DE_T(m) = 0;
        end
    end

    idx_m1 = find(orders == -1, 1);

    out.orders = orders;
    out.R = R;
    out.T = T;
    out.DE_R = DE_R;
    out.DE_T = DE_T;

    out.eta0 = DE_T(idx0);
    out.eta_m1 = DE_T(idx_m1);

    out.Rsum = sum(DE_R);
    out.Tsum = sum(DE_T);
    out.energy_sum = out.Rsum + out.Tsum;
end

%% ============================================================
%  局部函数 2：TE 利特罗简化模理论，两传播模
%% ============================================================
function out = smm_littrow_TE_two_modes(Lambda, f, h, lambda0, theta, n1, n2, N_overlap)

    k0 = 2*pi/lambda0;

    % alpha = 入射横向波矢
    alpha = k0*n1*sin(theta);

    target = cos(alpha*Lambda);

    % 求解色散方程 F(neff^2) = cos(alpha*Lambda)
    % 对利特罗入射，理论上 target = -1
    roots_neff = find_te_modal_roots(Lambda, f, lambda0, n1, n2, target);

    if numel(roots_neff) < 2
        error('SMM 未找到两个传播模根。当前找到 %d 个。', numel(roots_neff));
    end

    % 按有效折射率从大到小排序，取前两个传播模
    roots_neff = sort(roots_neff, 'descend');

    neff0 = roots_neff(1);
    neff1 = roots_neff(2);

    phi0 = k0*neff0*h;
    phi1 = k0*neff1*h;

    delta_phi = phi0 - phi1;

    % 利特罗两模简化公式
    eta0_SMM  = cos(delta_phi/2)^2;
    etam1_SMM = sin(delta_phi/2)^2;

    % 根据论文公式(3.3)计算模式与衍射级次之间的重叠积分
    overlap0 = compute_TE_mode_order_overlaps( ...
        neff0, Lambda, f, lambda0, theta, n1, n2, N_overlap);
    overlap1 = compute_TE_mode_order_overlaps( ...
        neff1, Lambda, f, lambda0, theta, n1, n2, N_overlap);

    out.neff0 = neff0;
    out.neff1 = neff1;
    out.phi0 = phi0;
    out.phi1 = phi1;
    out.delta_phi = delta_phi;
    out.eta0_SMM = eta0_SMM;
    out.etam1_SMM = etam1_SMM;

    out.overlap_m0_o0  = overlap0.order0;
    out.overlap_m0_om1 = overlap0.orderm1;
    out.overlap_m1_o0  = overlap1.order0;
    out.overlap_m1_om1 = overlap1.orderm1;
    out.overlap_m0_o1  = overlap0.order1;
    out.overlap_m1_o1  = overlap1.order1;
end

%% ============================================================
%  局部函数 3：计算 TE 模式与 0/-1/+1 衍射级次的重叠积分
%% ============================================================
function overlap = compute_TE_mode_order_overlaps(neff, Lambda, f, lambda0, theta, n1, n2, N_grid)

    ov0  = overlap_single_TE(neff, Lambda, f, lambda0, theta, n1, n2,  0, N_grid);
    ovm1 = overlap_single_TE(neff, Lambda, f, lambda0, theta, n1, n2, -1, N_grid);
    ov1  = overlap_single_TE(neff, Lambda, f, lambda0, theta, n1, n2,  1, N_grid);

    overlap.order0  = ov0;
    overlap.orderm1 = ovm1;
    overlap.order1  = ov1;
end

%% ============================================================
%  局部函数 4：单个 TE 模式与指定衍射级次 p 的重叠积分
%% ============================================================
function ov = overlap_single_TE(neff, Lambda, f, lambda0, theta, n1, n2, p_order, N_grid)

    k0 = 2*pi/lambda0;
    alpha = k0*n1*sin(theta);

    d1 = (1-f) * Lambda;   % 槽区宽度
    d2 = f * Lambda;       % 脊区宽度

    k1 = k0 * sqrt(complex(n1^2 - neff^2));
    k2 = k0 * sqrt(complex(n2^2 - neff^2));

    e1p = exp( 1i*k1*d1);
    e1m = exp(-1i*k1*d1);
    e2p = exp( 1i*k2*d2);
    e2m = exp(-1i*k2*d2);
    P   = exp(1i*alpha*Lambda);

    % 求解周期波导 TE 模式的场系数：
    % 槽区 u1=A exp(ik1 x)+B exp(-ik1 x)
    % 脊区 u2=C exp[ik2(x-d1)]+D exp[-ik2(x-d1)]
    M = [ ...
          e1p,       e1m,      -1,       -1;
          k1*e1p,   -k1*e1m,   -k2,       k2;
         -P,         -P,        e2p,      e2m;
         -P*k1,       P*k1,     k2*e2p,  -k2*e2m ...
        ];

    [~,~,V] = svd(M);
    c = V(:,end);
    A = c(1); B = c(2); C = c(3); D = c(4);

    x = linspace(0, Lambda, N_grid).';
    u = zeros(size(x));

    idx1 = x <= d1;
    u(idx1) = A*exp( 1i*k1*x(idx1)) + B*exp(-1i*k1*x(idx1));

    x2 = x(~idx1) - d1;
    u(~idx1) = C*exp( 1i*k2*x2) + D*exp(-1i*k2*x2);

    % 第 p 个衍射级次的 TE 电场横向分布
    kxp = k0*(n1*sin(theta) + p_order*lambda0/Lambda);
    Ep = exp(-1i*kxp*x);

    % 归一化重叠积分。这里使用复内积 conj(Ep).*u，数值上对应论文公式(3.3)。
    numerator = abs(trapz(x, conj(Ep).*u))^2;
    denominator = trapz(x, abs(Ep).^2) * trapz(x, abs(u).^2);

    ov = real(numerator / denominator);
    ov = max(0, min(1, ov));
end

%% ============================================================
%  局部函数 5：寻找 TE 模式有效折射率根
%% ============================================================
function roots_neff = find_te_modal_roots(Lambda, f, lambda0, n1, n2, target)

    % n1: 槽区/空气折射率
    % n2: 脊区/石英折射率
    %
    % 色散方程：
    % F = cos[k1(1-f)Lambda] cos[k2 f Lambda]
    %     - (k1^2+k2^2)/(2 k1 k2)
    %       sin[k1(1-f)Lambda] sin[k2 f Lambda]
    %
    % F = cos(alpha Lambda)

    n_min = 1e-6;
    n_max = n2 - 1e-6;

    N_scan = 4000;
    neff_grid = linspace(n_min, n_max, N_scan);

    y = nan(size(neff_grid));

    for i = 1:N_scan
        ne = neff_grid(i);
        val = te_dispersion_F(ne, Lambda, f, lambda0, n1, n2) - target;
        y(i) = real(val);
    end

    roots_neff = [];

    for i = 1:N_scan-1

        y1 = y(i);
        y2 = y(i+1);

        if ~isfinite(y1) || ~isfinite(y2)
            continue;
        end

        if y1 == 0
            root_candidate = neff_grid(i);
            roots_neff = [roots_neff; root_candidate]; %#ok<AGROW>
        elseif y1*y2 < 0
            a = neff_grid(i);
            b = neff_grid(i+1);

            try
                root_candidate = fzero(@(ne) real(te_dispersion_F(ne, Lambda, f, lambda0, n1, n2) - target), [a,b]);

                if root_candidate > n_min && root_candidate < n_max
                    roots_neff = [roots_neff; root_candidate]; %#ok<AGROW>
                end

            catch
                % 忽略该区间
            end
        end
    end

    % 去重
    if ~isempty(roots_neff)
        roots_neff = sort(roots_neff(:));
        roots_unique = roots_neff(1);

        for i = 2:numel(roots_neff)
            if abs(roots_neff(i) - roots_unique(end)) > 1e-5
                roots_unique = [roots_unique; roots_neff(i)]; %#ok<AGROW>
            end
        end

        roots_neff = roots_unique;
    end
end

%% ============================================================
%  局部函数 6：TE 色散方程左端 F(neff)
%% ============================================================
function F = te_dispersion_F(neff, Lambda, f, lambda0, n1, n2)

    k0 = 2*pi/lambda0;

    k1 = k0 * sqrt(complex(n1^2 - neff^2));
    k2 = k0 * sqrt(complex(n2^2 - neff^2));

    d1 = (1-f) * Lambda;   % 槽区宽度
    d2 = f * Lambda;       % 脊区宽度

    F = cos(k1*d1)*cos(k2*d2) ...
        - ((k1^2 + k2^2)/(2*k1*k2)) * sin(k1*d1)*sin(k2*d2);
end

%% ============================================================
%  局部函数 7：选择物理平方根分支
%% ============================================================
function q = physical_sqrt(x)

    q = sqrt(complex(x));

    for i = 1:numel(q)
        if real(q(i)) < 0
            q(i) = -q(i);
        end

        if abs(real(q(i))) < 1e-12 && imag(q(i)) < 0
            q(i) = -q(i);
        end
    end
end

%% ============================================================
%  局部函数 8：自定义 LHS 采样，不依赖 lhsdesign 工具箱
%% ============================================================
function X = lhs_unit_custom(N, D)

    % 输出 X 为 N × D，范围在 (0,1) 内。
    % 每一维都被均匀分成 N 层，每层随机取一个点，然后随机打乱。
    X = zeros(N, D);

    for d = 1:D
        edges_left = (0:N-1)' / N;
        u = rand(N, 1) / N;
        vals = edges_left + u;
        X(:, d) = vals(randperm(N));
    end
end

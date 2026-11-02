%% ============================================================
% RCWA and SMM convergence validation for the TE Littrow grating
%
% Based on:
%   RCWA_TE_Littrow_10000_overlap_LHS_dataset_code(4).m
%
% Purpose:
%   1. Check RCWA Fourier-truncation convergence.
%   2. Check SMM modal-root search-grid convergence.
%   3. Optionally check SMM modal-overlap integration convergence.
%   4. Compare stable RCWA results at N_harm = 15 with the original
%      RCWA labels stored in the dataset.
%
% IMPORTANT:
%   - Put this script in the same folder as
%     RCWA_TE_Littrow_10000_overlap_LHS_dataset.csv, or select the CSV
%     when prompted.
%   - This script does NOT retrain the neural network.
%   - No convergence result is assumed in advance; inspect the generated
%     CSV summaries and the text report after running.
%   - The RCWA boundary matching below uses a numerically stable reference
%     for the backward modal amplitudes. This avoids explicitly forming
%     exp(-1i*k0*q*h), which can grow exponentially for evanescent modes.
%% ============================================================

clear; clc; close all;

%% ===================== 1. User settings =====================
THIS_DIR = fileparts(mfilename('fullpath'));

DATA_FILE = fullfile(THIS_DIR, ...
    'RCWA_TE_Littrow_10000_overlap_LHS_dataset.csv');

if ~isfile(DATA_FILE)
    [fileName, filePath] = uigetfile('*.csv', ...
        'Select RCWA_TE_Littrow_10000_overlap_LHS_dataset.csv');
    if isequal(fileName, 0)
        error(['Dataset CSV was not selected. Please place the CSV in ', ...
               'the script folder or select it when prompted.']);
    end
    DATA_FILE = fullfile(filePath, fileName);
end

OUT_DIR = fullfile(THIS_DIR, 'convergence_validation_results');
if ~exist(OUT_DIR, 'dir')
    mkdir(OUT_DIR);
end

% Optical/material parameters consistent with the original MATLAB code.
n_inc  = 1.0;
n_air  = 1.0;
n_sio2 = 1.44964;
n_sub  = n_sio2;

% RCWA truncation indices. N_harm = 15 is the setting used to generate
% the original dataset. The other settings test convergence.
N_HARM_LIST = [11, 15, 21, 31];

% SMM modal-root search resolutions. The original code uses 4000 points.
N_SCAN_LIST = [1000, 2000, 4000, 8000];

% Optional SMM overlap-integral grid convergence test.
N_OVERLAP_LIST = [401, 801, 1601];

% Default: use a deterministic representative subset spanning the
% Lambda/lambda, fill-factor, and h/lambda parameter ranges.
% Set RUN_ALL_VALID_SAMPLES = true to run all valid rows. That can be slow.
RUN_ALL_VALID_SAMPLES = true;
QUANTILE_LEVELS = [0.05, 0.25, 0.50, 0.75, 0.95];
N_RANDOM_EXTRA = 20;
RANDOM_SEED = 20261029;

% Suggested screening tolerances. These are criteria to test, not
% pre-established results.
ABS_TOL_EFFICIENCY = 1e-4;
ABS_TOL_NEFF = 1e-6;
ABS_TOL_OVERLAP = 1e-4;
ABS_TOL_ENERGY_BALANCE = 1e-3;

fprintf('\n============================================================\n');
fprintf('RCWA and SMM convergence validation\n');
fprintf('Dataset: %s\n', DATA_FILE);
fprintf('Output folder: %s\n', OUT_DIR);
fprintf('============================================================\n');

%% ===================== 2. Load and validate dataset =====================
T_all = readtable(DATA_FILE);

requiredVars = {'Lambda_m','f','h_m','lambda_m'};
for k = 1:numel(requiredVars)
    if ~ismember(requiredVars{k}, T_all.Properties.VariableNames)
        error('Required column "%s" was not found in the dataset.', ...
            requiredVars{k});
    end
end

validMask = isfinite(T_all.Lambda_m) & isfinite(T_all.f) & ...
            isfinite(T_all.h_m) & isfinite(T_all.lambda_m);

if ismember('status', T_all.Properties.VariableNames)
    validMask = validMask & strcmpi(string(T_all.status), "OK");
end

if ismember('theta_rad', T_all.Properties.VariableNames)
    % theta_rad may be NaN in failed rows; those rows are excluded above
    % only if the geometry is invalid. Angle is recomputed when needed.
else
    warning('theta_rad column not found; Littrow angles will be recomputed.');
end

originalCSVRows = find(validMask);
T_valid = T_all(validMask, :);

if isempty(T_valid)
    error('No valid rows were found in the dataset.');
end

fprintf('Valid dataset rows: %d / %d\n', height(T_valid), height(T_all));

%% ===================== 3. Select representative samples =====================
rng(RANDOM_SEED, 'twister');

if RUN_ALL_VALID_SAMPLES
    selected = (1:height(T_valid)).';
else
    X = [T_valid.Lambda_m ./ T_valid.lambda_m, ...
         T_valid.f, ...
         T_valid.h_m ./ T_valid.lambda_m];

    scaleX = max(X, [], 1) - min(X, [], 1);
    scaleX(scaleX < eps) = 1;

    selected = [];
    for ia = 1:numel(QUANTILE_LEVELS)
        for ib = 1:numel(QUANTILE_LEVELS)
            for ic = 1:numel(QUANTILE_LEVELS)
                target = [ ...
                    local_quantile(X(:,1), QUANTILE_LEVELS(ia)), ...
                    local_quantile(X(:,2), QUANTILE_LEVELS(ib)), ...
                    local_quantile(X(:,3), QUANTILE_LEVELS(ic))];

                distance = sum(((X - target) ./ scaleX).^2, 2);
                [~, nearestIdx] = min(distance);
                selected(end+1,1) = nearestIdx; %#ok<SAGROW>
            end
        end
    end

    selected = unique(selected, 'stable');

    remaining = setdiff((1:height(T_valid)).', selected, 'stable');
    nExtra = min(N_RANDOM_EXTRA, numel(remaining));
    if nExtra > 0
        randomPick = remaining(randperm(numel(remaining), nExtra));
        selected = unique([selected; randomPick(:)], 'stable');
    end
end

T_sel = T_valid(selected, :);
sourceRows = originalCSVRows(selected);
nSamples = height(T_sel);

T_sel_out = addvars(T_sel, sourceRows, 'Before', 1, ...
    'NewVariableNames', 'OriginalCSVRow');
writetable(T_sel_out, fullfile(OUT_DIR, 'selected_convergence_samples.csv'));

fprintf('Samples selected for convergence tests: %d\n', nSamples);
if RUN_ALL_VALID_SAMPLES
    fprintf('Sampling mode: all valid samples.\n');
else
    fprintf('Sampling mode: representative subset (quantile grid + random samples).\n');
    fprintf('Up to %d additional random samples are included.\n', N_RANDOM_EXTRA);
end

%% ===================== 4. RCWA Fourier-truncation convergence =====================
nH = numel(N_HARM_LIST);
eta0_rcwa = nan(nSamples, nH);
etaM1_rcwa = nan(nSamples, nH);
Rsum_rcwa = nan(nSamples, nH);
Tsum_rcwa = nan(nSamples, nH);
energy_rcwa = nan(nSamples, nH);
time_rcwa = nan(nSamples, nH);
status_rcwa = strings(nSamples, nH);

fprintf('\n--- RCWA convergence test ---\n');

for i = 1:nSamples
    Lambda = T_sel.Lambda_m(i);
    fillFactor = T_sel.f(i);
    depth = T_sel.h_m(i);
    lambda0 = T_sel.lambda_m(i);

    theta = get_littrow_angle(T_sel, i, lambda0, Lambda, n_inc);

    for j = 1:nH
        Nharm = N_HARM_LIST(j);
        t0 = tic;
        try
            out = rcwa_lamellar_TE_1layer_stable( ...
                Lambda, fillFactor, depth, lambda0, theta, ...
                n_inc, n_sub, n_sio2, n_air, Nharm);

            eta0_rcwa(i,j) = out.eta0;
            etaM1_rcwa(i,j) = out.eta_m1;
            Rsum_rcwa(i,j) = out.Rsum;
            Tsum_rcwa(i,j) = out.Tsum;
            energy_rcwa(i,j) = out.energy_sum;
            status_rcwa(i,j) = "OK";
        catch ME
            status_rcwa(i,j) = "ERROR: " + string(ME.message);
            warning('RCWA failed for sample %d, N_harm=%d: %s', ...
                i, Nharm, ME.message);
        end
        time_rcwa(i,j) = toc(t0);
    end

    if mod(i, 10) == 0 || i == nSamples
        fprintf('RCWA: completed %d / %d samples\n', i, nSamples);
    end
end

% Save per-sample RCWA results.
T_rcwa = T_sel_out;
for j = 1:nH
    Nharm = N_HARM_LIST(j);
    T_rcwa.(sprintf('eta0_N%d', Nharm)) = eta0_rcwa(:,j);
    T_rcwa.(sprintf('etaM1_N%d', Nharm)) = etaM1_rcwa(:,j);
    T_rcwa.(sprintf('Rsum_N%d', Nharm)) = Rsum_rcwa(:,j);
    T_rcwa.(sprintf('Tsum_N%d', Nharm)) = Tsum_rcwa(:,j);
    T_rcwa.(sprintf('energySum_N%d', Nharm)) = energy_rcwa(:,j);
    T_rcwa.(sprintf('time_s_N%d', Nharm)) = time_rcwa(:,j);
    T_rcwa.(sprintf('status_N%d', Nharm)) = status_rcwa(:,j);
end
writetable(T_rcwa, fullfile(OUT_DIR, 'RCWA_convergence_per_sample.csv'));

% Summarize each truncation setting relative to the largest tested order.
refH = nH;
meanDiffEta0 = nan(nH,1);
maxDiffEta0 = nan(nH,1);
meanDiffEtaM1 = nan(nH,1);
maxDiffEtaM1 = nan(nH,1);
fractionEta0WithinTol = nan(nH,1);
fractionEtaM1WithinTol = nan(nH,1);
meanEnergyError = nan(nH,1);
maxEnergyError = nan(nH,1);
nRCWAFailed = zeros(nH,1);
meanTime = nan(nH,1);

for j = 1:nH
    d0 = abs(eta0_rcwa(:,j) - eta0_rcwa(:,refH));
    dM1 = abs(etaM1_rcwa(:,j) - etaM1_rcwa(:,refH));
    energyError = abs(energy_rcwa(:,j) - 1);

    meanDiffEta0(j) = finite_mean(d0);
    maxDiffEta0(j) = finite_max(d0);
    meanDiffEtaM1(j) = finite_mean(dM1);
    maxDiffEtaM1(j) = finite_max(dM1);
    fractionEta0WithinTol(j) = fraction_below(d0, ABS_TOL_EFFICIENCY);
    fractionEtaM1WithinTol(j) = fraction_below(dM1, ABS_TOL_EFFICIENCY);
    meanEnergyError(j) = finite_mean(energyError);
    maxEnergyError(j) = finite_max(energyError);
    nRCWAFailed(j) = sum(~isfinite(eta0_rcwa(:,j)) | ...
                         ~isfinite(etaM1_rcwa(:,j)));
    meanTime(j) = finite_mean(time_rcwa(:,j));
end

RCWA_summary = table( ...
    N_HARM_LIST(:), (2*N_HARM_LIST(:)+1), ...
    meanDiffEta0, maxDiffEta0, meanDiffEtaM1, maxDiffEtaM1, ...
    fractionEta0WithinTol, fractionEtaM1WithinTol, ...
    meanEnergyError, maxEnergyError, nRCWAFailed, meanTime, ...
    'VariableNames', {'N_harm','NumberOfHarmonics', ...
    'MeanAbsDiff_eta0_vs_maxN','MaxAbsDiff_eta0_vs_maxN', ...
    'MeanAbsDiff_etaM1_vs_maxN','MaxAbsDiff_etaM1_vs_maxN', ...
    'Fraction_eta0_within_tolerance','Fraction_etaM1_within_tolerance', ...
    'MeanAbsEnergyBalanceError','MaxAbsEnergyBalanceError', ...
    'FailedSamples','MeanRuntime_s'});
writetable(RCWA_summary, fullfile(OUT_DIR, 'RCWA_convergence_summary.csv'));

% Explicitly assess current N_harm=15 and the two largest adjacent orders.
idx15 = find(N_HARM_LIST == 15, 1);
idxPrevHigh = nH - 1;
idxMax = nH;

if ~isempty(idx15)
    diff15_eta0 = abs(eta0_rcwa(:,idx15) - eta0_rcwa(:,idxMax));
    diff15_etaM1 = abs(etaM1_rcwa(:,idx15) - etaM1_rcwa(:,idxMax));
else
    diff15_eta0 = nan(nSamples,1);
    diff15_etaM1 = nan(nSamples,1);
end

diffHigh_eta0 = abs(eta0_rcwa(:,idxPrevHigh) - eta0_rcwa(:,idxMax));
diffHigh_etaM1 = abs(etaM1_rcwa(:,idxPrevHigh) - etaM1_rcwa(:,idxMax));

nFailHighPair = sum( ...
    ~isfinite(eta0_rcwa(:,idxPrevHigh)) | ~isfinite(etaM1_rcwa(:,idxPrevHigh)) | ...
    ~isfinite(eta0_rcwa(:,idxMax)) | ~isfinite(etaM1_rcwa(:,idxMax)));
rcwaHighPairPass = nFailHighPair == 0 && ...
    finite_max([diffHigh_eta0; diffHigh_etaM1]) < ABS_TOL_EFFICIENCY;

if ~isempty(idx15)
    nFailCurrent15 = sum( ...
        ~isfinite(eta0_rcwa(:,idx15)) | ~isfinite(etaM1_rcwa(:,idx15)) | ...
        ~isfinite(eta0_rcwa(:,idxMax)) | ~isfinite(etaM1_rcwa(:,idxMax)));
    rcwaCurrent15Pass = nFailCurrent15 == 0 && ...
        finite_max([diff15_eta0; diff15_etaM1]) < ABS_TOL_EFFICIENCY;
else
    nFailCurrent15 = nSamples;
    rcwaCurrent15Pass = false;
end

nFailMaxOrder = sum(~isfinite(energy_rcwa(:,idxMax)));
rcwaEnergyPass = nFailMaxOrder == 0 && ...
    finite_max(abs(energy_rcwa(:,idxMax)-1)) < ABS_TOL_ENERGY_BALANCE;

%% ===================== 5. Compare original RCWA labels with stable solver =====================
% This is a diagnostic for the existing training labels. A discrepancy does
% not automatically prove that the original labels are wrong, but a large
% difference should be investigated before claiming the dataset is validated.

if ~isempty(idx15) && ...
        ismember('eta0_RCWA_TE', T_sel.Properties.VariableNames) && ...
        ismember('etam1_RCWA_TE', T_sel.Properties.VariableNames)

    oldEta0 = T_sel.eta0_RCWA_TE;
    oldEtaM1 = T_sel.etam1_RCWA_TE;

    diffOldEta0 = abs(oldEta0 - eta0_rcwa(:,idx15));
    diffOldEtaM1 = abs(oldEtaM1 - etaM1_rcwa(:,idx15));

    T_compare = table(sourceRows, oldEta0, eta0_rcwa(:,idx15), ...
        diffOldEta0, oldEtaM1, etaM1_rcwa(:,idx15), diffOldEtaM1, ...
        'VariableNames', {'OriginalCSVRow','OriginalEta0_RCWA', ...
        'StableEta0_N15','AbsDiffEta0','OriginalEtaM1_RCWA', ...
        'StableEtaM1_N15','AbsDiffEtaM1'});
    writetable(T_compare, fullfile(OUT_DIR, ...
        'RCWA_original_labels_vs_stable_N15.csv'));

    originalLabelMeanDiffEta0 = finite_mean(diffOldEta0);
    originalLabelMaxDiffEta0 = finite_max(diffOldEta0);
    originalLabelMeanDiffEtaM1 = finite_mean(diffOldEtaM1);
    originalLabelMaxDiffEtaM1 = finite_max(diffOldEtaM1);
else
    originalLabelMeanDiffEta0 = NaN;
    originalLabelMaxDiffEta0 = NaN;
    originalLabelMeanDiffEtaM1 = NaN;
    originalLabelMaxDiffEtaM1 = NaN;
    warning(['Original RCWA efficiency columns were not found, so the ', ...
        'comparison with original labels was skipped.']);
end

%% ===================== 6. SMM modal-root convergence =====================
nS = numel(N_SCAN_LIST);
rootCount = nan(nSamples, nS);
neff0_smm = nan(nSamples, nS);
neff1_smm = nan(nSamples, nS);
eta0_smm = nan(nSamples, nS);
etaM1_smm = nan(nSamples, nS);
time_smm = nan(nSamples, nS);
status_smm = strings(nSamples, nS);
rootsText = strings(nSamples, nS);

fprintf('\n--- SMM modal-root convergence test ---\n');

for i = 1:nSamples
    Lambda = T_sel.Lambda_m(i);
    fillFactor = T_sel.f(i);
    depth = T_sel.h_m(i);
    lambda0 = T_sel.lambda_m(i);
    theta = get_littrow_angle(T_sel, i, lambda0, Lambda, n_inc);

    for j = 1:nS
        Nscan = N_SCAN_LIST(j);
        t0 = tic;
        try
            out = smm_littrow_TE_two_modes( ...
                Lambda, fillFactor, depth, lambda0, theta, ...
                n_air, n_sio2, Nscan);

            rootCount(i,j) = out.n_roots;
            rootsText(i,j) = mat2str(out.roots(:).', 8);

            if out.n_roots >= 2
                neff0_smm(i,j) = out.neff0;
                neff1_smm(i,j) = out.neff1;
                eta0_smm(i,j) = out.eta0_SMM;
                etaM1_smm(i,j) = out.eta_m1_SMM;
                status_smm(i,j) = "OK";
            else
                status_smm(i,j) = "FEWER_THAN_TWO_ROOTS";
            end
        catch ME
            status_smm(i,j) = "ERROR: " + string(ME.message);
            warning('SMM failed for sample %d, N_scan=%d: %s', ...
                i, Nscan, ME.message);
        end
        time_smm(i,j) = toc(t0);
    end

    if mod(i, 10) == 0 || i == nSamples
        fprintf('SMM root search: completed %d / %d samples\n', i, nSamples);
    end
end

T_smm = T_sel_out;
for j = 1:nS
    Nscan = N_SCAN_LIST(j);
    T_smm.(sprintf('rootCount_N%d', Nscan)) = rootCount(:,j);
    T_smm.(sprintf('neff0_N%d', Nscan)) = neff0_smm(:,j);
    T_smm.(sprintf('neff1_N%d', Nscan)) = neff1_smm(:,j);
    T_smm.(sprintf('eta0SMM_N%d', Nscan)) = eta0_smm(:,j);
    T_smm.(sprintf('etaM1SMM_N%d', Nscan)) = etaM1_smm(:,j);
    T_smm.(sprintf('roots_N%d', Nscan)) = rootsText(:,j);
    T_smm.(sprintf('time_s_N%d', Nscan)) = time_smm(:,j);
    T_smm.(sprintf('status_N%d', Nscan)) = status_smm(:,j);
end
writetable(T_smm, fullfile(OUT_DIR, 'SMM_root_convergence_per_sample.csv'));

refS = nS;
meanDiffNeff0 = nan(nS,1);
maxDiffNeff0 = nan(nS,1);
meanDiffNeff1 = nan(nS,1);
maxDiffNeff1 = nan(nS,1);
meanDiffSMMEta0 = nan(nS,1);
maxDiffSMMEta0 = nan(nS,1);
meanDiffSMMEtaM1 = nan(nS,1);
maxDiffSMMEtaM1 = nan(nS,1);
rootCountMismatch = zeros(nS,1);
fractionRootsSame = nan(nS,1);
nSMMFailed = zeros(nS,1);
meanSMMTime = nan(nS,1);

for j = 1:nS
    dN0 = abs(neff0_smm(:,j) - neff0_smm(:,refS));
    dN1 = abs(neff1_smm(:,j) - neff1_smm(:,refS));
    dE0 = abs(eta0_smm(:,j) - eta0_smm(:,refS));
    dEM1 = abs(etaM1_smm(:,j) - etaM1_smm(:,refS));

    meanDiffNeff0(j) = finite_mean(dN0);
    maxDiffNeff0(j) = finite_max(dN0);
    meanDiffNeff1(j) = finite_mean(dN1);
    maxDiffNeff1(j) = finite_max(dN1);
    meanDiffSMMEta0(j) = finite_mean(dE0);
    maxDiffSMMEta0(j) = finite_max(dE0);
    meanDiffSMMEtaM1(j) = finite_mean(dEM1);
    maxDiffSMMEtaM1(j) = finite_max(dEM1);

    bothCountsFinite = isfinite(rootCount(:,j)) & isfinite(rootCount(:,refS));
    if any(bothCountsFinite)
        rootCountMismatch(j) = sum(rootCount(bothCountsFinite,j) ~= ...
            rootCount(bothCountsFinite,refS));
        fractionRootsSame(j) = mean(rootCount(bothCountsFinite,j) == ...
            rootCount(bothCountsFinite,refS));
    else
        rootCountMismatch(j) = nSamples;
        fractionRootsSame(j) = NaN;
    end

    nSMMFailed(j) = sum(~isfinite(eta0_smm(:,j)) | ~isfinite(etaM1_smm(:,j)));
    meanSMMTime(j) = finite_mean(time_smm(:,j));
end

SMM_summary = table( ...
    N_SCAN_LIST(:), meanDiffNeff0, maxDiffNeff0, ...
    meanDiffNeff1, maxDiffNeff1, ...
    meanDiffSMMEta0, maxDiffSMMEta0, ...
    meanDiffSMMEtaM1, maxDiffSMMEtaM1, ...
    rootCountMismatch, fractionRootsSame, nSMMFailed, meanSMMTime, ...
    'VariableNames', {'N_scan','MeanAbsDiff_neff0_vs_maxN', ...
    'MaxAbsDiff_neff0_vs_maxN','MeanAbsDiff_neff1_vs_maxN', ...
    'MaxAbsDiff_neff1_vs_maxN','MeanAbsDiff_eta0_vs_maxN', ...
    'MaxAbsDiff_eta0_vs_maxN','MeanAbsDiff_etaM1_vs_maxN', ...
    'MaxAbsDiff_etaM1_vs_maxN','RootCountMismatch_vs_maxN', ...
    'FractionRootCountsSame','FailedSamples','MeanRuntime_s'});
writetable(SMM_summary, fullfile(OUT_DIR, 'SMM_convergence_summary.csv'));

idx4000 = find(N_SCAN_LIST == 4000, 1);
idx8000 = find(N_SCAN_LIST == 8000, 1);
if isempty(idx4000), idx4000 = nS-1; end
if isempty(idx8000), idx8000 = nS; end

smmRootCountPass = rootCountMismatch(idx4000) == 0 && ...
    all(isfinite(rootCount(:,idx4000))) && all(isfinite(rootCount(:,idx8000))) && ...
    nSMMFailed(idx4000) == 0 && nSMMFailed(idx8000) == 0;

neffPairFailed = any(~isfinite(neff0_smm(:,idx4000)) | ...
    ~isfinite(neff1_smm(:,idx4000)) | ~isfinite(neff0_smm(:,idx8000)) | ...
    ~isfinite(neff1_smm(:,idx8000)));
smmNeffPass = ~neffPairFailed && finite_max([ ...
    abs(neff0_smm(:,idx4000)-neff0_smm(:,idx8000)); ...
    abs(neff1_smm(:,idx4000)-neff1_smm(:,idx8000))]) < ABS_TOL_NEFF;

effPairFailed = any(~isfinite(eta0_smm(:,idx4000)) | ...
    ~isfinite(etaM1_smm(:,idx4000)) | ~isfinite(eta0_smm(:,idx8000)) | ...
    ~isfinite(etaM1_smm(:,idx8000)));
smmEfficiencyPass = ~effPairFailed && finite_max([ ...
    abs(eta0_smm(:,idx4000)-eta0_smm(:,idx8000)); ...
    abs(etaM1_smm(:,idx4000)-etaM1_smm(:,idx8000))]) < ABS_TOL_EFFICIENCY;

%% ===================== 7. Optional SMM overlap-integral convergence =====================
% The overlap integrals are auxiliary quantities in the original dataset;
% they are not direct inputs to the current Transformer model. This section
% tests numerical integration stability using roots from the finest N_scan.

nO = numel(N_OVERLAP_LIST);
nOverlapMetrics = 6;
overlapValues = nan(nSamples, nO, nOverlapMetrics);
overlapStatus = strings(nSamples, nO);

overlapMetricNames = { ...
    'mode0_order0','mode0_order_m1','mode0_order_p1', ...
    'mode1_order0','mode1_order_m1','mode1_order_p1'};

fprintf('\n--- SMM overlap-integral convergence test ---\n');

for i = 1:nSamples
    if ~isfinite(neff0_smm(i,refS)) || ~isfinite(neff1_smm(i,refS))
        overlapStatus(i,:) = "SKIPPED_NO_TWO_ROOTS";
        continue;
    end

    Lambda = T_sel.Lambda_m(i);
    fillFactor = T_sel.f(i);
    depth = T_sel.h_m(i); %#ok<NASGU>
    lambda0 = T_sel.lambda_m(i);
    theta = get_littrow_angle(T_sel, i, lambda0, Lambda, n_inc);

    for j = 1:nO
        Ngrid = N_OVERLAP_LIST(j);
        try
            ov0 = compute_TE_mode_order_overlaps( ...
                neff0_smm(i,refS), Lambda, fillFactor, lambda0, theta, ...
                n_air, n_sio2, Ngrid);
            ov1 = compute_TE_mode_order_overlaps( ...
                neff1_smm(i,refS), Lambda, fillFactor, lambda0, theta, ...
                n_air, n_sio2, Ngrid);

            overlapValues(i,j,:) = reshape([ ...
                ov0.order0, ov0.orderm1, ov0.order1, ...
                ov1.order0, ov1.orderm1, ov1.order1], 1, 1, nOverlapMetrics);
            overlapStatus(i,j) = "OK";
        catch ME
            overlapStatus(i,j) = "ERROR: " + string(ME.message);
            warning('Overlap integral failed for sample %d, N_overlap=%d: %s', ...
                i, Ngrid, ME.message);
        end
    end
end

T_overlap = T_sel_out;
for j = 1:nO
    Ngrid = N_OVERLAP_LIST(j);
    for k = 1:nOverlapMetrics
        varName = sprintf('%s_N%d', overlapMetricNames{k}, Ngrid);
        T_overlap.(varName) = overlapValues(:,j,k);
    end
    T_overlap.(sprintf('status_N%d', Ngrid)) = overlapStatus(:,j);
end
writetable(T_overlap, fullfile(OUT_DIR, 'SMM_overlap_convergence_per_sample.csv'));

refO = nO;
overlapSummaryRows = cell(nO*nOverlapMetrics, 5);
summaryRow = 0;
for j = 1:nO
    for k = 1:nOverlapMetrics
        summaryRow = summaryRow + 1;
        d = abs(overlapValues(:,j,k) - overlapValues(:,refO,k));
        overlapSummaryRows(summaryRow,:) = { ...
            N_OVERLAP_LIST(j), overlapMetricNames{k}, ...
            finite_mean(d), finite_max(d), ...
            fraction_below(d, ABS_TOL_OVERLAP)};
    end
end
Overlap_summary = cell2table(overlapSummaryRows, ...
    'VariableNames', {'N_overlap','OverlapMetric', ...
    'MeanAbsDiff_vs_maxGrid','MaxAbsDiff_vs_maxGrid', ...
    'FractionWithinTolerance'});
writetable(Overlap_summary, fullfile(OUT_DIR, 'SMM_overlap_convergence_summary.csv'));

%% ===================== 8. Save figures =====================
try
    % RCWA max absolute differences from the highest tested truncation.
    fig = figure('Visible','off','Color','w');
    hold on;
    plot(N_HARM_LIST, max(maxDiffEta0, eps), 'o-', 'LineWidth', 1.5);
    plot(N_HARM_LIST, max(maxDiffEtaM1, eps), 's-', 'LineWidth', 1.5);
    set(gca, 'YScale', 'log');
    xlabel('Fourier truncation index N_{harm}');
    ylabel('Maximum absolute efficiency difference vs. largest N');
    legend('\eta_0','\eta_{-1}','Location','best');
    grid on; box on;
    title('RCWA Fourier-truncation convergence');
    saveas(fig, fullfile(OUT_DIR, 'RCWA_convergence.png'));
    close(fig);

    % SMM efficiency differences from the finest root-search grid.
    fig = figure('Visible','off','Color','w');
    hold on;
    plot(N_SCAN_LIST, max(maxDiffSMMEta0, eps), 'o-', 'LineWidth', 1.5);
    plot(N_SCAN_LIST, max(maxDiffSMMEtaM1, eps), 's-', 'LineWidth', 1.5);
    set(gca, 'YScale', 'log');
    xlabel('SMM root-search grid points');
    ylabel('Maximum absolute efficiency difference vs. largest grid');
    legend('\eta_0^{SMM}','\eta_{-1}^{SMM}','Location','best');
    grid on; box on;
    title('SMM root-search convergence');
    saveas(fig, fullfile(OUT_DIR, 'SMM_convergence.png'));
    close(fig);
catch ME
    warning('Plot generation failed: %s', ME.message);
end

%% ===================== 9. Write human-readable report =====================
reportFile = fullfile(OUT_DIR, 'convergence_validation_report.txt');
fid = fopen(reportFile, 'w');

fprintf(fid, 'RCWA AND SMM CONVERGENCE VALIDATION REPORT\n');
fprintf(fid, '==========================================\n\n');
fprintf(fid, 'Dataset: %s\n', DATA_FILE);
fprintf(fid, 'Selected samples: %d\n', nSamples);
fprintf(fid, 'RCWA N_harm list: %s\n', mat2str(N_HARM_LIST));
fprintf(fid, 'SMM N_scan list: %s\n', mat2str(N_SCAN_LIST));
fprintf(fid, 'SMM N_overlap list: %s\n\n', mat2str(N_OVERLAP_LIST));

fprintf(fid, 'Suggested screening criteria (must be checked against results):\n');
fprintf(fid, 'Efficiency absolute difference tolerance: %.3g\n', ABS_TOL_EFFICIENCY);
fprintf(fid, 'Effective-index absolute difference tolerance: %.3g\n', ABS_TOL_NEFF);
fprintf(fid, 'Overlap absolute difference tolerance: %.3g\n', ABS_TOL_OVERLAP);
fprintf(fid, 'Energy-balance absolute error tolerance: %.3g\n\n', ABS_TOL_ENERGY_BALANCE);

fprintf(fid, 'RCWA:\n');
fprintf(fid, '  Highest adjacent pair: N_harm=%d vs %d\n', ...
    N_HARM_LIST(idxPrevHigh), N_HARM_LIST(idxMax));
fprintf(fid, '  Max abs difference eta0 (adjacent pair): %.8g\n', ...
    finite_max(diffHigh_eta0));
fprintf(fid, '  Max abs difference eta-1 (adjacent pair): %.8g\n', ...
    finite_max(diffHigh_etaM1));
fprintf(fid, '  Highest-pair efficiency criterion passed: %s\n', ...
    pass_text(rcwaHighPairPass));
fprintf(fid, '  Max energy-balance error at N_harm=%d: %.8g\n', ...
    N_HARM_LIST(idxMax), finite_max(abs(energy_rcwa(:,idxMax)-1)));
fprintf(fid, '  Energy-balance criterion passed at highest order: %s\n', ...
    pass_text(rcwaEnergyPass));
if ~isempty(idx15)
    fprintf(fid, '  Current N_harm=15 vs N_harm=%d max abs difference eta0: %.8g\n', ...
        N_HARM_LIST(idxMax), finite_max(diff15_eta0));
    fprintf(fid, '  Current N_harm=15 vs N_harm=%d max abs difference eta-1: %.8g\n', ...
        N_HARM_LIST(idxMax), finite_max(diff15_etaM1));
    fprintf(fid, '  Current N_harm=15 within efficiency tolerance: %s\n', ...
        pass_text(rcwaCurrent15Pass));
end
fprintf(fid, '  Original labels vs stable N_harm=15: mean/max abs difference eta0 = %.8g / %.8g\n', ...
    originalLabelMeanDiffEta0, originalLabelMaxDiffEta0);
fprintf(fid, '  Original labels vs stable N_harm=15: mean/max abs difference eta-1 = %.8g / %.8g\n\n', ...
    originalLabelMeanDiffEtaM1, originalLabelMaxDiffEtaM1);

fprintf(fid, 'SMM:\n');
fprintf(fid, '  Compared root grids: %d vs %d points\n', ...
    N_SCAN_LIST(idx4000), N_SCAN_LIST(idx8000));
fprintf(fid, '  Root-count mismatch samples: %d\n', rootCountMismatch(idx4000));
fprintf(fid, '  Max abs difference neff0: %.8g\n', ...
    finite_max(abs(neff0_smm(:,idx4000)-neff0_smm(:,idx8000))));
fprintf(fid, '  Max abs difference neff1: %.8g\n', ...
    finite_max(abs(neff1_smm(:,idx4000)-neff1_smm(:,idx8000))));
fprintf(fid, '  Max abs difference eta0: %.8g\n', ...
    finite_max(abs(eta0_smm(:,idx4000)-eta0_smm(:,idx8000))));
fprintf(fid, '  Max abs difference eta-1: %.8g\n', ...
    finite_max(abs(etaM1_smm(:,idx4000)-etaM1_smm(:,idx8000))));
fprintf(fid, '  Root-count criterion passed: %s\n', pass_text(smmRootCountPass));
fprintf(fid, '  Effective-index criterion passed: %s\n', pass_text(smmNeffPass));
fprintf(fid, '  Efficiency criterion passed: %s\n\n', pass_text(smmEfficiencyPass));

fprintf(fid, 'Interpretation:\n');
fprintf(fid, ['These are results from the selected samples and the tested numerical ', ...
    'settings only. If any criterion fails, increase the corresponding resolution/order, ', ...
    'inspect failed samples, and repeat the test before claiming convergence.\n']);
fprintf(fid, ['A discrepancy between the original RCWA labels and the stable N_harm=15 ', ...
    'results should be investigated. If the original labels are materially affected, ', ...
    'regenerate the RCWA dataset and retrain/re-evaluate the ML model.\n']);
fclose(fid);

%% ===================== 10. Print key results =====================
fprintf('\n================ SUMMARY ================\n');
fprintf('RCWA highest pair N=%d vs N=%d: max |delta eta0|=%.6g, max |delta eta-1|=%.6g\n', ...
    N_HARM_LIST(idxPrevHigh), N_HARM_LIST(idxMax), ...
    finite_max(diffHigh_eta0), finite_max(diffHigh_etaM1));
fprintf('RCWA highest-pair efficiency criterion: %s\n', pass_text(rcwaHighPairPass));
fprintf('RCWA energy-balance criterion at N=%d: %s (max error %.6g)\n', ...
    N_HARM_LIST(idxMax), pass_text(rcwaEnergyPass), ...
    finite_max(abs(energy_rcwa(:,idxMax)-1)));
if ~isempty(idx15)
    fprintf('Current N=15 vs N=%d: max |delta eta0|=%.6g, max |delta eta-1|=%.6g\n', ...
        N_HARM_LIST(idxMax), finite_max(diff15_eta0), finite_max(diff15_etaM1));
    fprintf('Current N=15 within efficiency tolerance: %s\n', ...
        pass_text(rcwaCurrent15Pass));
end
fprintf('SMM N_scan=%d vs %d: max |delta neff0|=%.6g, max |delta neff1|=%.6g\n', ...
    N_SCAN_LIST(idx4000), N_SCAN_LIST(idx8000), ...
    finite_max(abs(neff0_smm(:,idx4000)-neff0_smm(:,idx8000))), ...
    finite_max(abs(neff1_smm(:,idx4000)-neff1_smm(:,idx8000))));
fprintf('SMM root-count / neff / efficiency criteria: %s / %s / %s\n', ...
    pass_text(smmRootCountPass), pass_text(smmNeffPass), pass_text(smmEfficiencyPass));
fprintf('Results saved to: %s\n', OUT_DIR);
fprintf('=========================================\n');

%% ===================== Local helper functions =====================

function theta = get_littrow_angle(T, i, lambda0, Lambda, n_inc)
    theta = NaN;
    if ismember('theta_rad', T.Properties.VariableNames)
        theta = T.theta_rad(i);
    end
    if ~isfinite(theta)
        arg = lambda0 / (2*n_inc*Lambda);
        if abs(arg) > 1
            error('Littrow condition is not satisfied for sample row %d.', i);
        end
        theta = asin(arg);
    end
end

function q = local_quantile(x, p)
    % Toolbox-independent linearly interpolated quantile.
    x = sort(x(isfinite(x)));
    if isempty(x)
        q = NaN;
        return;
    end
    if numel(x) == 1
        q = x(1);
        return;
    end
    pos = 1 + (numel(x)-1)*p;
    lo = floor(pos);
    hi = ceil(pos);
    q = x(lo) + (pos-lo)*(x(hi)-x(lo));
end

function m = finite_mean(x)
    x = x(isfinite(x));
    if isempty(x)
        m = NaN;
    else
        m = mean(x);
    end
end

function m = finite_max(x)
    x = x(isfinite(x));
    if isempty(x)
        m = NaN;
    else
        m = max(x);
    end
end

function f = fraction_below(x, tol)
    x = x(isfinite(x));
    if isempty(x)
        f = NaN;
    else
        f = sum(x < tol) / numel(x);
    end
end

function s = pass_text(tf)
    if isscalar(tf) && isfinite(double(tf)) && tf
        s = 'PASS';
    else
        s = 'CHECK / FAIL';
    end
end

function out = rcwa_lamellar_TE_1layer_stable( ...
    Lambda, f, h, lambda0, theta, n_inc, n_sub, n_ridge, n_groove, N_harm)
    % Numerically stable boundary matching for a single grating layer.
    % The backward modal amplitude is referenced to z=h. Therefore the
    % layer propagation factor X=exp(i*k0*q*h) is used without explicitly
    % constructing the exponentially growing exp(-i*k0*q*h) factor.

    k0 = 2*pi/lambda0;
    orders = (-N_harm:N_harm).';
    Nh = numel(orders);

    Kx = n_inc*sin(theta) + orders*lambda0/Lambda;
    q_inc = physical_sqrt(n_inc^2 - Kx.^2);
    q_sub = physical_sqrt(n_sub^2 - Kx.^2);
    q0_inc = n_inc*cos(theta);

    eps_r = n_ridge^2;
    eps_g = n_groove^2;
    deps = eps_r - eps_g;

    p_all = (-2*N_harm:2*N_harm).';
    eps_fourier = zeros(numel(p_all),1);
    for idx = 1:numel(p_all)
        p = p_all(idx);
        if p == 0
            eps_fourier(idx) = eps_g + deps*f;
        else
            eps_fourier(idx) = deps*sin(pi*p*f)/(pi*p);
        end
    end

    E = zeros(Nh,Nh);
    for row = 1:Nh
        for col = 1:Nh
            p = orders(row)-orders(col);
            idx = p + 2*N_harm + 1;
            E(row,col) = eps_fourier(idx);
        end
    end

    A = E - diag(Kx.^2);
    [W,D] = eig(A);
    q_layer = physical_sqrt(diag(D));
    Q = diag(q_layer);

    % This factor is decaying for evanescent modes on the selected branch.
    X = diag(exp(1i*k0*q_layer*h));

    Ivec = zeros(Nh,1);
    idx0 = find(orders == 0, 1);
    Ivec(idx0) = 1;

    Qinc = diag(q_inc);
    Qsub = diag(q_sub);
    Z = zeros(Nh,Nh);
    Id = eye(Nh);

    % Unknowns: reflected amplitudes R, forward amplitudes a referenced
    % at z=0, backward amplitudes b referenced at z=h, transmitted T.
    %
    % z=0:
    %   I + R = W(a + X*b)
    %   Qinc(I - R) = WQ(a - X*b)
    % z=h:
    %   W(X*a + b) = T
    %   WQ(X*a - b) = Qsub*T
    M = [ ...
        Id,     -W,       -W*X,  Z;
       -Qinc,   -W*Q,      W*Q*X, Z;
        Z,       W*X,       W,   -Id;
        Z,       W*Q*X,    -W*Q, -Qsub];

    rhs = [-Ivec; -Qinc*Ivec; zeros(Nh,1); zeros(Nh,1)];
    sol = M \ rhs;

    R = sol(1:Nh);
    T = sol(3*Nh+1:4*Nh);

    DE_R = zeros(Nh,1);
    DE_T = zeros(Nh,1);
    for m = 1:Nh
        if real(q_inc(m)) > 1e-12
            DE_R(m) = real(q_inc(m)/q0_inc)*abs(R(m))^2;
        end
        if real(q_sub(m)) > 1e-12
            DE_T(m) = real(q_sub(m)/q0_inc)*abs(T(m))^2;
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

function out = smm_littrow_TE_two_modes( ...
    Lambda, f, h, lambda0, theta, n1, n2, N_scan)
    k0 = 2*pi/lambda0;
    alpha = k0*n1*sin(theta);
    target = cos(alpha*Lambda);

    roots_neff = find_te_modal_roots( ...
        Lambda, f, lambda0, n1, n2, target, N_scan);
    roots_neff = sort(roots_neff, 'descend');

    out.roots = roots_neff;
    out.n_roots = numel(roots_neff);
    out.neff0 = NaN;
    out.neff1 = NaN;
    out.phi0 = NaN;
    out.phi1 = NaN;
    out.delta_phi = NaN;
    out.eta0_SMM = NaN;
    out.eta_m1_SMM = NaN;

    if numel(roots_neff) >= 2
        neff0 = roots_neff(1);
        neff1 = roots_neff(2);
        phi0 = k0*neff0*h;
        phi1 = k0*neff1*h;
        delta_phi = phi0-phi1;

        out.neff0 = neff0;
        out.neff1 = neff1;
        out.phi0 = phi0;
        out.phi1 = phi1;
        out.delta_phi = delta_phi;
        out.eta0_SMM = cos(delta_phi/2)^2;
        out.eta_m1_SMM = sin(delta_phi/2)^2;
    end
end

function roots_neff = find_te_modal_roots( ...
    Lambda, f, lambda0, n1, n2, target, N_scan)
    n_min = 1e-6;
    n_max = n2 - 1e-6;
    neff_grid = linspace(n_min, n_max, N_scan);
    y = nan(size(neff_grid));

    for i = 1:N_scan
        val = te_dispersion_F(neff_grid(i), Lambda, f, ...
            lambda0, n1, n2) - target;
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
            roots_neff(end+1,1) = neff_grid(i); %#ok<AGROW>
        elseif y1*y2 < 0
            a = neff_grid(i);
            b = neff_grid(i+1);
            try
                rootCandidate = fzero(@(ne) real(te_dispersion_F( ...
                    ne, Lambda, f, lambda0, n1, n2)-target), [a,b]);
                if rootCandidate > n_min && rootCandidate < n_max
                    roots_neff(end+1,1) = rootCandidate; %#ok<AGROW>
                end
            catch
                % Ignore intervals where fzero cannot refine a root.
            end
        end
    end

    if ~isempty(roots_neff)
        roots_neff = sort(roots_neff(:));
        rootsUnique = roots_neff(1);
        for i = 2:numel(roots_neff)
            if abs(roots_neff(i)-rootsUnique(end)) > 1e-5
                rootsUnique(end+1,1) = roots_neff(i); %#ok<AGROW>
            end
        end
        roots_neff = rootsUnique;
    end
end

function F = te_dispersion_F(neff, Lambda, f, lambda0, n1, n2)
    k0 = 2*pi/lambda0;
    k1 = k0*sqrt(complex(n1^2-neff^2));
    k2 = k0*sqrt(complex(n2^2-neff^2));
    d1 = (1-f)*Lambda;
    d2 = f*Lambda;

    F = cos(k1*d1)*cos(k2*d2) ...
        - ((k1^2+k2^2)/(2*k1*k2))*sin(k1*d1)*sin(k2*d2);
end

function overlap = compute_TE_mode_order_overlaps( ...
    neff, Lambda, f, lambda0, theta, n1, n2, N_grid)
    overlap.order0 = overlap_single_TE( ...
        neff, Lambda, f, lambda0, theta, n1, n2, 0, N_grid);
    overlap.orderm1 = overlap_single_TE( ...
        neff, Lambda, f, lambda0, theta, n1, n2, -1, N_grid);
    overlap.order1 = overlap_single_TE( ...
        neff, Lambda, f, lambda0, theta, n1, n2, 1, N_grid);
end

function ov = overlap_single_TE( ...
    neff, Lambda, f, lambda0, theta, n1, n2, p_order, N_grid)
    k0 = 2*pi/lambda0;
    alpha = k0*n1*sin(theta);

    d1 = (1-f)*Lambda;
    d2 = f*Lambda;

    k1 = k0*sqrt(complex(n1^2-neff^2));
    k2 = k0*sqrt(complex(n2^2-neff^2));

    e1p = exp(1i*k1*d1);
    e1m = exp(-1i*k1*d1);
    e2p = exp(1i*k2*d2);
    e2m = exp(-1i*k2*d2);
    P = exp(1i*alpha*Lambda);

    M = [ ...
         e1p,       e1m,      -1,       -1;
         k1*e1p,   -k1*e1m,   -k2,       k2;
        -P,         -P,        e2p,      e2m;
        -P*k1,       P*k1,     k2*e2p,  -k2*e2m];

    [~,~,V] = svd(M);
    c = V(:,end);
    A = c(1); B = c(2); C = c(3); D = c(4);

    x = linspace(0, Lambda, N_grid).';
    u = zeros(size(x));

    idx1 = x <= d1;
    u(idx1) = A*exp(1i*k1*x(idx1)) + B*exp(-1i*k1*x(idx1));

    x2 = x(~idx1)-d1;
    u(~idx1) = C*exp(1i*k2*x2) + D*exp(-1i*k2*x2);

    kxp = k0*(n1*sin(theta) + p_order*lambda0/Lambda);
    Ep = exp(-1i*kxp*x);

    numerator = abs(trapz(x, conj(Ep).*u))^2;
    denominator = trapz(x, abs(Ep).^2)*trapz(x, abs(u).^2);

    ov = real(numerator/denominator);
    ov = max(0, min(1, ov));
end

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

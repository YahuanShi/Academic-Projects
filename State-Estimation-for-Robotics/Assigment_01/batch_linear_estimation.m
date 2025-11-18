%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% batch estimation for δ = [1000, 100, 10, 1]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc;
load dataset1.mat;

T = 0.1;              % sampling period
sigma_q = v_var;      
sigma_r = r_var;      
y = l - r;            
K = length(t);        

deltas = [1000, 100, 10, 1];   
num_d = length(deltas);

% For storing results
Results = struct();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop: run estimation for each δ
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for idx = 1:num_d
    delta = deltas(idx);

    % Number of segments
    numSeg = floor(K/delta);
    if mod(K, delta) == 0
        numSeg = numSeg - 1;
    end

    % Build reduced odometry + measurement
    sub_u = zeros(numSeg+1,1);
    sub_y = zeros(numSeg+1,1);

    for k = 1:numSeg
        s = (k-1)*delta + 1;
        e = k*delta;
        sub_u(k) = sum(T * v(s:e));
        sub_y(k) = y(e);
    end

    s = numSeg*delta + 1;
    sub_u(numSeg+1) = sum(T * v(s:K));
    sub_y(numSeg+1) = y(K);

    % Build z
    N = numSeg + 2;
    z = [1; sub_u; 1; sub_y];

    % Build H
    H = sparse(2*N, N);
    H(1:N, :) = speye(N);
    H(N+1:end, :) = speye(N);

    for i = 2:N
        H(i, i-1) = -1; 
    end

    % Build W
    W = sparse(2*N, 2*N);
    W(1:N,1:N) = sigma_q * speye(N);
    W(N+1:end, N+1:end) = sigma_r * speye(N);

    W(1,1) = 1e-3;
    W(N+1,N+1) = 1e-3;

    % Solve batch (Cholesky only)
    W_inv = inv(W);
    A = H' * W_inv * H;
    L = chol(A, 'lower');
    d = L \ (H' * W_inv * z);
    x_est = L' \ d;
    P = inv(A);

    % Ground truth sampling
    sub_t = zeros(N,1);
    sub_true = zeros(N,1);

    for i = 1:N-1
        sub_t(i) = t((i-1)*delta + 1);
        sub_true(i) = x_true((i-1)*delta + 1);
    end
    sub_t(N) = t(K);
    sub_true(N) = x_true(K);

    Error = x_est - sub_true;
    sigma_vec = sqrt(diag(P));

    Results(idx).delta = delta;
    Results(idx).t = sub_t;
    Results(idx).est = x_est;
    Results(idx).true = sub_true;
    Results(idx).err = Error;
    Results(idx).sigma = sigma_vec;
end


output_folder = 'Batch_Figures';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ERRORS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig1 = figure('Name','Error plots');
for idx = 1:num_d
    subplot(2,2,idx);
    plot(Results(idx).t, Results(idx).err, '.-');
    title(['Error (δ = ', num2str(deltas(idx)), ')']);
    xlabel('t (s)'); ylabel('Error (m)');
    grid on;
end

saveas(fig1, fullfile(output_folder, 'Errors_All.png'));
saveas(fig1, fullfile(output_folder, 'Errors_All.fig'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TRAJECTORY
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig2 = figure('Name','Trajectories');
for idx = 1:num_d
    subplot(2,2,idx);
    plot(Results(idx).t, Results(idx).est,'.-'); hold on;
    plot(Results(idx).t, Results(idx).true,'-');
    legend('estimate','true');
    title(['Trajectory (δ = ', num2str(deltas(idx)), ')']);
    xlabel('t (s)'); ylabel('x (m)');
    grid on;
end

saveas(fig2, fullfile(output_folder, 'Trajectories_All.png'));
saveas(fig2, fullfile(output_folder, 'Trajectories_All.fig'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% UNCERTAINTY ENVELOPES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig3 = figure('Name','Uncertainty Envelopes');
for idx = 1:num_d
    subplot(2,2,idx);
    plot(Results(idx).t, 3*Results(idx).sigma,'r.'); hold on;
    plot(Results(idx).t, -3*Results(idx).sigma,'r.');
    plot(Results(idx).t, Results(idx).err,'.-');
    title(['±3σ Envelope (δ = ', num2str(deltas(idx)), ')']);
    xlabel('t (s)'); ylabel('±3σ');
    grid on;
end

saveas(fig3, fullfile(output_folder, 'Uncertainty_All.png'));
saveas(fig3, fullfile(output_folder, 'Uncertainty_All.fig'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% HISTOGRAMS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig4 = figure('Name','Error Histograms');
for idx = 1:num_d
    subplot(2,2,idx);
    histogram(Results(idx).err,'Normalization','pdf'); hold on;
    pd = fitdist(Results(idx).err,'Normal');
    x_vals = linspace(min(Results(idx).err), max(Results(idx).err), 200);
    plot(x_vals, pdf(pd,x_vals),'r','LineWidth',2);
    title(['Histogram (δ = ', num2str(deltas(idx)), ')']);
    xlabel('Error'); ylabel('PDF');
    grid on;
end

saveas(fig4, fullfile(output_folder, 'Histograms_All.png'));
saveas(fig4, fullfile(output_folder, 'Histograms_All.fig'));

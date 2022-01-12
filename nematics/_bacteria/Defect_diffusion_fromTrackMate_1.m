%% IMPORT FILES
% [filename, pathname] = uigetfile('*.*',...
%     'C:\Users\USER\Downloads\B-sub-sur-minus-in-supernatant-40X-100fps\TrackMate');
% % %%
% ALL = readtable([pathname,filename]);
ALL = readtable('C:\Users\USER\Downloads\B-sub-sur-minus-in-supernatant-40X-100fps\TrackMate\spots_n.csv');
%%
% close all
% clear all

SPACE_UNITS = 'µm';
TIME_UNITS = 's';

N_PARTICLES = 20;
N_TIME_STEPS = 200;

% Diffusion coefficient. Will set the amplitude of the random displacement
D  = 1e-3; % µm^2/s
% Time step between acquisition; fast acquisition!
dT = 0.05; % s,

% Mean velocity
vm = 0.05; % µm/s

% Area size, just used to disperse particles in 2D. Has no impact on
% analysis.
SIZE = 2; % µm

tracks = cell(N_PARTICLES, 1);

k = sqrt(2 * D * dT);
for i = 1 : N_PARTICLES

    % Time
    time = (0 : N_TIME_STEPS-1)' * dT;

    % Velocity orientation
    theta = 2 * pi * rand;

    % Mean velocity
    v = vm * (1 + 1/4*randn);

    % Initial position
    X0 = SIZE .* rand(1, 2);

    % Instantaneous displacement:
    dX_brownian = k * randn(N_TIME_STEPS, 2);
    dX_directed = v * dT * ...
        [ cos(theta)*ones(N_TIME_STEPS,1) sin(theta)*ones(N_TIME_STEPS,1) ];

    % Integrate uncorrelated displacement
    dX = dX_brownian + dX_directed;
    dX(1, :) = X0;
    X = cumsum(dX, 1);

    % Store
    tracks{i} = [time X];

end
clear i X dX time X0

%%
ma = msdanalyzer(2, SPACE_UNITS, TIME_UNITS);
% ma = ma.addAll(tracks);
ma = ma.addAll(tracks_all);
figure
ma.plotTracks
ma.labelPlotTracks
%%
figure
ma.plotMSD;
%%
ma = ma.computeMSD;
figure
ma.plotMeanMSD(gca, true);
set(gca, 'XScale', 'log');set(gca, 'YScale', 'log');
%%
A = ma.getMeanMSD;
t = A(:, 1); % delay vector
msd = A(:,2); % msd
std_msd = A(:,3); % we will use inverse of the std as weights for the fit
std_msd(1) = std_msd(2); % avoid infinity weight

ft = fittype('a*x + c*x^2');
[fo, gof] = fit(t, msd, ft, 'Weights', 1./std_msd, 'StartPoint', [0 0]);

hold on
plot(fo)
legend off
ma.labelPlotMSD

Dfit = fo.a / 4;
Vfit = sqrt(fo.c);

ci = confint(fo);
Dci = ci(:,1) / 4;
Vci = sqrt(ci(:,2));

fprintf('Parabolic fit of the average MSD curve with 95%% confidence interval:\n')

fprintf('D = %.3g [ %.3g - %.3g ] %s, real value was %.3g %s\n', ...
    Dfit, Dci(1), Dci(2), [SPACE_UNITS '²/' TIME_UNITS], D, [SPACE_UNITS '²/' TIME_UNITS]);

fprintf('V = %.3g [ %.3g - %.3g ] %s, real value was %.3g %s\n', ...
    Vfit, Vci(1), Vci(2), [SPACE_UNITS '/' TIME_UNITS], vm, [SPACE_UNITS '/' TIME_UNITS]);
%%
ma = ma.computeVCorr;
figure
ma.plotMeanVCorr
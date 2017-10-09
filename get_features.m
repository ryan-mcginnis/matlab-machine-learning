
function feat = get_features(t,data,win_size,win_inc)
% Function to compute features from 1d signal data, with associated time
% stamps t. Features computed from sliding window of length win_size that
% slides by win_inc

% Extract sampling rate
fs = 1/mean(diff(t));

feat = [];
win_end = win_size;
while win_end <= length(data)
    %Define start of window
    win_start = win_end - win_size + 1;
    
    % Extract window of data
    d = data(win_start:win_end,:);

    % Extract standard features from signal
    featt = signal_features(d, fs);

    % Add to end of feature array
    feat = [feat; t(win_end), featt];
    
    %Slide window by number of samples defined in win_inc
    win_end = win_end + win_inc;
end

end

function feat = signal_features(x,fs)
    % Initialize feature vector
    feat = zeros(1,29);
    
    % Time domain
    feat(1) = mean(x);
    feat(2) = rms(x);
    feat(3) = skewness(x);
    feat(4) = kurtosis(x);
    feat(5) = range(x);
    feat(6) = max(x);
    feat(7) = min(x);
    feat(8) = std(x);
    feat(9) = peak2rms(x);
    feat(10:12) = covFeatures(x, fs);
    
    % Frequency domain
    feat(13:24) = spectralPeaksFeatures(x, fs);
    feat(25:29) = spectralPowerFeatures(x, fs);
end


% -----Helper functions
function feats = covFeatures(x, fs)

feats = zeros(1,3);

[c, lags] = xcorr(x);

minprom = 0.0005;
mindist_xunits = 0.3;
minpkdist = floor(mindist_xunits/(1/fs));
[pks,locs] = findpeaks(c,...
    'minpeakprominence',minprom,...
    'minpeakdistance',minpkdist);

tc = (1/fs)*lags;
tcl = tc(locs);
% Feature 1 - peak height at 0
if(~isempty(tcl))   % else f1 already 0
    feats(1) = pks((end+1)/2);
end
% Features 2 and 3 - position and height of first peak 
if(length(tcl) >= 3)   % else f2,f3 already 0
    feats(2) = tcl((end+1)/2+1);
    feats(3) = pks((end+1)/2+1);
end
end

function feats = spectralPeaksFeatures(x, fs)

mindist_xunits = 0.3;

feats = zeros(1,12);

N = 4096;
minpkdist = floor(mindist_xunits/(fs/N));

[p, f] = pwelch(x,rectwin(length(x)),[],N,fs);

[pks,locs] = findpeaks(p,'npeaks',20,'minpeakdistance',minpkdist);
if(~isempty(pks))
    mx = min(6,length(pks));
    [spks, idx] = sort(pks,'descend');
    slocs = locs(idx);

    pks = spks(1:mx);
    locs = slocs(1:mx);

    [slocs, idx] = sort(locs,'ascend');
    spks = pks(idx);
    pks = spks;
    locs = slocs;
end
fpk = f(locs);

% Features 1-6 positions of highest 6 peaks
feats(1:length(pks)) = fpk;
% Features 7-12 power levels of highest 6 peaks
feats(7:7+length(pks)-1) = pks;
end

function feats = spectralPowerFeatures(x, fs)

feats = zeros(1,5);

edges = [0.5, 1.5, 5, 10, 15, 20];

[p, f] = periodogram(x,[],4096,fs);
    
for kband = 1:length(edges)-1
    feats(kband) = sum(p( (f>=edges(kband)) & (f<edges(kband+1)) ));
end
end
%% Cluster Analysis
% This script provides example implementation of k-means cluster analysis
% in matlab  
% Written by Ryan S. McGinnis - ryan.mcginis14@gmail.com - July 9, 2016

% Copyright (C) 2016  Ryan S. McGinnis
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.


%% Load CSV of data - clomuns = features, rows = observations
filename = 'IRIS.csv';
dt_raw = readtable(filename, 'ReadVariableNames', false);


%% Clean observation with missing data
TF = ismissing(dt_raw);
dt = dt_raw(~any(TF,2),:);


%% Extract labels from variables
labels = dt(:,end);
dt = dt(:,1:end-1);


%% ID number of clusters to consider for k-means

% Extract data to array
X = table2array(dt);

% % Normalize
% X = (X - ones(size(X,1),1) * mean(X,1))./(ones(size(X,1),1) * std(X,1));

% Plot correlations between each variable
figure;
plotmatrix(X,X)

% Generate scree plot
num_clust = 5;
total_d = zeros(num_clust,1);
for i=1:num_clust
    [~,~,sumd] = kmeans(X,i,'Distance','sqeuclidean');
    total_d(i) = sum(sumd);
end

figure;
set(gcf,'name','total distance vs number of clusters');
plot(1:num_clust,total_d);
xlabel('Number of Clusters K');
ylabel('Total Within-Cluster Distance');


%% Run k-means for the appropriate number of clusters
T = kmeans(X,3,'Distance','sqeuclidean');

figure;
set(gcf,'name','silhouette plot of clusters');
[silh3,h] = silhouette(X,T,'sqeuclidean');
xlabel('Silhouette Value','fontsize',16);
ylabel('Cluster','fontsize',16);


%%  Plot of first two principal components in feature space
[coeff,score,latent,tsquared,explained] = pca(X);

figure;
gscatter(score(:,1),score(:,2),T)
ylabel('Component 2');
xlabel('Component 1');

% Class 5 - Aiyagari model
%
clear all; close all; clc;

% 1st task - The basic Aiyagari Model
% 2nd task - Heterogeneous returns
% Solutions are given by g, w, r and steady-state capital and labor

call_parameters;
numerical_parameters;
[grid] = create_grid(par,num);

% set seed for reproducibility
rng(1);

% Define models, which are basic Aiyagari or heterogeneous returns. We are going to loop over them.
models = {'basic_Aiyagari', 'heterogeneous_returns'};
for model = models
    model = model{1};
    fprintf('Running model: %s\n', model);
    if strcmp(model, 'basic_Aiyagari')
        par.r_factors = [1 1] ;
    elseif strcmp(model, 'heterogeneous_returns')
        par.r_factors = [0.95 1.05] ;
    end

    [r, w] = find_prices(par, num, grid);
    
    [g, c, aprime, labor_supply, capital_supply] = hh_problem(r, w, par, num, grid);
    [capital_demand, labor_demand] = firm_problem(labor_supply, r, w, par, num, grid);

    % Check market clearance
    fprintf('for the %s model:\n', model);
    checkMarketClearance(capital_demand, capital_supply, labor_demand, labor_supply, num);

    % Store the results
    results.(model).r = r;
    results.(model).w = w;
    results.(model).g = g;
    results.(model).c = c;
    results.(model).aprime = aprime;
    results.(model).labor_supply = labor_supply;
    results.(model).capital_supply = capital_supply;
    results.(model).capital_demand = capital_demand;
    results.(model).labor_demand = labor_demand;
end

% Display the results comparatively: r, w, k (capital demand)
modelNames = fieldnames(results); % Get the names of the models

fprintf('Comparative results:\n');
for i = 1:length(modelNames)
    modelName = modelNames{i}; % Get the current model name
    modelData = results.(modelName); % Access the model data
    
    % Print the results for the current model
    fprintf('%s:\n', strrep(modelName, '_', ' ')); % Replace underscores with spaces for readability
    fprintf('r = %.4f, w = %.4f, k = %.4f\n', modelData.r, modelData.w, sum(modelData.capital_demand));
end

%% Plot aggregate wealth distribution on the same chart.
figure;
hold on;
plotHandles = gobjects(0); % Initialize plotHandles as an empty array of graphic objects

for i = 1:length(modelNames)
    modelName = modelNames{i};
    modelData = results.(modelName);
    
    % Calculate the aggregate wealth distribution
    aggregate_wealth_distribution = sum(modelData.g, 2) * grid.da;
    
    % Plot and store the first handle only
    h = plot(grid.a, aggregate_wealth_distribution, 'DisplayName', strrep(modelName, '_', ' '));
    plotHandles(i) = h(1); % Store only the first handle if 'h' is an array
end

xlabel('Assets');
ylabel('Aggregate wealth distribution');
legend(plotHandles, 'Location', 'best');
title('Aggregate wealth distribution for different models');

% Plot the wealth distribution conditional on employment status
figure;
plotHandlesLow = gobjects(length(modelNames), 1); % Preallocate for low productivity
plotHandlesHigh = gobjects(length(modelNames), 1); % Preallocate for high productivity

for i = 1:length(modelNames)
    modelName = modelNames{i};
    modelData = results.(modelName);
    
    % Low productivity
    subplot(2, 1, 1);
    hold on;
    h1 = plot(grid.a, modelData.g(:, 1) / sum(modelData.g(:, 1)), 'DisplayName', strrep(modelName, '_', ' '));
    plotHandlesLow(i) = h1(1); % Store only the first handle
    
    % High productivity
    subplot(2, 1, 2);
    hold on;
    h2 = plot(grid.a, modelData.g(:, 2) / sum(modelData.g(:, 2)), 'DisplayName', strrep(modelName, '_', ' '));
    plotHandlesHigh(i) = h2(1); % Store only the first handle
end

% Low productivity legend
subplot(2, 1, 1);
xlabel('Assets');
ylabel('Wealth distribution (low productivity)');
legend(plotHandlesLow, 'Location', 'best');
title('Wealth distribution conditional on employment status (low productivity)');

% High productivity legend
subplot(2, 1, 2);
xlabel('Assets');
ylabel('Wealth distribution (high productivity)');
legend(plotHandlesHigh, 'Location', 'best');
title('Wealth distribution conditional on employment status (high productivity)');
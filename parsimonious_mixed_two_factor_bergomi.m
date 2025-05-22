function [prices] = parsiminious_mixed_two_factor_bergomi( )

eta_omega = 4.910; epsilon_omega = 14.128; beta_omega = 11.512; delta_omega = 0.727; % omega 1
eta_gamma = 8.748; epsilon_gamma = 19.173; beta_gamma = 2.336; delta_gamma = 2.0; % gamma
nu = 1.171; % nu


k1 = 7.54; k2 = 0.24; rho = 0.7; theta = 0.23;

% flat forward variance
xi0 = 0.03; rate = 0.0559;

% VIX maturities
t = 3/12;
T1 = t;
T2 = T1 + years(days(30));

omega1 = eta_omega*exp(-epsilon_omega*T1) + beta_omega*exp(-delta_omega*T1);  

gamma = eta_gamma*exp(-epsilon_gamma*T1) + beta_gamma*exp(-delta_gamma*T1);
gamma = gamma/(1+gamma);

alphaThetaSquared = 1 / ((1-theta)^2 + theta^2 + 2*rho*theta*(1-theta));

% Correlation between X1 and X2
nume = rho * ((1 - exp( -(k1 + k2) * t)) / (k1 + k2));
SD_X1 = sqrt((1 - exp(-2 * k1 * t)) / (2 * k1));
SD_X2 = sqrt((1 - exp(-2 * k2 * t)) / (2 * k2));
rho12 = nume / (SD_X1 * SD_X2);

% Quantizer and probabilities
X12_quantizer = load("qpoints_1450");
X12_quantizer(end, :) = [];

% Extract probabilities and quantization points
Probs = X12_quantizer(:, 1);
X1_quantizer = X12_quantizer(:, 2);
X2_quantizer_temp = X12_quantizer(:, 3);

% Adjust X2 quantizer
X2_quantizer = rho12 * X1_quantizer + sqrt(1 - rho12^2) * X2_quantizer_temp;

% Guass-Legendre (GL) quadrature for computing N values of h(t, T), f(x, t), etc.
% lgwt outputs the abcissors/nodes (T) and weights (W) for GL quadrature
N = 20;
[T, W] = lgwt(N, T1, T2);

% Compute h(t, T) and x(t, T)
h_tT = alphaThetaSquared * ((1-theta)^2 .* exp(-2 * k1 * (T - t)) .* SD_X1.^2 + ...
    theta^2 .* exp(-2 * k2 * (T - t)) .* SD_X2.^2 + ...
    2 * theta * (1-theta) .* exp(-(k1 + k2) * (T - t)) .* nume);

x_tT = sqrt(alphaThetaSquared) * ((1-theta) .* exp(-k1 * (T - t)) .* SD_X1 .* X1_quantizer' + ...
    theta .* exp(-k2 * (T - t)) .* SD_X2 .* X2_quantizer');

% Compute f(t, x)
f_tT = (1 - gamma) * exp(omega1 * x_tT - 0.5 * omega1^2 * h_tT) + ...
    gamma * exp(nu * x_tT - 0.5 * nu^2 * h_tT);

% Compute V(t, T1, T2) and model future price
V_t_T1_T2 = 10000*xi0 * sum(W .* f_tT, 1) / (T2 - T1);
model_future_price = sum(Probs' .* sqrt(V_t_T1_T2)); % Model future price

% Model call prices
strikes = [13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32]';
call_prices_paths = sqrt(V_t_T1_T2) - strikes;
call_prices_paths(call_prices_paths < 0) = 0; % Set negatives to zero

% Model call prices
model_call_prices = exp(-rate*T1)*sum(call_prices_paths .* ...
    repmat(Probs', size(call_prices_paths, 1), 1), 2);

% Model implied volatility
iv_model = blsimpv(exp(-rate*T1)*model_future_price, strikes, ...
    rate, T1, model_call_prices);

% spline interpolation
xx = linspace(strikes(1), strikes(end), 1000);
yy = spline(strikes, iv_model, xx);

figure;
plot(xx, yy, 'red', 'LineWidth', 1.5  );
hold on

plot([model_future_price model_future_price], [0 (max(iv_model)+10)],...
    'red', 'LineWidth', 1.5, 'LineStyle','--');

ylim([(min(iv_model)-0.2) (max(iv_model)+0.2)])
xlim([(min(strikes)-1) (max(strikes) + 2)])
hold off

set(gca, 'FontSize', 20, 'LineWidth', 1.2, 'TickDir', 'out');
legend('Model smile', 'Model future', 'Location','best','FontSize',18)
legend('boxoff')

t = sprintf('T = %.2f', T1);  
title(t, 'Interpreter', 'latex', 'FontSize', 26);
xlabel('Strike','FontSize',22)
ylabel('Implied volatility','FontSize',22)

prices.model_future_price = model_future_price;
prices.model_call_prices = model_call_prices;

end

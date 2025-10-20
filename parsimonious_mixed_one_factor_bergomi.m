
% Quantizer and probabilities
X_quantizer = load("qpoints_1000");
X_quantizer(end, :) = [];

eta_omega = 5.492; epsilon_omega = 14.087; beta_omega = 7.118; delta_omega = 0.929; % omega
eta_gamma = 6.870; epsilon_gamma = 16.121; beta_gamma = 2.188; delta_gamma = 1.523; % gamma

nu =  0.760; % nu

% Model parameter
k = 1.06;

% flat forward variance
xi0 = 0.03; rate = 0.0559;

% VIX maturities
t = 3/12;
T1 = t;
T2 = T1 + years(days(30));

omega1 = eta_omega*exp(-epsilon_omega*T1) + beta_omega*exp(-delta_omega*T1);

gamma = eta_gamma*exp(-epsilon_gamma*T1) + beta_gamma*exp(-delta_gamma*T1);
gamma = gamma/(1+gamma);

% Extract probabilities and quantization points
Probs = X_quantizer(:, 1);
quant_points = X_quantizer(:, 2);

% Standard deviation of X
SD_X = sqrt((1 - exp(-2 * k * t)) / (2 * k));

% Gauss-Legendre (GL) quadrature for computing N values of h(t, T), f(x, t), etc.
N = 20;
[T, W] = lgwt(N, T1, T2);

% h(t, T)
h_tT = exp(-2 * k * (T - t)) * SD_X^2;

% x(t, T)
x_tT = exp(-k * (T - t)) * SD_X * quant_points';

% Compute f(t, X)
f_tT =  (1 - gamma) * exp(omega1 * x_tT - 0.5 * omega1^2 * h_tT) + ...
    gamma * exp(nu * x_tT - 0.5 * nu^2 * h_tT);

% Compute V(t, T1, T2) and model futures price
V_t_T1_T2 = 10000*xi0 * sum(W .* f_tT, 1) / (T2 - T1);
model_future_price = sum(Probs' .* sqrt(V_t_T1_T2)); % Model futures price

% Model call prices
strikes = [13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32]';
call_prices_paths = sqrt(V_t_T1_T2) - strikes;
call_prices_paths(call_prices_paths < 0) = 0; 

model_call_prices = exp(- rate * T1) * sum(call_prices_paths .* ...
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



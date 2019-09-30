function [] = accuracyDemo(hmin)
% accuracyDemo(hmin);
% Demo program illustrating the order of accuracy (and global error) in
% the Euler method, Heun's method and Classical Runge-Kutta. The function
% plots the global (relative) error as a function of stepsize h. The 
% program begin with h=0.5 and gradually halve the stepsize down to h=hmin.
% If called without hmin, the default is hmin = 0.0001.

% The plot will also display the slopes of the graphs, i.e. the order of
% accuracy. The slope is calculated through a 1st degree polynomial fitting
% on the part of the graph that depend on discretization error (rather than
% roundoff errors).
%
% The program solves the ODE y'=t*y + t.^3, compare it with the analytic
% solution (exist for this ODE) and calculate the relative error.


% Close all figure windows (if any)
close all;

h_too_small = 1e-7; % Lower limit on h
hmax = 0.5;         % Upper limit on h
if nargin==0
    hmin = 1e-4;     % hmin default
end

if hmin > hmax/8
    disp(['hmin too big, must be <= ',num2str(hmax/8)]);
    disp('Program interupted. Run again with another hmin'); 
    return;
end

% Create a vector h with h-values from hmax and gradually halving down to hmin
k = floor(log2(hmin/hmax));
h = (2.^(0:-1:k))'*hmax;
if hmin < h_too_small
    disp('hmin too small => computation will take forever.');
    disp('Program interupted. Run again with a larger hmin');
    return;
end


% Initial value and interval
y0 = 1;
tspan= [0 1];
% Initiate error vectors
errE = zeros(length(h),1);
errH = zeros(length(h),1);
errRK = zeros(length(h),1);
% Solve the ode:n for each h-value and three methods
for i = 1:length(h)
    [t,y] = euler(@ft1, tspan, y0, h(i));
    y_korr = (2+y0).*exp((t.^2)/2) - t.^2 -2; % Analytic solution
    errE(i) = norm(y_korr - y)/norm(y_korr);
    [~,y] = heun(@ft1, tspan, y0, h(i));
    errH(i) = norm(y_korr - y)/norm(y_korr);
    [~,y] = RK4(@ft1, tspan, y0, h(i));
    errRK(i) = norm(y_korr - y)/norm(y_korr);
end

% Draw the graph
figure('Name','Error as a function of stepsize','NumberTitle','off');
loglog(h,errE,'b-','LineWidth',1);
hold on
loglog(h,errH,'r-.','LineWidth',1);
loglog(h,errRK,'m--','LineWidth',1);
hold off
ylabel('Relative error');
xlabel('Stepsize h');
legend({'Euler''s metod','Heun''s metod','Classical Runge-Kutta'},'Location','NorthWest');

% Find slope if many enough h-values in h
if length(h)>=8
    ind= 4:1:8;
    p_slope = polyfit(log(h(ind)),log(errE(ind)),1);
    slopeE = round(p_slope(1),1);
    p_slope = polyfit(log(h(ind)),log(errH(ind)),1);
    slopeH = round(p_slope(1),1);
    p_slope = polyfit(log(h(ind)),log(errRK(ind)),1);
    slopeRK = round(p_slope(1),1);
    text(0.23,1e-1,['Slope: ' num2str(slopeE)]);
    text(0.23,errH(1),['Slope: ' num2str(slopeH)]);
    text(0.23,errRK(1),['Slope: ' num2str(slopeRK)]);
end

% ----------------------------
% Internal functions
% ----------------------------

% The ODE right-hand-side
function y_out = ft1(t, y)
y_out = t*y + t.^3;

%--
% Euler's method

function [t,yout] = euler(func, tspan, y0, h)
%
t = (tspan(1):h:tspan(2))';
yout = zeros(length(t),1);
yout(1) = y0;
for i = 2:length(t)
    yout(i) = yout(i-1)+h*func(t(i-1),yout(i-1));
end

%--
% Classical Runge-Kutta

function [t,yout] = RK4(func, tspan, y0, h)
t = (tspan(1):h:tspan(2))';
yout = zeros(length(t),1);
yout(1) = y0;
for i = 2:length(t)
    k1 = func(t(i-1),yout(i-1));
    k2 = func(t(i-1)+0.5*h, yout(i-1)+0.5*h*k1);
    k3 = func(t(i-1)+0.5*h, yout(i-1)+0.5*h*k2);
    k4 = func(t(i-1)+h,yout(i-1)+k3*h);
    yout(i) = yout(i-1)+(h/6)*(k1+ 2*k2 + 2*k3 + k4);
end

%--
% Heun's method

function [t,yout] = heun(func, tspan, y0, h)
t = (tspan(1):h:tspan(2))';
yout = zeros(length(t),1);
yout(1) = y0;
for i = 2:length(t)
    k1 = func(t(i-1),yout(i-1));
    k2 = func(t(i),yout(i-1)+h*k1);
    yout(i) = yout(i-1)+0.5*h*(k1+k2);
end
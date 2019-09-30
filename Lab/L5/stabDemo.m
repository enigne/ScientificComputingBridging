function [] = stabDemo(lambda)
% StabDemo(lambda)
% Demo program illustrating numerical stability in explicit Euler's method
% and implicit Euler method. 
% The program solves the differential equationen
%    y' = -lambda*y
%    y0 = 1
% on the interval t = [0 1]. The parameter lambda is an inparameter
% (default lambda = 50). The problem gets increasingly stiffer, when lambda
% gets larger.
% The program displays the stability criterion for explicit Euler, and the 
% user is asked to input stepsize h. Try both h smaller than, equal to, and 
% larger than the stability criterion, and see what effect it has on the 
% solutions from the two methods. Also choose different lambda-values and 
% study what constraint it will put on the explicit Euler method.
%
% (Eulers method has stability criterion h < 2/lambda for this problem, and
% implicit Euler is unconditionally stable, i.e. stable for all 
% stepsizes h).

% Stefan P�lsson, 2010. Updated 2018

close all;

if (nargin == 0)
    lambda = 50;
    disp('You did not call the function with inparameter lambda');
    disp(['Default lambda = ',num2str(lambda),' will be used']);
    disp(' ');
end

y0 = 1;
tspan = [0 1];
disp(['Stability criterion for explicit Euler method is h<',num2str(2/lambda)]);
h = input('Choose stepsize h: ');

% Call explicit and implicit Euler
[texpl, yexpl] = euler(@(t,y) odeRHS(t,y,lambda), tspan, y0, h);
[timpl, yimpl] = implicitEuler(@(t,y) odeRHS(t,y,lambda), tspan, y0, h);


% Analytic solution
tkorr = linspace(tspan(1),tspan(2),300);
ykorr = y0*exp(-lambda*tkorr);

% Plotting
f1=figure('Name','Stability - explicit Euler','NumberTitle','on');
plot(texpl,yexpl,'r-',tkorr,ykorr,'b--','LineWidth',1);
title(['Explicit Euler, stepsize h = ',num2str(h),...
    ' \lambda = ',num2str(lambda)]);
legend('Explicit Euler','Analytic solution');
ylabel('y(t)');
xlabel('t');

figpos = get(f1,'Position'); % Don't want figure windows on top of each other
figpos=figpos-[100 50 0 0];
figure('Name','Stability - Implicit Euler','NumberTitle','on','Position',figpos);
plot(timpl,yimpl,'r-',tkorr,ykorr,'b--','LineWidth',1);
title(['Implicit Euler, stepsize h = ',num2str(h),' \lambda = ',num2str(lambda)]);
legend('Implicit Euler','Analytic solution');
ylabel('y(t)');
xlabel('t');

end % main


% -----
% Internal functions
% -----

% The right-hand-side of the ODE
function yout = odeRHS(~, y, lambda)
yout = -lambda*y;
end

% Euler's method (explicit)
function [t,yout] = euler(func, tspan, y0, h)
%
t = (tspan(1):h:tspan(2))';
y = zeros(length(t),1);
y(1) = y0;
for i = 2:length(t)
    y(i) = y(i-1)+h*func(t(i-1),y(i-1));
end
yout = y;
end

% Implicit Euler
function [t,yout] = implicitEuler(func, tspan, y0, h)
%
t = (tspan(1):h:tspan(2))';
yout = zeros(length(t),1);
yout(1) = y0;
for i = 2:length(t)
    f = @(x) h*(func(t(i),x)) - x + yout(i-1);  
    yout(i) = fzero(f, yout(i-1));
end
end
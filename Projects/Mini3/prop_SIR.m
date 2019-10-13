function w = prop_SIR(u, p)
%
% w = prop_predprey(u, p)
% Propensities, w, for the Predator-Prey system.
% 
% Input: u - the current state. 
%        p - the parameters used to calculate the propensities 
%
% The current state variables (u) are ordered as:
%    [S; I; R]
% The parameters (in p) are ordered as:
%   [b; d; beta; u; v]
%

b = p(1);
d = p(2);
beta = p(3);
uu = p(4);
vv = p(5);

S = u(1);
I = u(2);
R = u(3);
N = S+I+R;

w = [beta*I*S/N;
    uu*I;
    vv*S;
    d*S;
    d*I;
    d*R;
    b*N];


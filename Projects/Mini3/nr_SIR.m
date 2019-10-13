function nr = nr_SIR( )
% nr = nr_predprey()
%
% Stoichiometry matrix, nr, for the vilar oscillator.
% The variables (corresponding to the columns in nr) are ordered as:
%   y1  y2
%

nr = zeros(7, 3);

nr(1,:) =   [-1 1 0];
nr(2,:) =   [0 -1 1];
nr(3,:) =   [-1 0 1];
nr(4,:) =   [-1 0 0];
nr(5,:) =   [0 -1 0];
nr(6,:) =   [0 0 -1];
nr(7,:) =   [1 0 0];

    

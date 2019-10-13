function nr = nr_SIR( )
%
% Stoichiometry matrix, nr, for the SIR.
% The variables (corresponding to the columns in nr) are ordered as:
%   S  I  R
%

nr = zeros(7, 3);

nr(1,:) =   [-1 1 0];
nr(2,:) =   [0 -1 1];
nr(3,:) =   [-1 0 1];
nr(4,:) =   [-1 0 0];
nr(5,:) =   [0 -1 0];
nr(6,:) =   [0 0 -1];
nr(7,:) =   [1 0 0];

    

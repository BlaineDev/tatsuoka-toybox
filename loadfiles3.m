% number of files being read in each group.
fileJ = 14;        %Group1, A0 in code   
fileK = 16;        %Group2, A1 in code
  
% enter in N# of filenames.  This creates an array of the files
 fname1 = { 'pat2.txt', ...
            'pat4.txt', ...
            'pat6.txt', ...
            'pat7.txt', ...
            'pat10.txt', ...
            'pat11.txt', ...
            'pat12.txt', ...
            'pat18.txt', ...
            'pat19.txt',  ...
            'pat20.txt', ...
            'pat22.txt', ...
            'pat25.txt', ...
            'pat26.txt', ...
            'pat28.txt'
        };
           
 
fname2 = {  'pat1.txt', ...
            'pat3.txt', ...
            'pat5.txt', ...
            'pat8.txt', ...
            'pat9.txt', ...
            'pat13.txt', ...
            'pat14.txt', ...
            'pat15.txt', ...
            'pat16.txt', ...
            'pat17.txt', ...
            'pat21.txt', ...
            'pat23.txt', ...
            'pat24.txt', ...
            'pat27.txt', ...
            'pat29.txt', ...
            'pat30.txt'
        };    
             








%%

    
for i = 1:fileJ
      fname1{i} = readmatrix(fname1{i});   %opens the content of each filename, reusing the array locations 
      A0{i} = fname1{i}+ triu(fname1{i},1)'; %convert triangle matrix to square, into a new array %M = M + triu(M,1)';
     % A0_eigens{i} = eig(A0{i});  % get eigenvalues @blaine: of Graph
     % Laplacian
     % A0_eigen_max{i} = max(A0_eigens{i});  %get max eigenvalue
end



%%
 
 for i = 1:fileK
      fname2{i} = readmatrix(fname2{i});   %opens the content of each filename, reusing the array locations 
      A1{i} = fname2{i}+ triu(fname2{i},1)'; %convert triangle matrix to square, into a new array %M = M + triu(M,1)';
      %A1_eigens{i} = eig(A1{i});  % get eigenvalues
      %A1_eigen_max{i} = max(A1_eigens{i});  %get max eigenvalue
 end

 
A4 = A0_eigen_max';  %note transpose 
A5 = A1_eigen_max';

A4 = max([A4{:}]);  %find max value
A5 = max([A5{:}]);
disp(A4); 
disp(A5);
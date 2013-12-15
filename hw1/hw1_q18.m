%=====================================================================
% Course:   Machine Learning Foundations
% Topic:    Homework 1 - Q18, Q19, Q20
% Details:  For Questions 15-20, you will play with PLA and pocket
%           algorithm. First, we use an artificial data set to study
%           PLA.
%=====================================================================
function hw1_q18

   %------------------------------------------------------------------
   % Parameters
   %------------------------------------------------------------------

   FILENAME_18_TRAIN = 'hw1_18_train.dat';
   FILENAME_18_TEST  = 'hw1_18_test.dat';

   %------------------------------------------------------------------
   % Data Initialization
   %------------------------------------------------------------------

   % Load the data.
   Data = load(FILENAME_18_TRAIN);
   [Rows Cols] = size(Data);

   % Initialize w and y.
   w = zeros(Cols, 1);
   y = Data(1:Rows, Cols);

   % Initialize X.
   X = ones(Rows, Cols);
   X(1:Rows, 2:Cols) = Data(1:Rows, 1:(Cols - 1));

   % Load the test data set.
   DataT = load(FILENAME_18_TEST);
   [RowsT ColsT] = size(DataT);

   % Initialize Xt and yt;
   yt = DataT(1:RowsT, ColsT);
   Xt = ones(RowsT, ColsT);
   Xt(1:RowsT, 2:ColsT) = DataT(1:RowsT, 1:(ColsT - 1));

   %------------------------------------------------------------------
   % Question 18
   %------------------------------------------------------------------
   
   disp('[Question 18]')

   tic;
   err_rate = 0;
   for k = 1 : 2000
      [Count wg wu] = Pocket(X, y, w, 50);

      if Count ~= 50
         disp('Oops...');
      end

      err_rate = err_rate + sum((sign(Xt * wg .* yt) <= 0));
   end

   err_rate = err_rate / RowsT / 2000;
   ElapsedTime = toc;
   
   disp(sprintf('Error rate:   %.2f', err_rate));
   disp(sprintf('Elapsed time: %.2f\n', ElapsedTime));

   %------------------------------------------------------------------
   % Question 19
   %------------------------------------------------------------------

   disp('[Question 19]')

   tic;
   err_rate = 0;
   for k = 1 : 2000
      [Count wg wu] = Pocket(X, y, w, 50);

      if Count ~= 50
         disp('Oops...');
      end

      err_rate = err_rate + sum((sign(Xt * wu .* yt) <= 0));
   end

   err_rate = err_rate / RowsT / 2000;
   ElapsedTime = toc;

   disp(sprintf('Error rate:   %.2f', err_rate));
   disp(sprintf('Elapsed time: %.2f\n', ElapsedTime));

   %------------------------------------------------------------------
   % Question 20
   %------------------------------------------------------------------

   disp('[Question 20]')

   tic;
   err_rate = 0;
   for k = 1 : 2000
      [Count wg wu] = Pocket(X, y, w, 100);

      if Count ~= 100
         disp('Oops...');
      end

      err_rate = err_rate + sum((sign(Xt * wg .* yt) <= 0));
   end

   err_rate = err_rate / RowsT / 2000;
   ElapsedTime = toc;

   disp(sprintf('Error rate:   %.2f', err_rate));
   disp(sprintf('Elapsed time: %.2f\n', ElapsedTime));

end

%=====================================================================
% Pocket Algorithm
%=====================================================================

function [Count wg wu] = Pocket(X, y, w, update_limit)
   Count = 0;
   idx = 0;
   [Rows Cols] = size(X);
   wg = w;
   wu = w;
   err_wg = Rows;

   while true
      err_w = sum((sign(X * w .* y) <= 0));

      if err_w < err_wg
         err_wg = err_w;
         wg = w;
      end

      if ~err_w
         break;
      end

      idxs = find(sign(X * w .* y) <= 0);

      % Pick a mistake randomly.
      idx = randsample(idxs, 1);

      % Update
      w = w + (X(idx, :) * y(idx))';
      Count = Count + 1;

      % Prevention
      if Count >= update_limit
         wu = w;
         break;
      end
   end

end

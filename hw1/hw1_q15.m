%=====================================================================
% Course:   Machine Learning Foundations
% Topic:    Homework 1 - Q15, Q16, Q17
% Details:  For Questions 15-20, you will play with PLA and pocket
%           algorithm. First, we use an artificial data set to study
%           PLA.
%=====================================================================
function hw1_q15()

   %------------------------------------------------------------------
   % Parameters
   %------------------------------------------------------------------

   FILENAME_15_TRAIN = 'hw1_15_train.dat'

   %------------------------------------------------------------------
   % Data Initialization
   %------------------------------------------------------------------

   % Load the data.
   Data = load(FILENAME_15_TRAIN);
   [Rows Cols] = size(Data);

   % Initialize w and y.
   w = zeros(Cols, 1);
   y = Data(1:Rows, Cols);

   % Initialize X.
   X = ones(Rows, Cols);
   X(1:Rows, 2:Cols) = Data(1:Rows, 1:(Cols - 1));

   %------------------------------------------------------------------
   % Question 15
   %------------------------------------------------------------------

   [Count Flag] = PLA(X, y, w, 1, false);

   if Flag
      disp(sprintf('Q15 Updates: %d', Count));
   else
      disp('Update > 10000!');
      return;
   end

   %------------------------------------------------------------------
   % Question 16
   %------------------------------------------------------------------

   update_all = 0;
   for k = 1 : 2000
      [Count Flag] = PLA(X, y, w, 1, true);
      if ~Flag
         disp('Oops...')
         return;
      else
         update_all = update_all + Count;
      end
   end

   disp(sprintf('Q16 Updates: %.2f', update_all / 2000));

   %------------------------------------------------------------------
   % Question 17
   %------------------------------------------------------------------
   
   update_all = 0;
   for k = 1 : 2000
      [Count Flag] = PLA(X, y, w, .5, true);
      if ~Flag
         disp('Oops...');
         return
      else
         update_all = update_all + Count;
      end
   end

   disp(sprintf('Q17 Updates: %.2f', update_all / 2000));

end

%=====================================================================
% Perceptron Learning Algorithm
%=====================================================================
function [Count Flag] = PLA(X, y, w, eta, israndom)
   Count = 0;
   idx = 0;
   [Rows Cols] = size(X);

   while sum((sign(X * w .* y) <= 0))
      idxs = find(sign(X * w .* y) <= 0);

      if ~israndom
         % Pick a mistake next to the previous one. If
         % it does not exist, pick the first met one.
         idx = find(idxs > idx, 1);
         if isempty(idx)
            idx = idxs(1);
         else
            idx = idxs(idx);
         end
      else
         % Pick a mistake randomly.
         idx = randsample(idxs, 1);
      end

      % Update
      w = w + (X(idx, :) * y(idx))' * eta;
      Count = Count + 1;

      % Prevention
      if Count > 10000
         Flag = false;
         return;
      end
   end

   Flag = true;
end

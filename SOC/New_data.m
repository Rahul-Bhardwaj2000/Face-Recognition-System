function New_X=New_data(X)
  [mean,principal,s]=pca(X);
  X=X/255;  %changed now
  X=X';
  [n m] =size(X)
  X=X-(sum(X,2))/m;
  New_X=(principal')*(X);
  New_X=New_X';


We need it to load in the 1d gradients from a json file 
This thing just needs to have y values at x values for like eight different analysis gradients

Then there will be a function that takes some integer number, which is the number of effective gradients
One function will take parameters and make the effective gradients
Then it will try to work the normalization on each of the gradietns to minimize the difference of the sum of the effective gradient to each analysis gradient. 
We also penalize normalizations being different from 1.0 (gaussian?)

This process is repeated for each number of effective gradients
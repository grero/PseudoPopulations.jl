Usage
-------------

`Pseudopopulations` allows one to concatenate labeled data to create a fake population with a number of rows equal to the sum of the rows of each population.

```julia
X = Array{Float64,3}[randn(20,5,1), randn(23,4,1)]
labels = Array{Int64,1}[rand(1:2,20), rand(1:2, 23)]
pp = PseudoPopulations.PseudoPopulation(X, labels)
```

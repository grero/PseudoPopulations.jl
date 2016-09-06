using Base.Test
import PseudoPopulations

X = Array{Float64,3}[randn(20,5,1), randn(23,4,1)]
labels = Array{Int64,1}[rand(1:2,20), rand(1:2, 23)]
pp_train = PseudoPopulations.PseudoPopulation(X, labels)
pp_test = PseudoPopulations.PseudoPopulation(X, labels,pp_train.index)

@test_approx_eq X[1][pp_train.index[1], :,:] pp_train.X[:,1:5,:]
@test_approx_eq X[2][pp_train.index[2], :,:] pp_train.X[:,6:end,:]

@test_approx_eq X[1][pp_test.index[1], :,:] pp_test.X[:,1:5,:]
@test_approx_eq X[2][pp_test.index[2], :,:] pp_test.X[:,6:end,:]

@test isempty(intersect(pp_train.index, pp_test.index))

println("All tests passed")

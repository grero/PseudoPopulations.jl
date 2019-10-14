using PseudoPopulations
using Test

X = Array{Float64,3}[randn(20,5,1), randn(23,4,1)]
labels = Array{Int64,1}[rand(1:2,20), rand(1:2, 23)]
pp_train = PseudoPopulations.PseudoPopulation(X, labels)
pp_test = PseudoPopulations.PseudoPopulation(X, labels,pp_train.samples.index)

@test X[1][pp_train.samples.index[1], :,:] ≈ pp_train.X[:,1:5,:]
@test X[2][pp_train.samples.index[2], :,:] ≈ pp_train.X[:,6:end,:]

@test X[1][pp_test.samples.index[1], :,:] ≈ pp_test.X[:,1:5,:]
@test X[2][pp_test.samples.index[2], :,:] ≈ pp_test.X[:,6:end,:]

@test labels[1][pp_test.samples.index[1]] == pp_test.label
@test labels[2][pp_test.samples.index[2]] == pp_test.label

@test labels[1][pp_train.samples.index[1]] == pp_train.label
@test labels[2][pp_train.samples.index[2]] == pp_train.label

@test isempty(intersect(pp_train.samples.index, pp_test.samples.index))

Za = rand(3,10,2)
labels = Array{Int64,1}[rand(1:2,rand(1:10)) for j in 1:2]
session_id = rand(1:2, size(Za,1))
PP = PseudoPopulations.PseudoPopulation(Za, labels, session_id,5);
@test PP.X[:,1,:] ≈ Za[1, PP.samples.index[1,:], :]
@test PP.X[:,2,:] ≈ Za[2, PP.samples.index[2,:], :]
@test PP.X[:,3,:] ≈ Za[3, PP.samples.index[3,:], :]

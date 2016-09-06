module PseudoPopulations

type PseudoPopulation
	X::Array{Float64,3}
	label::Array{Int64,1}
	index::Array{Array{Int64,1},1}
end

"""
Create a pseudopopulation by concatenating along the first dimension of `Z`, creating pseuotrials by balancing identical labels across each ement of `Z`.

	function PseudoPopulation(Z::Array{Array{Float64,3},1},labels::Array{Array{Int64,1},1})
"""
function PseudoPopulation(Z::Array{Array{Float64,3},1},labels::Array{Array{Int64,1},1})
	nbins = size(Z[1],3)
	min_ntrials = minimum(map(length, labels))
	ulabels = Array(Int64,0)
	min_nlabels = Dict{Int64,Int64}()
	for (_labels) in labels
		_label_count = Dict{Int64,Int64}()
		for l in _labels
			_label_count[l] = get(_label_count, l, 0) + 1
		end
		for (k,v) in _label_count
			min_nlabels[k] = min(get(min_nlabels,k,typemax(Int64)), v)
		end
	end
	ulabels = collect(keys(min_nlabels))
	sort!(ulabels)
	nlabels = length(ulabels)

	ntrain_per_label = Dict([k=>round(Int,0.8*v) for (k,v) in min_nlabels])
	ntrain = sum(values(ntrain_per_label))
	#ntest_per_label = div(ntest,nlabels)
	#ntest = ntest*per_label*nlabels

	ncells = sum(x->size(x,2), Z)
	#training set
	Z_train = zeros(ntrain,ncells,nbins)
	labels_train = zeros(Int64,ntrain)
	used_indices = [Array(Int64,0) for i in 1:length(Z)]
	#Z_test = zeros(ntest,nbins,ncells)
	toffset = 0
	for l in ulabels
		offset = 0
		for (z,_label,_indices) in zip(Z, labels,used_indices)
			_idx = find(_label.==l)
			shuffle!(_idx)
			for j in 1:ntrain_per_label[l]
				push!(_indices, _idx[j])
				labels_train[toffset+j] = l
			end
			for b in 1:size(z,3)
				for c in 1:size(z,2)
					for j in 1:ntrain_per_label[l]
						Z_train[toffset + j, c+offset,b] = z[_idx[j],c,b]
					end
				end
			end
			offset += size(z,2)
		end
		toffset += ntrain_per_label[l]
	end
	PseudoPopulation(Z_train,labels_train,used_indices)
end

end#mddule

module PseudoPopulations

abstract TrialSampler

type UniqueTrialSampler <: TrialSampler
	index::Array{Array{Int64,1},1}
end

type RepeatedTrialSampler <: TrialSampler
	index::Array{Int64,2}
end

type PseudoPopulation{T<:TrialSampler}
	X::Array{Float64,3}
	label::Array{Int64,1}
	samples::T
end

function PseudoPopulation(Za::Array{Float64,3},trial_labels::Array{Array{Int64,1},1}, session_id::Array{Int64,1},trials_per_location=50)
	ulabels = intersect(map(unique,trial_labels)...)
	sort!(ulabels)
	nlabels = length(ulabels)
	ncells, ntrials, nbins = size(Za)
	nn = trials_per_location
	N = nn*nlabels #total number of trials
	Xq = zeros(nbins, ncells, nn*nlabels)
	trialidx = zeros(Int64, ncells,nn*nlabels)
	labels = zeros(Int64,N)
	for (li,l) in enumerate(ulabels)
		for i in 1:nn
			kk = (li-1)*nn + i
			labels[kk] = l
			for c in 1:ncells
				_sidx = session_id[c]
				_tlabels = trial_labels[_sidx]
				_idx = max(1,findfirst(x->(x==l)&(rand()<0.5), _tlabels))
				trialidx[c,kk] = _idx
				Xq[:,c,kk] .= Za[c,_idx,:]
			end
		end
	end
	PseudoPopulation(permutedims(Xq,[3,2,1]), labels, RepeatedTrialSampler(trialidx))
end

"""
Create a pseudopopulation by concatenating along the first dimension of `Z`, creating pseuotrials by balancing identical labels across each ement of `Z`.

	function PseudoPopulation(Z::Array{Array{Float64,3},1},labels::Array{Array{Int64,1},1})
"""
function PseudoPopulation(Z::Array{Array{Float64,3},1},labels::Array{Array{Int64,1},1},sample_ratio::Real=0.8)
	nbins = size(Z[1],3)
	min_ntrials = minimum(map(length, labels))
	ulabels = Array(Int64,0)
	min_nlabels = Dict{Int64,Int64}()
	label_counts = [similar(min_nlabels) for i in 1:length(Z)]
	for (_labels,_label_count) in zip(labels,label_counts)
		for l in _labels
			_label_count[l] = get(_label_count, l, 0) + 1
		end
		append!(ulabels, collect(keys(_label_count)))
	end
	ulabels = unique(ulabels)
	sort!(ulabels)
	for l in ulabels
		min_nlabels[l] = minimum([get(_label_count,l,0) for _label_count in label_counts])
	end
	nlabels = length(ulabels)

	ntrain_per_label = Dict(k=>round(Int,sample_ratio*v) for (k,v) in min_nlabels)
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
			if isempty(_idx)
				continue
			end
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
	PseudoPopulation(Z_train,labels_train,UniqueTrialSampler(used_indices))
end

function PseudoPopulation(Z::Array{Array{Float64,3},1},labels, sample_ratio)
	cellidx = Range{Int64}[1:size(z,2) for z in Z]
	PseudoPopulation(Z, labels, cellidx, sample_ratio)
end

PseudoPopulation(Z,labels) = PseudoPopulation(Z,labels, 0.8)

function PseudoPopulation(Z::Array{Array{Float64,3},1},labels::Array{Array{Int64,1},1},exclude_indices::Array{Array{Int64,1},1})
	Z2 = similar(Z)
	labels2 = similar(labels)
	use_idx = Array(Array{Int64,1},length(Z))
	for (i,(z,label,idx)) in enumerate(zip(Z,labels,exclude_indices))
		use_idx[i] = setdiff(1:size(z,1), idx)
		Z2[i] = z[use_idx[i], :,:]
		labels2[i] = label[use_idx[i]]
	end
	pp = PseudoPopulation(Z2,labels2,1.0)
	#relabel
	for (i,(l1,l2)) in enumerate(zip(use_idx, pp.samples.index))
		pp.samples.index[i] = l1[l2]
	end
	pp
end

end#mddule

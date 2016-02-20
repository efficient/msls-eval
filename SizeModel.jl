module SizeModel

using Common
# using NLopt
using Ipopt

function init_sizes(X::Distribution, l1_size::Float64, growth_factor::Float64=0., level_count::Int64=0)
	n = X.count

	if growth_factor != 0.
		# we are fine
	elseif level_count != 0
		growth_factor = exp(log(Float64(n) / l1_size) / Float64(level_count - 1))
	else
		@assert false
	end

	sizes = Array(Float64, 0)
	i = 1
	while true
		size = l1_size * ceil(growth_factor ^ Float64(i - 1))
		if size < Float64(n)
			push!(sizes, size)
		else
			push!(sizes, Float64(n))
			break
		end
		i += 1
	end

	sizes
end

function calculate_ra!(X::Distribution, X_q::Distribution, log_size::Float64, l0_count::Float64, sizes::Array{Float64}, ra::Array{Float64})
end

function calculate_wa!(X::Distribution, log_size::Float64, l0_count::Float64, sizes::Array{Float64}, wa_r::Array{Float64}, wa_w::Array{Float64})
	@assert sizes[end] == Float64(X.count)

	# mem->log
	wa_r[1] = 0.
	wa_w[1] = 1.

	# log->0
	wa_r[2] = 0.
	wa_w[2] = Common.unique(X, log_size) / log_size

	# ## amortized, full destination level
	# # 0->1
	# wa[3] = Common.unique(X, unique_inv(X, sizes[1]) + (log_size * l0_count)) / (log_size * l0_count)
	# # 1->2, 2->3, ...
	# for i in 1:(length(sizes) - 1)
	#	if i < length(sizes) - 1
	#		wa[3 + i] = merge(X, sizes[i + 1], sizes[i]) / unique_inv(X, sizes[i])
	#	else
	#		wa[3 + i] = Float64(X.count) / unique_inv(X, sizes[i])
	#	end
	# end

	# ## amortized, compact entire level, discrete interval calculation (maybe accurate, but does not work with optimizer)
	# # 0->1
	# interval = log_size * l0_count
	# next_interval = unique_inv(X, sizes[1])
	# effective_next_interval = floor(next_interval / interval + 1.) * interval
	# wa[3] = unique_avg(X, interval, effective_next_interval) / interval
	# # 1->2, 2->3, ...
	# for i in 1:(length(sizes) - 1)
	#	interval = effective_next_interval
	#	if i < length(sizes) - 1
	#		next_interval = unique_inv(X, sizes[i + 1])
	#		effective_next_interval = floor(next_interval / interval + 1.) * interval
	#		wa[3 + i] = unique_avg(X, interval, effective_next_interval) / interval
	#	else
	#		wa[3 + i] = sizes[end] / interval
	#	end
	# end

	# note that "interval * 0.5" is the overflown amount that causes compaction
	# 0.5 is just an approximate; it should be lower under high skew or with a key count close to the total unique count
	# because the level grows slowly as its size approaches the maximum level size.

	# ## amortized, compact entire level, continuous interval calculation (maybe less accurate)
	# # 0->1
	# interval = log_size * l0_count
	# next_interval = unique_inv(X, sizes[1])
	# wa[3] = unique_avg(X, interval, interval * 0.5 + next_interval) / interval
	# # 1->2, 2->3, ...
	# for i in 1:(length(sizes) - 1)
	#   interval = interval * 0.5 + next_interval
	#   if i < length(sizes) - 1
	#   	next_interval = unique_inv(X, sizes[i + 1])
	#   	wa[3 + i] = unique_avg(X, interval, interval * 0.5 + next_interval) / interval
	#   else
	#   	wa[3 + i] = sizes[end] / interval
	#   end
	# end

	## deamortized, compact each sstable in a round-robin way
	# 0->1
	interval = log_size * l0_count
	next_interval = unique_inv(X, sizes[1])
	wa_r[3] = (Common.unique(X, log_size) * l0_count + sizes[1]) / interval
	wa_w[3] = Common.unique(X, interval + next_interval) / interval
	# 1->2, 2->3, ...
	for i in 1:(length(sizes) - 1)
		# we need to take the previous interval as part of this interval ("interval +")
		# because the current level temporarily has to accommodate the data from the previous level
		interval = interval + interval_from_density(X, sizes[i])
		if i < length(sizes) - 1
			next_interval = unique_inv(X, sizes[i + 1])
			# plus unique(X, interval) * 1 to WA because of the overlapping tables' keys that do not actually overlap the compaction key range
			# TODO: this may become less accurate with spatial locality in key range because the overlapping tables' key range may be sparse
			wa_r[3 + i] = (Common.unique(X, interval) + sizes[i + 1] + Common.unique(X, interval) * 1.) / interval
			wa_w[3 + i] = (Common.unique(X, interval + next_interval) + Common.unique(X, interval) * 1.) / interval
		else
			wa_r[3 + i] = (Common.unique(X, interval) + sizes[end] + Common.unique(X, interval) * 1.) / interval
			wa_w[3 + i] = (sizes[end] + Common.unique(X, interval) * 1.) / interval
		end
	end

	wa_r, wa_w
end

function calculate_wa(X::Distribution, log_size::Float64, l0_count::Float64, sizes::Array{Float64})
	wa_r = Array(Float64, 2 + length(sizes))
	wa_w = Array(Float64, 2 + length(sizes))
	calculate_wa!(X::Distribution, log_size::Float64, l0_count::Float64, sizes::Array{Float64}, wa_r, wa_w)
end

function optimize_wa(X::Distribution, log_size::Float64, l0_count::Float64, init_sizes::Array{Float64}, wa_r_factor::Float64, ftol::Float64, max_time::Float64)
	n = X.count
	level_count = length(init_sizes)

	# v2 = Array(Float64, level_count)
	# v2[level_count] = n

	# count = 0
	# wa = Array(Float64, 2 + level_count)
	# f = (v, grad) -> begin
	#	if length(grad) > 0
	#		v2[1:level_count - 1] = v
	#		y = get_wa(wa_r_factor, calculate_wa!(X, log_size, l0_count, v2, wa))
	#		for i = 1:length(grad)
	#			org = v2[i]
	#			v2[i] += 1
	#			grad[i] = get_wa(wa_r_factor, calculate_wa!(X, log_size, l0_count, v2, wa)) - y
	#			v2[i] = org
	#		end
	#	end
	#	count += 1
	#	v2[1:level_count - 1] = v
	#	get_wa(wa_r_factor, calculate_wa!(X, log_size, l0_count, v2, wa))
	# end

	# gen_g = (i) -> (v, grad) -> begin
	#	if length(grad) > 0
	#		for j = 1:length(grad)
	#			if i == j
	#				grad[j] = 1.
	#			elseif i + 1 == j
	#				grad[j] = -1.
	#			else
	#				grad[j] = 0.
	#			end
	#		end
	#	end
	#	v[i] - v[i + 1]
	# end

	# v = init_sizes[1:end - 1]

	# v_L = [0. for i = 1:level_count - 1]
	# v_U = [2.e19 for i = 1:level_count - 1]

	# opt = Opt(:LN_COBYLA, level_count - 1)
	# # opt = Opt(:LD_MMA, level_count - 1)
	# min_objective!(opt, f)
	# lower_bounds!(opt, v_L)
	# upper_bounds!(opt, v_U)
	# for i = 1:(level_count - 2)
	#	# inequality_constraint!(opt, (v, grad) -> v[i] - v[i + 1])		# <= 0
	#	inequality_constraint!(opt, gen_g(i))                      		# <= 0
	# end
	# # inequality_constraint!(opt, (v, grad) -> v[level_count - 2] - n)		# <= 0
	# ftol_abs!(opt, ftol)
	# maxtime!(opt, max_time)
	# @time (minf, minx, ret) = optimize(opt, v)
	# println("got $minf at $minx after $count iterations (returned $ret)")

	# cat(1, minx, [n])

	#########################

	v2 = Array(Float64, level_count)
	v2[level_count] = n

	count = 0
	wa_r = Array(Float64, 2 + level_count)
	wa_w = Array(Float64, 2 + level_count)

	eval_f = (v) -> begin
		count += 1
		v2[1:level_count - 1] = v
		get_wa(wa_r_factor, calculate_wa!(X, log_size, l0_count, v2, wa_r, wa_w))
	end

	eval_grad_f = (v, grad_f) -> begin
		v2[1:level_count - 1] = v
		y = get_wa(wa_r_factor, calculate_wa!(X, log_size, l0_count, v2, wa_r, wa_w))
		for i = 1:(level_count - 1)
			diff = max(v2[i] * 0.001, 1.)
			org = v2[i]
			v2[i] += diff
			grad_f[i] = (get_wa(wa_r_factor, calculate_wa!(X, log_size, l0_count, v2, wa_r, wa_w)) - y) / diff
			v2[i] = org
		end
	end

	eval_g = (v, g) -> begin
		for i = 1:(level_count - 2)
			g[i] = v[i] - v[i + 1]
		end
	end

	# level i's size - level i+1's size <= 0
	eval_jac_g = (v, mode, rows, cols, values) -> begin
		if mode == :Structure
			c = 1
			for i = 1:level_count - 2
				rows[c] = i
				cols[c] = i
				c += 1
				rows[c] = i
				cols[c] = i + 1
				c += 1
			end
		else
			c = 1
			for i = 1:level_count - 2
				values[c] = 1.
				c += 1
				values[c] = -1.
				c += 1
			end
		end
	end

	v_L = [1. for i = 1:level_count - 1]
	v_U = [Float64(n) for i = 1:level_count - 1]

	g_L = [-2.e19 for i = 1:level_count - 2]
	g_U = [0. for i = 1:level_count - 2]

	prob = createProblem(level_count - 1, v_L, v_U,
						 level_count - 2, g_L, g_U,
						 (level_count - 2) * 2, 0,
						 eval_f, eval_g, eval_grad_f, eval_jac_g)

	addOption(prob, "hessian_approximation", "limited-memory")

	addOption(prob, "tol", ftol)
	addOption(prob, "max_cpu_time", max_time)
	addOption(prob, "acceptable_iter", 1000)

	addOption(prob, "print_level", 2)

	prob.x = init_sizes[1:end - 1]

	@time status = solveProblem(prob)

	ret = Ipopt.ApplicationReturnStatus[status]
	minf = prob.obj_val
	minx = prob.x
	println("got $minf at $minx after $count iterations (returned $ret)")

	cat(1, minx, [n])
end


###########

function calculate_random_compaction_wa!(X::Distribution, log_size::Float64, l0_count::Float64, sizes::Array{Float64}, wa_r::Array{Float64}, wa_w::Array{Float64})
	@assert sizes[end] == Float64(X.count)

	# mem->log
	wa_r[1] = 0.
	wa_w[1] = 1.

	# log->0
	wa_r[2] = 0.
	wa_w[2] = Common.unique(X, log_size) / log_size

	## deamortized, compact each sstable in a random way
	# 0->1
	interval = log_size * l0_count
	next_interval = unique_inv(X, sizes[1])
	wa_r[3] = (Common.unique(X, log_size) * l0_count + sizes[1]) / interval
	wa_w[3] = Common.unique(X, interval + next_interval) / interval
	# 1->2, 2->3, ...
	for i in 1:(length(sizes) - 1)
		# we need to take the previous interval as part of this interval ("interval +")
		# because the current level temporarily has to accommodate the data from the previous level
		interval = interval + next_interval
		if i < length(sizes) - 1
			next_interval = unique_inv(X, sizes[i + 1])
			# plus unique(X, interval) * 1 to WA because of the overlapping tables' keys that do not actually overlap the compaction key range
			# TODO: this may become less accurate with spatial locality in key range because the overlapping tables' key range may be sparse
			wa_r[3 + i] = (Common.unique(X, interval) + sizes[i + 1] + Common.unique(X, interval) * 1.) / interval
			wa_w[3 + i] = (Common.unique(X, interval + next_interval) + Common.unique(X, interval) * 1.) / interval
		else
			wa_r[3 + i] = (Common.unique(X, interval) + sizes[end] + Common.unique(X, interval) * 1.) / interval
			wa_w[3 + i] = (sizes[end] + Common.unique(X, interval) * 1.) / interval
		end
	end

	wa_r, wa_w
end

function calculate_random_compaction_wa(X::Distribution, log_size::Float64, l0_count::Float64, sizes::Array{Float64})
	wa_r = Array(Float64, 2 + length(sizes))
	wa_w = Array(Float64, 2 + length(sizes))
	calculate_random_compaction_wa!(X::Distribution, log_size::Float64, l0_count::Float64, sizes::Array{Float64}, wa_r, wa_w)
end




###########


function calculate_mbuf_wa!(X::Distribution, mbuf_size::Float64, sizes::Array{Float64}, wa_r::Array{Float64}, wa_w::Array{Float64})
	@assert sizes[end] == Float64(X.count)

	# mem->log
	wa_r[1] = 0.
	wa_w[1] = 1.

	# log->mbuf
	wa_r[2] = 0.
	wa_w[2] = 0.

	## deamortized, compact each sstable in a round-robin way
	# mbuf->1
	# no overflow from log because mbuf must be compacted proactively.
	interval = interval_from_density(X, mbuf_size)
	next_interval = unique_inv(X, sizes[1])
	# # however, we have to consider false overlaps since this is now incremental compaction.
	# wa[3] = (Common.unique(X, interval + next_interval) + Common.unique(X, interval) * 1.) / interval
	wa_r[3] = (Common.unique(X, log_size) * l0_count + sizes[1]) / interval
	wa_w[3] = Common.unique(X, interval + next_interval) / interval
	# 1->2, 2->3, ...
	for i in 1:(length(sizes) - 1)
		# we need to take the previous interval as part of this interval ("interval +")
		# because the current level temporarily has to accommodate the data from the previous level
		interval = interval + interval_from_density(X, sizes[i])
		if i < length(sizes) - 1
			next_interval = unique_inv(X, sizes[i + 1])
			# plus unique(X, interval) * 1 to WA because of the overlapping tables' keys that do not actually overlap the compaction key range
			# TODO: this may become less accurate with spatial locality in key range because the overlapping tables' key range may be sparse
			wa_r[3 + i] = (Common.unique(X, interval) + sizes[i + 1] + Common.unique(X, interval) * 1.) / interval
			wa_w[3 + i] = (Common.unique(X, interval + next_interval) + Common.unique(X, interval) * 1.) / interval
		else
			wa_r[3 + i] = (Common.unique(X, interval) + sizes[end] + Common.unique(X, interval) * 1.) / interval
			wa_w[3 + i] = (sizes[end] + Common.unique(X, interval) * 1.) / interval
		end
	end

	wa_r, wa_w
end

function calculate_mbuf_wa(X::Distribution, mbuf_size::Float64, sizes::Array{Float64})
	wa_r = Array(Float64, 2 + length(sizes))
	wa_w = Array(Float64, 2 + length(sizes))
	calculate_mbuf_wa!(X::Distribution, mbuf_size::Float64, sizes::Array{Float64}, wa_r, wa_w)
end

function optimize_mbuf_wa(X::Distribution, mbuf_size::Float64, init_sizes::Array{Float64}, wa_r_factor::Float64, ftol::Float64, max_time::Float64)
	n = X.count
	level_count = length(init_sizes)

	v2 = Array(Float64, level_count)
	v2[level_count] = n

	count = 0
	wa = Array(Float64, 2 + level_count)

	eval_f = (v) -> begin
		count += 1
		v2[1:level_count - 1] = v
		get_wa(wa_r_factor, calculate_mbuf_wa!(X, mbuf_size, v2, wa))
	end

	eval_grad_f = (v, grad_f) -> begin
		v2[1:level_count - 1] = v
		y = get_wa(wa_r_factor, calculate_mbuf_wa!(X, mbuf_size, v2, wa))
		for i = 1:(level_count - 1)
			diff = max(v2[i] * 0.001, 1.)
			org = v2[i]
			v2[i] += diff
			grad_f[i] = (get_wa(wa_r_factor, calculate_mbuf_wa!(X, mbuf_size, v2, wa)) - y) / diff
			v2[i] = org
		end
	end

	eval_g = (v, g) -> begin
		for i = 1:(level_count - 2)
			g[i] = v[i] - v[i + 1]
		end
	end

	# level i's size - level i+1's size <= 0
	eval_jac_g = (v, mode, rows, cols, values) -> begin
		if mode == :Structure
			c = 1
			for i = 1:level_count - 2
				rows[c] = i
				cols[c] = i
				c += 1
				rows[c] = i
				cols[c] = i + 1
				c += 1
			end
		else
			c = 1
			for i = 1:level_count - 2
				values[c] = 1.
				c += 1
				values[c] = -1.
				c += 1
			end
		end
	end

	# v_L = [1. for i = 1:level_count - 1]
	v_L = [mbuf_size for i = 1:level_count - 1]
	v_U = [Float64(n) for i = 1:level_count - 1]

	g_L = [-2.e19 for i = 1:level_count - 2]
	g_U = [0. for i = 1:level_count - 2]

	prob = createProblem(level_count - 1, v_L, v_U,
						 level_count - 2, g_L, g_U,
						 (level_count - 2) * 2, 0,
						 eval_f, eval_g, eval_grad_f, eval_jac_g)

	addOption(prob, "hessian_approximation", "limited-memory")

	addOption(prob, "tol", ftol)
	addOption(prob, "max_cpu_time", max_time)
	addOption(prob, "acceptable_iter", 1000)

	addOption(prob, "print_level", 2)

	prob.x = init_sizes[1:end - 1]

	@time status = solveProblem(prob)

	ret = Ipopt.ApplicationReturnStatus[status]
	minf = prob.obj_val
	minx = prob.x
	println("got $minf at $minx after $count iterations (returned $ret)")

	cat(1, minx, [n])
end






###########



function print(X::Distribution, log_size::Float64, l0_count::Float64, sizes::Array{Float64}, wa_r_factor::Float64)
	level_count = length(sizes)

	println("sizes = ", [iround(v) for v in sizes])
	println("(", [round(sizes[i] / sizes[i - 1] * 100.) / 100. for i in 2:length(sizes)], " X)")
	wa = calculate_wa(X, log_size, l0_count, sizes)
	println("WA (mem->log) = ", wa[2][1])
	println("WA (log->0) = ", wa[2][2])
	for i = 1:level_count; println("WA ($(i-1)->$i) = ", wa[2][i + 2]) end
	println("WA = ", get_wa(wa_r_factor, wa))
end


## TODO: COLA and SAMT should be moved to IntervalModel

# COLA
function calculate_wa_cola!(X::Distribution, log_size::Float64, r::Int64, L::Int64, wa_r::Array{Float64}, wa_w::Array{Float64})
	# mem->log
	wa_r[1] = 0.
	wa_w[1] = 1.

	# mem->1, 1->2, 2->3, ...
	interval = 0.
	next_interval = log_size
	for i in 0:(L - 2)
		interval = next_interval
		next_interval = interval * r
		r_ = 0.
		w = 0.
		# a level accepts merges up to r-1 times.
		# this means that we set r (g in the COLA paper) to be (B^e + 1), which is still in Theta(B^e).
		# choosing r in that way makes the number of levels bounded by O(log_{B^e + 1} N) = O(log_r N),
		# which results in the level count we intend to obtain.
		for j in 0:(r - 2)
			if i == 0
				r_ += Common.unique(X, interval * j)
			else
				r_ += Common.unique(X, interval) + Common.unique(X, interval * j)
			end
			w += Common.unique(X, interval + interval * j)
		end
		wa_r[2 + i] += r_ / next_interval
		wa_w[2 + i] += w / next_interval
	end

	# (L-1)->L
	interval = next_interval
	wa_r[2 + L - 1] = (Common.unique(X, interval) + X.count) / interval
	wa_w[2 + L - 1] = X.count / interval

	wa_r, wa_w
end

function calculate_wa_cola(X::Distribution, log_size::Float64, r::Int64, L::Int64)
	wa_r = Array(Float64, 1 + L)
	wa_w = Array(Float64, 1 + L)
	calculate_wa_cola!(X::Distribution, log_size::Float64, r::Int64, L::Int64, wa_r, wa_w)
end

# SAMT
function calculate_wa_samt!(X::Distribution, log_size::Float64, r::Int64, L::Int64, wa_r::Array{Float64}, wa_w::Array{Float64})
	# mem->log
	wa_r[1] = 0.
	wa_w[1] = 1.

	# mem->1, 1->2, 2->3, ...
	interval = 0.
	next_interval = log_size
	for i in 0:(L - 2)
		interval = next_interval
		next_interval = interval * r
		# a level has r slots to put merges
		# actually, we do not write anything to the last slot because
		# we can merge the level into the next level, which makes
		# COLA and SAMT identical when r = 2
		#wa[2 + i] = ((r - 1) * Common.unique(X, interval)) / next_interval
		# but we choose to do maintain full r slots because the SAMT paper seems to intend it.
		# this makes SAMT more expensive (and wasteful) than COLA with r = 2.
		# however, SAMT usually uses r = 4, and the compaction only needs to do up to r-way merge
		# (not ((r-1)^l)-way in COLA), which makes more sense in a practical standpoint.
		wa_r[2 + i] = (r * Common.unique(X, interval)) / next_interval
		wa_w[2 + i] = (r * Common.unique(X, interval)) / next_interval
	end

	# (L-1)->L
	interval = next_interval
	wa_r[2 + L - 1] = (r * Common.unique(X, interval) + X.count) / interval
	wa_w[2 + L - 1] = X.count / interval

	wa_r, wa_w
end

function calculate_wa_samt(X::Distribution, log_size::Float64, r::Int64, L::Int64)
	wa_r = Array(Float64, 1 + L)
	wa_w = Array(Float64, 1 + L)
	calculate_wa_samt!(X::Distribution, log_size::Float64, r::Int64, L::Int64, wa_r, wa_w)
end

###########


# original SILT with major compaction from HashStore to SortedStore
function calculate_wa_silt!(X::Distribution, hash_size::Float64, hash_occupancy::Float64, hash_count::Int64, wa_r::Array{Float64}, wa_w::Array{Float64})
	convert_interval = unique_inv(X, hash_size * hash_occupancy)

	# TODO: wa_r

	# mem->log store
	wa_r[1] = 0.
	wa_w[1] = 1.

	# log store->hash store
	wa_r[2] = 0.
	wa_w[2] = hash_size / convert_interval

	# hash stores->sorted store
	wa_r[3] = 0.
	wa_w[3] = X.count / (convert_interval * hash_count)

	wa_r, wa_w
end

function calculate_wa_silt(X::Distribution, log_size::Float64, hash_occupancy::Float64, hash_count::Int64)
	wa = Array(Float64, 3)
	calculate_wa_silt!(X::Distribution, log_size::Float64, hash_occupancy::Float64, hash_count::Int64, wa)
end


# SILT + minor compaction among HashStore; assume any size of HashStore can be created
function calculate_wa_silt_multi!(X::Distribution, hash_size::Float64, hash_occupancy::Float64, hash_count::Int64, hash_threshold::Float64, wa_r::Array{Float64}, wa_w::Array{Float64})
	convert_interval = unique_inv(X, hash_size * hash_occupancy)

	# TODO: wa_r

	# the interval of minor compaction
	minor_compaction_interval = convert_interval * hash_count

	# the number of minor compaction to trigger major compaction; the last minor compaction does not actually write data
	minor_compaction_count = floor(unique_inv(X, X.count * hash_threshold) / minor_compaction_interval)
	@assert minor_compaction_count >= 1.0

	# the interval of major compaction
	major_compaction_interval = minor_compaction_interval * minor_compaction_count

	# mem->log
	wa_r[1] = 0.
	wa_w[1] = 1.

	# log store->hash store
	wa_r[2] = 0.
	wa_w[2] = hash_size / convert_interval

	# hash stores->hash store
	wa_r[3] = 0.
	wa_w[3] = 0.
	if minor_compaction_count >= 2
		for j in 0:(minor_compaction_count - 2)
			wa_w[3] += (Common.unique(X, minor_compaction_interval + minor_compaction_interval * j) / hash_occupancy) / major_compaction_interval
		end
	end

	# hash stores->sorted
	wa_r[4] = 0.
	wa_w[4] = X.count / major_compaction_interval

	wa_r, wa_w
end

function calculate_wa_silt_multi(X::Distribution, hash_size::Float64, hash_occupancy::Float64, hash_count::Int64, hash_threshold::Float64)
	wa = Array(Float64, 4)
	calculate_wa_silt_multi!(X::Distribution, hash_size::Float64, hash_occupancy::Float64, hash_count::Int64, hash_threshold::Float64, wa)
end


end


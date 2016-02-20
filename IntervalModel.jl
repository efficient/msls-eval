module IntervalModel

using Common
# using NLopt
using Ipopt

function init_intervals(log_size::Float64, l0_count::Float64, level_count::Int64)
	interval = Array(Float64, 0)
	for i = 1:level_count
		# push!(interval, log_size * l0_count)
		push!(interval, log_size * l0_count * (10. ^ Float64(i - 1)))
	end
	interval
end

function calculate_wa_twolevel!(X::Distribution, log_size::Float64, l0_count::Float64, intervals::Array{Float64}, wa_r::Array{Float64}, wa_w::Array{Float64})
	# mem->log
	wa_r[1] = 0.
	wa_w[1] = 1.

	# log->0
	wa_r[2] = 0.
	wa_w[2] = Common.unique(X, log_size) / log_size

	# ## amortized, full destination level
	# # 0->1, 1->2, ...
	# for i in 1:(length(intervals) - 1)
	#	wa[2 + i] = Common.unique(X, intervals[i] + intervals[i + 1]) / intervals[i]
	# end
	# wa[2 + length(intervals)] = Float64(X.count) / intervals[end]

	# ## amortized, compact entire level
	# # 0->1, 1->2, ...
	# interval = 0.
	# next_interval = intervals[1]
	# for i in 1:(length(intervals) - 1)
	#	interval = interval * 0.5 + next_interval
	#	next_interval = intervals[i + 1]
	#	wa[2 + i] = unique_avg(X, interval, interval * 0.5 + next_interval) / interval
	# end
	# interval = interval * 0.5 + next_interval
	# wa[2 + length(intervals)] = Float64(X.count) / interval

	## deamortized, compact each sstable in a round-robin way
	# 0->1, 1->2, ...
	interval = 0.
	next_interval = intervals[1]
	for i in 1:(length(intervals) - 1)
		if i == 1
			# 0->1 compaction is usually a whole level
			# do not use interval_from_density() and adding extra unique() to WA that are caused by using small tables
			interval = next_interval
			next_interval = intervals[i + 1]
			wa_r[2 + i] = (Common.unique(X, log_size) * l0_count + Common.unique(X, next_interval)) / interval
			wa_w[2 + i] = Common.unique(X, interval + next_interval) / interval
		else
			interval = interval + interval_from_density(X, Common.unique(X, next_interval))
			next_interval = intervals[i + 1]
			# using additional unique(); see SizeMode.jl for details
			wa_r[2 + i] = (Common.unique(X, interval) + Common.unique(X, next_interval) + Common.unique(X, interval) * 1.) / interval
			wa_w[2 + i] = (Common.unique(X, interval + next_interval) + Common.unique(X, interval) * 1.) / interval
		end
	end
	interval = interval + interval_from_density(X, Common.unique(X, next_interval))
	# using additional unique(); see SizeMode.jl for details
	wa_r[2 + length(intervals)] = (Common.unique(X, interval) + Float64(X.count) + Common.unique(X, interval) * 1.) / interval
	wa_w[2 + length(intervals)] = (Float64(X.count) + Common.unique(X, interval) * 1.) / interval

	wa_r, wa_w
end

function calculate_wa_twolevel(X::Distribution, log_size::Float64, l0_count::Float64, intervals::Array{Float64})
	wa_r = Array(Float64, 2 + length(intervals))
	wa_w = Array(Float64, 2 + length(intervals))
	calculate_wa_twolevel!(X::Distribution, log_size::Float64, l0_count::Float64, intervals::Array{Float64}, wa_r, wa_w)
end

# function calculate_wa_twolevel_ratios(X::Distribution, log_size::Float64, l0_count::Float64, interval_ratios::Array{Float64})
#	current_interval = interval_ratios[1]
#	intervals = interval_ratios * (log_size * l0_count / current_interval)
#	return calculate_wa_twolevel(X, log_size, l0_count, intervals)
# end

function calculate_sizes_twolevel(X::Distribution, log_size::Float64, l0_count::Float64, intervals::Array{Float64})
	sizes = Array(Float64, length(intervals))

	for i in 1:(length(intervals) - 1)
		sizes[i] = Common.unique(X, intervals[i + 1])
	end
	sizes[length(intervals)] = Float64(X.count)

	sizes
end

function calculate_wa_multilevel!(X::Distribution, log_size::Float64, l0_count::Float64, intervals::Array{Float64}, wa_r::Array{Float64}, wa_w::Array{Float64})
	# TODO: wa_r

	# mem->log
	wa_r[1] = 0.
	wa_w[1] = 1.

	# log->0
	wa_r[2] = 0.
	wa_w[2] = Common.unique(X, log_size) / log_size

	# ## amortized, full destination level
	# # 0->1, 1->2, ...
	# for i in 1:(length(intervals) - 1)
	#	# level-0...1 size, level-0...2 size, ...
	#	level_size = Common.unique(X, geom_mean(intervals[(i + 1):end]))
	#	wa[2 + i] = level_size / intervals[i]
	# end
	# wa[2 + length(intervals)] = Float64(X.count) / intervals[end]

	## amortized, compact entire level (TODO: do we need to modify interval to consider "0.5" factor?)
	# 0->1, 1->2, ...
	# interval = 0.
	# next_interval = geom_mean(intervals)
	for i in 1:(length(intervals) - 1)
		wa_r[2 + i] = 0.
		wa_w[2 + i] = unique_avg(X, geom_mean(intervals[i:end]), geom_mean(intervals[i:end]) * 0.5 + geom_mean(intervals[(i + 1):end])) / intervals[i]
		# interval = interval * 0.5 + next_interval
		# next_interval = geom_mean(intervals[(i + 1):end])
		# wa[2 + i] = unique_avg(X, interval, interval * 0.5 + next_interval) / interval
	end
	wa_r[2 + length(intervals)] = 0.
	wa_w[2 + length(intervals)] = Float64(X.count) / intervals[end]

	wa_r, wa_w
end

function calculate_wa_multilevel(X::Distribution, log_size::Float64, l0_count::Float64, intervals::Array{Float64})
	wa_r = Array(Float64, 2 + length(intervals))
	wa_w = Array(Float64, 2 + length(intervals))
	calculate_wa_multilevel!(X::Distribution, log_size::Float64, l0_count::Float64, intervals::Array{Float64}, wa_r, wa_w)
end

# function calculate_wa_multilevel_ratios(X::Distribution, log_size::Float64, l0_count::Float64, interval_ratios::Array{Float64})
#	current_interval = geom_mean(interval_ratios)
#	intervals = interval_ratios * (log_size * l0_count / current_interval)
#	return calculate_wa_multilevel(X, log_size, l0_count, intervals)
# end

function calculate_sizes_multilevel(X::Distribution, log_size::Float64, l0_count::Float64, intervals::Array{Float64})
	sizes = Array(Float64, length(intervals))

	for i in 1:(length(intervals) - 1)
		sizes[i] = Common.unique(X, geom_mean(intervals[i + 1:end]))
	end
	sizes[length(intervals)] = Float64(X.count)

	sizes
end

function optimize_wa_twolevel(X::Distribution, log_size::Float64, l0_count::Float64, init_intervals::Array{Float64}, wa_r_factor::Float64, ftol::Float64, max_time::Float64)
	n = X.count
	level_count = length(init_intervals)

	# v2 = Array(Float64, level_count)
	# v2[1] = log_size * l0_counum

	# count = 0
	# wa_r = Array(Float64, 2 + level_count)
	# wa_w = Array(Float64, 2 + level_count)
	# f = (v, grad) -> begin
	#	count += 1
	#	v2[2:level_count] = v
	#	get_wa(wa_r_factor, calculate_wa_twolevel!(X, log_size, l0_count, v2, wa_r, wa_w))
	# end

	# v = init_intervals[2:end]

	# opt = Opt(:LN_COBYLA, level_count - 1)
	# min_objective!(opt, f)
	# # inequality_constraint!(opt, (v, grad) -> log_size * l0_count - v[1])		# <= 0
	# for i = 1:(level_count - 2)
	#	inequality_constraint!(opt, (v, grad) -> v[i] - v[i + 1])		# <= 0
	# end
	# ftol_abs!(opt, ftol)
	# maxtime!(opt, max_time)
	# @time (minf, minx, ret) = optimize(opt, v)
	# println("got $minf at $minx after $count iterations (returned $ret)")

	# cat(1, [log_size * l0_count], minx)

	#######################

	v2 = Array(Float64, level_count)
	v2[1] = log_size * l0_count

	count = 0
	wa_r = Array(Float64, 2 + level_count)
	wa_w = Array(Float64, 2 + level_count)

	eval_f = (v) -> begin
		count += 1
		v2[2:level_count] = v
		get_wa(wa_r_factor, calculate_wa_twolevel!(X, log_size, l0_count, v2, wa_r, wa_w))
	end

	eval_grad_f = (v, grad_f) -> begin
		v2[2:level_count] = v
		y = get_wa(wa_r_factor, calculate_wa_twolevel!(X, log_size, l0_count, v2, wa_r, wa_w))
		for i = 2:level_count
			diff = max(v2[i] * 0.001, 1.)
			org = v2[i]
			v2[i] += diff
			grad_f[i - 1] = (get_wa(wa_r_factor, calculate_wa_twolevel!(X, log_size, l0_count, v2, wa_r, wa_w)) - y) / diff
			v2[i] = org
		end
	end

	eval_g = (v, g) -> begin
		for i = 1:(level_count - 2)
			g[i] = v[i] - v[i + 1]
		end
	end

	# level i's interval - level i+1's interval <= 0
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

	v_L = [log_size * l0_count for i = 1:level_count - 1]
	v_U = [2.e19 for i = 1:level_count - 1]

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

	prob.x = init_intervals[2:end]

	@time status = solveProblem(prob)

	ret = Ipopt.ApplicationReturnStatus[status]
	minf = prob.obj_val
	minx = prob.x
	println("got $minf at $minx after $count iterations (returned $ret)")

	cat(1, [log_size * l0_count], minx)
end

function optimize_wa_multilevel(X::Distribution, log_size::Float64, l0_count::Float64, init_intervals::Array{Float64}, wa_r_factor::Float64, ftol::Float64, max_time::Float64)
	n = X.count
	level_count = length(init_intervals)

	# v2 = Array(Float64, level_count)

	# count = 0
	# wa_r = Array(Float64, 2 + level_count)
	# wa_w = Array(Float64, 2 + level_count)
	# f = (v, grad) -> begin
	#	count += 1
	#	# we need to make geom_mean(cat(1, [X], v)) = log_size * l0_count
	#	# 1/X + ..  = 1 / (log_size * l0_count)
	#	# 1/X = 1 / (log_size * l0_count) - ...
	#	# X = 1 / (1 / (log_size * l0_count) - ...)
	#	#   = geom_mean(cat(1, [log_size * l0_count], -v))

	#	# v2[1] = geom_mean(cat(1, [log_size * l0_count], -v))

	#	v2[1] = -(log_size * l0_count)
	#	v2[2:level_count] = v
	#	v2[1] = -geom_mean(v2)
	#	get_wa(wa_r_factor, calculate_wa_multilevel!(X, log_size, l0_count, v2, wa_r, wa_w))
	# end

	# v = init_intervals[2:end]

	# opt = Opt(:LN_COBYLA, level_count - 1)
	# min_objective!(opt, f)
	# for i = 1:(level_count - 2)
	#	inequality_constraint!(opt, (v, grad) -> v[i] - v[i + 1])		# <= 0
	# end
	# ftol_abs!(opt, ftol)
	# maxtime!(opt, max_time)
	# @time (minf, minx, ret) = optimize(opt, v)
	# println("got $minf at $minx after $count iterations (returned $ret)")

	# x = geom_mean(cat(1, [log_size * l0_count], -minx))
	# cat(1, [x], minx)

	#######################

	v2 = Array(Float64, level_count)
	v2[1] = log_size * l0_count

	count = 0
	wa_r = Array(Float64, 2 + level_count)
	wa_w = Array(Float64, 2 + level_count)

	eval_f = (v) -> begin
		count += 1
		v2[1] = -(log_size * l0_count)
		v2[2:level_count] = v
		v2[1] = -geom_mean(v2)
		# note that v2[1] can become negative accidentally, which is not valid for unique()
		get_wa(wa_r_factor, calculate_wa_multilevel!(X, log_size, l0_count, v2, wa_r, wa_w))
	end

	eval_grad_f = (v, grad_f) -> begin
		v2[1] = -(log_size * l0_count)
		v2[2:level_count] = v
		v2[1] = -geom_mean(v2)
		y = get_wa(wa_r_factor, calculate_wa_multilevel!(X, log_size, l0_count, v2, wa_r, wa_w))
		for i = 2:level_count
			diff = max(v2[i] * 0.001, 1.)
			org = v2[i]
			v2[i] += diff
			v2[1] = -(log_size * l0_count)
			v2[1] = -geom_mean(v2)
			grad_f[i - 1] = (get_wa(wa_r_factor, calculate_wa_multilevel!(X, log_size, l0_count, v2, wa_r, wa_w)) - y) / diff
			v2[i] = org
		end
	end

	eval_g = (v, g) -> begin
		for i = 1:(level_count - 2)
			g[i] = v[i] - v[i + 1]
		end
	end

	# level i's interval - level i+1's interval <= 0
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
				values[c] = 1
				c += 1
				values[c] = -1
				c += 1
			end
		end
	end

	v_L = [log_size * l0_count for i = 1:level_count - 1]
	v_U = [2.e19 for i = 1:level_count - 1]

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

	prob.x = init_intervals[2:end]

	@time status = solveProblem(prob)

	ret = Ipopt.ApplicationReturnStatus[status]
	minf = prob.obj_val
	minx = prob.x
	println("got $minf at $minx after $count iterations (returned $ret)")

	x = geom_mean(cat(1, [log_size * l0_count], -minx))
	cat(1, [x], minx)
end

function print_twolevel(X::Distribution, log_size::Float64, l0_count::Float64, intervals::Array{Float64}, wa_r_factor::Float64)
	level_count = length(intervals)

	println("intervals = ", [iround(v) for v in intervals])
	println("exp. size = ", [iround(Common.unique(X, v)) for v in intervals])
	println("(", [round(intervals[i] / intervals[i - 1] * 100.) / 100. for i in 2:length(intervals)], " X)")
	wa = calculate_wa_twolevel(X, log_size, l0_count, intervals)
	println("WA (mem->log) = ", wa[2][1])
	println("WA (log->0) = ", wa[2][2])
	for i = 1:level_count; println("WA ($(i-1)->$i) = ", wa[2][i + 2]) end
	println("WA = ", get_wa(wa_r_factor, wa))
end

function print_multilevel(X::Distribution, log_size::Float64, l0_count::Float64, intervals::Array{Float64}, wa_r_factor::Float64)
	level_count = length(intervals)

	println("intervals = ", [iround(v) for v in intervals])
	println("(", [round(intervals[i] / intervals[i - 1] * 100.) / 100. for i in 2:length(intervals)], " X)")
	wa = calculate_wa_multilevel(X, log_size, l0_count, intervals)
	println("avg L0 intervals = ", iround(geom_mean(intervals)))
	println("WA (mem->log) = ", wa[2][1])
	println("WA (log->0) = ", wa[2][2])
	for i = 1:level_count; println("WA ($(i-1)->$i) = ", wa[2][i + 2]) end
	println("WA = ", get_wa(wa_r_factor, wa))
end

end

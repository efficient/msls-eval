module Analysis

using Common
using SizeModel
using IntervalModel

function test_dist()
	X = Distribution(4, Array(Float64, 0), Array(Float64, 0), Array(Float64, 0), Array(Float64, 0))
	X.p = [0.1, 0.2, 0.3, 0.4]
	# X.p = [1. / float64(X.count) for i = 1:X.count]
	X.c = [1. for i = 1:X.count]
	check_validity(X)
	update_derived_values(X)

	# println(Common.unique(X, 5.77))
	println(unique_inv(X, 3.))
	println(ccp(X, 3))
	# println(ccp(X, 4))

	# println(unique_inv(X, 19.))
	# println(ccp(X, 19))
end

function test_basic()
	# n = unique key count
	# s = zipf skew
	n = 2 * 1000 * 1000
	# n = 10 * 1000 * 1000
	# n = 100 * 1000 * 1000
	# n = 1000 * 1000 * 1000
	# s = 0.    # uniform
	s = 0.99    # zipf
	# s = 1.		# zipf, fast generation
	println("n = ", n)
	println("s = ", s)

	# @time X = zipf(n, s)
	# println("X.c = ", length(X.c))
	# @time X = compress(X, 0.1)
	# println("X.c = ", length(X.c), " (after compression)")
	@time X = load_zipf_compressed(n, s, 0.1)
	# @time X = load_zipf_compressed(n, s, 0.001)
	println("X.c = ", length(X.c), " (after compression)")
	println()

	@time update_derived_values(X)


	# ## basic
	# println("== basic")
	# println("unique(X, 4000.) = ", Common.unique(X, 4000.))
	# println()


	# ## fill time
	# println("== fill time")
	# println("unique_inv(X, 10000.) = ", unique_inv(X, 10000.))
	# println("unique(unique_inv(X, 10000.)) = ", Common.unique(X, unique_inv(X, 10000.)))
	# println("unique_inv(X, 100000.) = ", unique_inv(X, 100000.))
	# println("unique(unique_inv(X, 100000.)) = ", Common.unique(X, unique_inv(X, 100000.)))
	# println("unique_inv(X, 1000000.) = ", unique_inv(X, 1000000.))
	# println("unique(unique_inv(X, 1000000.)) = ", Common.unique(X, unique_inv(X, 1000000.)))
	# println()
end

function test_wa_estimation()
	# #### WA estimation
	# println("== WA estimation")

	# # log size, level 0 table count
	# log_size = 4. * 1048576. / 1000
	# l0_count = 4.
	# println("log_size = ", log_size);
	# println("l0_count = ", l0_count);
	# println()

	# ## size-based compaction
	# println("= size-based")
	# sizes = [100000., 1000000., float64(n)]
	# # sizes = [30000., 200000., float64(n)]
	# println("sizes = ", sizes);
	# wa = LevelDB_WA(X, log_size, l0_count, sizes)
	# println("WA (log->0) = ", wa[1])
	# println("WA (0->1) = ", wa[2])
	# println("WA (1->2) = ", wa[3])
	# println("WA (2->3) = ", wa[4])
	# println("WA = ", sum(wa))

	# ## interval_based multi-level compaction
	# println("= interval_based multi-level")

	# # Level 1, 2, 3 compaction interval ratios (not normalized)
	# compact_interval_ratio = [18., 130., 1000.]
	# current_compact_interval = geom_mean(compact_interval_ratio)

	# compact_interval = compact_interval_ratio * (log_size * l0_count / current_compact_interval)
	# println("compact_interval = ", compact_interval)
	# println("avg_compact_interval = ", geom_mean(compact_interval))

	# wa = ProbMultiLevel_WA(X, log_size, l0_count, compact_interval)
	# println("WA (log->0) = ", wa[1])
	# println("WA (0->1) = ", wa[2])
	# println("WA (1->2) = ", wa[3])
	# println("WA (2->3) = ", wa[4])
	# println("WA = ", sum(wa))
end

function test_wa_optimization()
	#### WA optimization
	println("== WA optimization")

	# ftol = 1.e-8
	# max_time = 1.
	ftol = 1.e-16
	max_time = 5.
	#max_time = 10.

	# log size, level 0 table count, total level count (besides level-0)
	item_size = 1000.
	l1_size = 10. * 1048576. / item_size
	log_size = 4. * 1048576. / item_size
	l0_count = 4.
	# l0_count = 8.
	# l0_count = 12.
	println("log_size = ", log_size);
	println("l0_count = ", l0_count);
	println()

	# println("=========")
	# SizeModel.print(X, log_size, l0_count, [28500., 90856., 278944., 802296., 2000000.])
	# println()
	# IntervalModel.print_twolevel(X, log_size, l0_count, [log_size * l0_count, unique_inv(X, 28500.), unique_inv(X, 90856.), unique_inv(X, 278944.), unique_inv(X, 802296.)])
	# println("=========")
	# println()

	results_wa = []
	results_params = []

	# growth_factors = [3.]
	# level_counts = [5]
	growth_factors = [3., 4., 5., 10., 20.]
	level_counts = [3, 4, 5, 6, 7, 8]

	# ## size-based compaction - fixed (LevelDB)
	# println("= size-based - fixed")
	# for growth_factor in growth_factors
	#	sizes = SizeModel.init_sizes(X, l1_size, growth_factor)
	#	wa = SizeModel.calculate_wa(X, log_size, l0_count, sizes)
	#   wa = ([round(v, 2) for v in wa[1]], [round(v, 2) for v in wa[2]])
	#   wa_sum = get_wa(wa_r_factor, wa)
	#	push!(results_wa, ["size_fixed_wa", growth_factor, wa_sum, wa])
	#	push!(results_params, ["size_fixed_sizes", growth_factor, sizes])
	#	SizeModel.print(X, log_size, l0_count, sizes)
	# end
	# println()

	# ## size-based compaction - flexible
	# println("= size-based - flexible")
	# for level_count in level_counts
	#	sizes = SizeModel.init_sizes(X, l1_size, 0., level_count)
	#	sizes = SizeModel.optimize_wa(X, log_size, l0_count, sizes, wa_r_factor, ftol, max_time)
	#	wa = SizeModel.calculate_wa(X, log_size, l0_count, sizes)
	#   wa = ([round(v, 2) for v in wa[1]], [round(v, 2) for v in wa[2]])
	#   wa_sum = get_wa(wa_r_factor, wa)
	#	push!(results_wa, ["size_flexible_wa", level_count, wa_sum, wa])
	#	push!(results_params, ["size_flexible_sizes", level_count, sizes])
	#	SizeModel.print(X, log_size, l0_count, sizes)
	# end
	# println()

	# ## interval_based two-level compaction
	# println("= interval-based two-level")
	# for level_count in level_counts
	#	intervals = IntervalModel.init_intervals(log_size, l0_count, level_count)
	#	# intervals = [log_size * l0_count, unique_inv(X, sizes[1]), unique_inv(X, sizes[2]), unique_inv(X, sizes[3]), unique_inv(X, sizes[4])]
	#	intervals = IntervalModel.optimize_wa_twolevel(X, log_size, l0_count, intervals, wa_r_factor, ftol, max_time)
	#	wa = IntervalModel.calculate_wa_twolevel(X, log_size, l0_count, intervals)
	#   wa = ([round(v, 2) for v in wa[1]], [round(v, 2) for v in wa[2]])
	#   wa_sum = get_wa(wa_r_factor, wa)
	#	push!(results_wa, ["interval_two_wa", level_count, wa_sum, wa])
	#	push!(results_params, ["interval_two_intervals", level_count, intervals])
	#	IntervalModel.print_twolevel(X, log_size, l0_count, intervals)
	# end
	# println()

	# ## interval_based multi-level compaction
	# println("= interval-based multi-level")
	# for level_count in level_counts
	#	intervals = IntervalModel.init_intervals(log_size, l0_count, level_count)
	#	intervals = IntervalModel.optimize_wa_multilevel(X, log_size, l0_count, intervals, wa_r_factor, ftol, max_time)
	#	wa = IntervalModel.calculate_wa_multilevel(X, log_size, l0_count, intervals)
	#   wa = ([round(v, 2) for v in wa[1]], [round(v, 2) for v in wa[2]])
	#   wa_sum = get_wa(wa_r_factor, wa)
	#	push!(results_wa, ["interval_multi_wa", level_count, wa_sum, wa])
	#	push!(results_params, ["interval_multi_intervals", level_count, intervals])
	#	IntervalModel.print_multilevel(X, log_size, l0_count, intervals)
	# end
	# println()
end

#function sensitivity_1()
#	println("== sensitivity")
#
#	results_sensitivity = []
#
#	# wa_r_factor = 0.7
#	wa_r_factor = 0.0
#
#	println("= sensitivity - item size")
#	item_sizes = [10, 100, 1000, 10000, 100000]
#	for item_size in item_sizes
#		n = 2 * 1024 * 1024 * 1024 / item_size
#		s = 0.99    # zipf
#		X = load_zipf_compressed(iround(n), s, 0.1)
#		update_derived_values(X)
#
#		l1_size = 10. * 1048576. / item_size
#		log_size = 4. * 1048576. / item_size
#		l0_count = 4.
#
#		sizes = SizeModel.init_sizes(X, l1_size, 10.)
#		wa = SizeModel.calculate_wa(X, log_size, l0_count, sizes)
#		wa = ([round(v, 2) for v in wa[1]], [round(v, 2) for v in wa[2]])
#		wa_sum = get_wa(wa_r_factor, wa)
#		push!(results_sensitivity, ["sensitivity_item_size_wa", item_size, wa_sum, wa])
#	end
#	println()
#
#	n = 2 * 1000 * 1000
#	s = 0.99    # zipf
#	X = load_zipf_compressed(n, s, 0.1)
#	update_derived_values(X)
#
#	item_size = 1000.
#	l1_size = 10. * 1048576. / item_size
#	log_size = 4. * 1048576. / item_size
#
#	println("= sensitivity - l0 count")
#	l0_counts = [4., 8., 12.]
#	for l0_count in l0_counts
#		sizes = SizeModel.init_sizes(X, l1_size, 10.)
#		wa = SizeModel.calculate_wa(X, log_size, l0_count, sizes)
#		wa = ([round(v, 2) for v in wa[1]], [round(v, 2) for v in wa[2]])
#		wa_sum = get_wa(wa_r_factor, wa)
#		push!(results_sensitivity, ["sensitivity_l0_count_wa", l0_count, wa_sum, wa])
#	end
#	println()
#
#
#	l0_count = 4.
#
#	println("= sensitivity - level size")
#	level_count = 3
#	mod_level = 1	# level 1, 2
#    exps = [i for i = -0.8:0.08:0.8]
#
#	sizes = SizeModel.init_sizes(X, l1_size, 0., level_count)
#	sizes_opt = SizeModel.optimize_wa(X, log_size, l0_count, sizes, wa_r_factor, ftol, max_time)
#    for exp1 in exps
#        for exp2 in exps
#			sizes = copy(sizes_opt)
#			sizes[mod_level] *= 10. ^ exp1
#			sizes[mod_level + 1] *= 10. ^ exp2
#			if mod_level - 1 >= 1 && sizes[mod_level - 1] > sizes[mod_level]
#				continue
#            elseif sizes[mod_level] > sizes[mod_level + 1]
#                continue
#            elseif mod_level + 2 <= level_count && sizes[mod_level + 1] > sizes[mod_level + 2]
#                continue
#			end
#			wa = SizeModel.calculate_wa(X, log_size, l0_count, sizes)
#			wa = ([round(v, 2) for v in wa[1]], [round(v, 2) for v in wa[2]])
#			wa_sum = get_wa(wa_r_factor, wa)
#			#push!(results_sensitivity, ["sensitivity_level_size_wa", ratio1, ratio2, wa_sum, wa])
#			push!(results_sensitivity, ["sensitivity_level_size_wa", exp1, exp2, wa_sum, wa])
#		end
#	end
#
#	write_output("output.txt", cat(1, results_wa, results_params, results_sensitivity))
#end

function write_output(filename, rows)
	f = open(filename, "w")
	for row in rows
		for i = 1:length(row)
			if typeof(row[i]) == Float64
				@printf(f, "%5.2f", row[i])
			else
				print(f, row[i])
			end
			if i < length(row)
				print(f, "\t")
			end
		end
		println(f)
	end
	close(f)
end

function write_output_more_digits(filename, rows)
	f = open(filename, "w")
	for row in rows
		for i = 1:length(row)
			if typeof(row[i]) == Float64
				@printf(f, "%5.4f", row[i])
			else
				print(f, row[i])
			end
			if i < length(row)
				print(f, "\t")
			end
		end
		println(f)
	end
	close(f)
end

function unique_data()
	results = []

	eval_f = (results, prefix, n, s) -> begin
		X = load_zipf_compressed(n, s, 0.1)
		update_derived_values(X)

		c = 1.
		count = 0
		while true
			u = Common.unique(X, c)
			if c == Inf || count > 1000
				break
			end
			push!(results, ["$(prefix)_unique", n, s, c, u])
			c = min(c * 1.2, unique_inv(X, u + float64(n) / 100.))
			count += 1
		end
	end

	#for n in [1000000, 3300000, 10 * 1000000, 33 * 1000000, 100 * 1000000, 330 * 1000000, 1000 * 1000000]
	#	for s in [0.00, 0.99]
	#		eval_f(results, "unique_item_count", n, s)
	#	end
	#end

	for n in [100 * 1000000]
		#for s in [0.00, 0.20, 0.40, 0.60, 0.80, 0.99, 1.20, 1.40, 1.60, 1.80, 2.00]
		for s in [0.00, 0.20, 0.40, 0.60, 0.80, 0.99, 1.20, 1.40, 1.60]
			eval_f(results, "unique_skewness", n, s)
		end
	end

	write_output("output_unique.txt", results)
end

function sensitivity_2()
	results = []

	l0_count = 4.
	# l0_count = 8.
	# l0_count = 12.

	# wa_r_factor = 0.7
	wa_r_factor = 0.0

	ftol = 1.e-16
	max_time = 300.

	eval_f = (results, prefix, n, s, item_size, level_count, log_size) -> begin
		println(prefix, ", n=", n, ", s=", s, ", item_size=", item_size, ", level_count=", level_count, ", log_size=", log_size)
		l1_size = 10. * 1048576. / item_size
		# log_size = 4. * 1048576. / item_size

		X = load_zipf_compressed(n, s, 0.1)
		update_derived_values(X)

		println("  leveldb")
		if level_count == 0
			sizes0 = SizeModel.init_sizes(X, l1_size, 10.)
			level_count = length(sizes0)
		else
			sizes0 = SizeModel.init_sizes(X, l1_size, 0., level_count)
		end
		wa0 = SizeModel.calculate_wa(X, log_size, l0_count, sizes0)
		wa0 = ([round(v, 2) for v in wa0[1]], [round(v, 2) for v in wa0[2]])
		wa0_sum = get_wa(wa_r_factor, wa0)
		push!(results, ["$(prefix)_leveldb_wa_r", n, s, round(Int, log_size * item_size), level_count, sum(wa0[1]), wa0[1]])
		push!(results, ["$(prefix)_leveldb_wa_w", n, s, round(Int, log_size * item_size), level_count, sum(wa0[2]), wa0[2]])
		push!(results, ["$(prefix)_leveldb_wa", n, s, round(Int, log_size * item_size), level_count, wa0_sum])
		push!(results, ["$(prefix)_leveldb_sizes", n, s, round(Int, log_size * item_size), level_count, sizes0])

		wa0_rc = SizeModel.calculate_random_compaction_wa(X, log_size, l0_count, sizes0)
		wa0_rc = ([round(v, 2) for v in wa0_rc[1]], [round(v, 2) for v in wa0_rc[2]])
		wa0_rc_sum = get_wa(wa_r_factor, wa0_rc)
		push!(results, ["$(prefix)_leveldb_rc_wa_r", n, s, round(Int, log_size * item_size), level_count, sum(wa0_rc[1]), wa0_rc[1]])
		push!(results, ["$(prefix)_leveldb_rc_wa_w", n, s, round(Int, log_size * item_size), level_count, sum(wa0_rc[2]), wa0_rc[2]])
		push!(results, ["$(prefix)_leveldb_rc_wa", n, s, round(Int, log_size * item_size), level_count, wa0_rc_sum])

		println("  leveldb_opt")
		sizes1 = SizeModel.optimize_wa(X, log_size, l0_count, sizes0, wa_r_factor, ftol, max_time)
		wa1 = SizeModel.calculate_wa(X, log_size, l0_count, sizes1)
		wa1 = ([round(v, 2) for v in wa1[1]], [round(v, 2) for v in wa1[2]])
		wa1_sum = get_wa(wa_r_factor, wa1)
		push!(results, ["$(prefix)_leveldb_opt_wa_r", n, s, round(Int, log_size * item_size), level_count, sum(wa1[1]), wa1[1]])
		push!(results, ["$(prefix)_leveldb_opt_wa_w", n, s, round(Int, log_size * item_size), level_count, sum(wa1[2]), wa1[2]])
		push!(results, ["$(prefix)_leveldb_opt_wa", n, s, round(Int, log_size * item_size), level_count, wa1_sum])
		push!(results, ["$(prefix)_leveldb_opt_sizes", n, s, round(Int, log_size * item_size), level_count, sizes1])

		println("  twolevel_opt")
		intervals2 = IntervalModel.init_intervals(log_size, l0_count, level_count)
		intervals2 = IntervalModel.optimize_wa_twolevel(X, log_size, l0_count, intervals2, wa_r_factor, ftol, max_time)
		wa2 = IntervalModel.calculate_wa_twolevel(X, log_size, l0_count, intervals2)
		wa2 = ([round(v, 2) for v in wa2[1]], [round(v, 2) for v in wa2[2]])
		wa2_sum = get_wa(wa_r_factor, wa2)
		sizes2 = IntervalModel.calculate_sizes_twolevel(X, log_size, l0_count, intervals2)
		push!(results, ["$(prefix)_twolevel_opt_wa_r", n, s, round(Int, log_size * item_size), level_count, sum(wa2[1]), wa2[1]])
		push!(results, ["$(prefix)_twolevel_opt_wa_w", n, s, round(Int, log_size * item_size), level_count, sum(wa2[2]), wa2[2]])
		push!(results, ["$(prefix)_twolevel_opt_wa", n, s, round(Int, log_size * item_size), level_count, wa2_sum])
		push!(results, ["$(prefix)_twolevel_opt_sizes", n, s, round(Int, log_size * item_size), level_count, sizes2])

		wa_best_sum = wa0_sum
		wa_best = wa0
		sizes_best = sizes0
		if wa_best_sum > wa1_sum
			wa_best_sum = wa1_sum
			wa_best = wa1
			sizes_best = sizes1
		end
		if wa_best_sum > wa2_sum
			wa_best_sum = wa2_sum
			wa_best = wa2
			sizes_best = sizes2
		end
		push!(results, ["$(prefix)_leveldb_best_wa_r", n, s, round(Int, log_size * item_size), level_count, sum(wa_best[1]), wa_best[1]])
		push!(results, ["$(prefix)_leveldb_best_wa_w", n, s, round(Int, log_size * item_size), level_count, sum(wa_best[2]), wa_best[2]])
		push!(results, ["$(prefix)_leveldb_best_wa", n, s, round(Int, log_size * item_size), level_count, wa_best_sum])
		push!(results, ["$(prefix)_leveldb_best_sizes", n, s, round(Int, log_size * item_size), level_count, sizes_best])

		# println("  multilevel")
		# intervals3 = IntervalModel.init_intervals(log_size, l0_count, level_count)
		# intervals3 = IntervalModel.optimize_wa_multilevel(X, log_size, l0_count, intervals3, wa_r_factor, ftol, max_time)
		# wa3 = IntervalModel.calculate_wa_multilevel(X, log_size, l0_count, intervals3)
		# wa3 = ([round(v, 2) for v in wa3[1]], [round(v, 2) for v in wa3[2]])
		# wa3_sum = get_wa(wa_r_factor, wa3)
		# sizes3 = IntervalModel.calculate_sizes_multilevel(X, log_size, l0_count, intervals3)
		# push!(results, ["$(prefix)_multilevel_wa", n, s, round(Int, log_size * item_size), level_count, wa3_sum, wa3])
		# push!(results, ["$(prefix)_multilevel_sizes", n, s, round(Int, log_size * item_size), level_count, sizes3])

		# println("  rocksdb")
		# # https://github.com/facebook/rocksdb/blob/master/util/options.cc
		# # ColumnFamilyOptions::OptimizeLevelStyleCompaction()
		# rocksdb_memtable_memory_budget = log_size * 2.		# so that we use the same memory for mbuf
		# rocksdb_write_buffer_size = rocksdb_memtable_memory_budget / 4.
		# rocksdb_min_write_buffer_number_to_merge = 2.
		# rocksdb_level0_file_num_compaction_trigger = 2.
		# rocksdb_max_bytes_for_level_base = rocksdb_memtable_memory_budget

		# new_log_size = rocksdb_write_buffer_size * rocksdb_min_write_buffer_number_to_merge
		# @assert new_log_size == log_size
		# new_l0_count = rocksdb_level0_file_num_compaction_trigger
		# new_l1_size = rocksdb_max_bytes_for_level_base
		# sizes3 = SizeModel.init_sizes(X, new_l1_size, 10.)
		# new_level_count = length(sizes3)
		# wa3 = SizeModel.calculate_wa(X, new_log_size, new_l0_count, sizes3)
		# wa3 = ([round(v, 2) for v in wa3[1]], [round(v, 2) for v in wa3[2]])
		# wa3_sum = get_wa(wa_r_factor, wa3)
		# push!(results, ["$(prefix)_rocksdb_wa", n, s, round(Int, log_size * item_size), new_level_count, wa3_sum, wa3])
		# push!(results, ["$(prefix)_rocksdb_sizes", n, s, round(Int, log_size * item_size), new_level_count, sizes3])

		# println("  mbuf")
		# sizes5 = copy(sizes0)
		# for i in 1:level_count
		#	if sizes5[i] < log_size
		#		sizes5[i] = log_size
		#	end
		# end
		# wa5 = SizeModel.calculate_mbuf_wa(X, log_size, sizes5)
		# wa5 = ([round(v, 2) for v in wa5[1]], [round(v, 2) for v in wa5[2]])
		# wa5_sum = get_wa(wa_r_factor, wa5)
		# push!(results, ["$(prefix)_mbuf_wa", n, s, round(Int, log_size * item_size), level_count, wa5_sum, wa5])
		# push!(results, ["$(prefix)_mbuf_sizes", n, s, round(Int, log_size * item_size), level_count, sizes5])

		# println("  mbuf_opt")
		# sizes6 = SizeModel.optimize_mbuf_wa(X, log_size, sizes5, wa_r_factor, ftol, max_time)
		# wa6 = SizeModel.calculate_mbuf_wa(X, log_size, sizes6)
		# wa6 = ([round(v, 2) for v in wa6[1]], [round(v, 2) for v in wa6[2]])
		# wa6_sum = get_wa(wa_r_factor, wa6)
		# push!(results, ["$(prefix)_mbuf_opt_wa", n, s, round(Int, log_size * item_size), level_count, wa6_sum, wa6])
		# push!(results, ["$(prefix)_mbuf_opt_sizes", n, s, round(Int, log_size * item_size), level_count, sizes6])

		# println("  leveldb_corrected")
		# sizes7 = copy(sizes0)
		# for i in 1:level_count
		#	if sizes7[i] < log_size
		#		sizes7[i] = log_size
		#	end
		# end
		# wa7 = SizeModel.calculate_wa(X, log_size, l0_count, sizes7)
		# wa7 = ([round(v, 2) for v in wa7[1]], [round(v, 2) for v in wa7[2]])
		# wa7_sum = get_wa(wa_r_factor, wa7)
		# push!(results, ["$(prefix)_leveldb_corrected_wa", n, s, round(Int, log_size * item_size), level_count, wa7_sum, wa7])
		# push!(results, ["$(prefix)_leveldb_corrected_sizes", n, s, round(Int, log_size * item_size), level_count, sizes7])

		println()
	end


	item_size = 1000

	for n in [1000000, 3300000, 10 * 1000000, 33 * 1000000, 100 * 1000000, 330 * 1000000, 1000 * 1000000]
	# for n in [1000000]
	# for n in [1000000, 3300000, 10 * 1000000]
	# for n in [100 * 1000000]
		for s in [0.00, 0.99]
		# for s in [0.00]
			eval_f(results, "sensitivity_item_count", n, s, item_size, 0, 4 * 1048576 / item_size)
			write_output("output_sensitivity.txt", results)
		end
	end

	# for n in [100 * 1000000, 200 * 1000000, 300 * 1000000, 400 * 1000000, 500 * 1000000, 600 * 1000000, 700 * 1000000, 800 * 1000000, 900 * 1000000, 1000 * 1000000]
	#	for s in [0.00, 0.99]
	#		eval_f(results, "sensitivity_item_count2", n, s, item_size, 0, 4 * 1048576 / item_size)
	#		write_output("output_sensitivity.txt", results)
	#	end
	# end

	for n in [100 * 1000000]
	  for s in [0.00, 0.20, 0.40, 0.60, 0.80, 0.99, 1.20, 1.40, 1.60]
		eval_f(results, "sensitivity_skewness", n, s, item_size, 0, 4 * 1048576 / item_size)
		write_output("output_sensitivity.txt", results)
	  end
	end

	for n in [100 * 1000000]
	  for s in [0.00, 0.99]
		for level_count in [3, 4, 5, 6, 7, 8, 9, 10]
			# this should be adjusted as used n (100 M = 5)
			if level_count == 5
				eval_f(results, "sensitivity_level_count", n, s, item_size, 0, 4 * 1048576 / item_size)
			else
				eval_f(results, "sensitivity_level_count", n, s, item_size, level_count, 4 * 1048576 / item_size)
			end
			write_output("output_sensitivity.txt", results)
		end
	  end
	end

	# use 10 M that we can plug into the implementation
	for n in [10 * 1000000]
		for s in [0.00, 0.99]
		# for s in [0.00]
			for log_size_b in [4 * 1048576, 10000000, 100000000, 1000000000]
				eval_f(results, "sensitivity_log_size", n, s, item_size, 0, log_size_b / item_size)
				write_output("output_sensitivity.txt", results)
			end
		end
	end
end

function density_analysis()
	item_size = 1000.
	l1_size = 10. * 1048576. / item_size

	n = 100 * 1000000

	results = []

	eval_f = (results, prefix, n, s, size) -> begin
		X = load_zipf_compressed(n, s, 0.1)
		update_derived_values(X)

		dinterval = interval_from_density(X, size)

		d = 0.
		while true
			ds = density(X, dinterval, d)
			push!(results, ["$(prefix)_unique", n, s, size, d, ds])
			if d == n - 1.
				break
			end
			d = min(d + n * 0.01, n - 1.)
		end
	end

	for s in [0.00, 0.99]
		size = l1_size * 1000.	# level 4

		X = load_zipf_compressed(n, s, 0.1)
		update_derived_values(X)

		println("n = ", n)
		println("s = ", s)
		println("size = ", size)
		println("UniqueInv = ", unique_inv(X, size))

		@time dinterval = interval_from_density(X, size)
		println("interval_from_density = ", dinterval)
		println("density_sum = ", density_sum(X, dinterval))

		eval_f(results, "density", n, s, size)
	end

	write_output_more_digits("output_density.txt", results)
end

function cola_test()
	#n = 1000000
	n = 100 * 1000000
	s = 0.00
	#s = 0.99

	item_size = 1000.
	log_size = 4. * 1048576. / item_size

	# growth factor
	#r = 2
	r = 3
	#r = 4
	#r = 10
	# level count
	L = round(Int, ceil(log(n / log_size) / log(float(r))))

	# wa_r_factor = 0.7
	wa_r_factor = 0.0

	X = load_zipf_compressed(n, s, 0.1)
	update_derived_values(X)

	wa = SizeModel.calculate_wa_cola(X, log_size, r, L)

	wa = ([round(v, 2) for v in wa[1]], [round(v, 2) for v in wa[2]])
	wa_sum = get_wa(wa_r_factor, wa)
	println(wa_sum, " ", wa)

	wa = SizeModel.calculate_wa_samt(X, log_size, r, L)

	wa = ([round(v, 2) for v in wa[1]], [round(v, 2) for v in wa[2]])
	wa_sum = get_wa(wa_r_factor, wa)
	println(wa_sum, " ", wa)
end

function silt_test()
	#n = 1000000
	n = 100 * 1000000
	s = 0.00

	hash_occupancy = 0.93
	tag_size = 15

	# from Section 5 in the SILT paper
	partition_count = 4
	hash_size = 2 ^ (tag_size + 2)
	hash_count = 62

	# wa_r_factor = 0.7
	wa_r_factor = 0.0

	mem_use_func = (n, partition_count, hash_size, hash_count) -> begin
		((6 * hash_size) + (2 * hash_size * hash_count / 2) + (0.4 * round(n / partition_count))) * partition_count
	end

	X = load_zipf_compressed(round(Int, round(n / partition_count)), s, 0.1)
	update_derived_values(X)

	wa = SizeModel.calculate_wa_silt(X, float64(hash_size), hash_occupancy, hash_count)
	wa = ([round(v, 2) for v in wa[1]], [round(v, 2) for v in wa[2]])
	wa_sum = get_wa(wa_r_factor, wa)
	println(wa_sum, " ", wa)
	mem_use = mem_use_func(n, partition_count, hash_size, hash_count)
	println(mem_use / 1000000., " ", "MB")

	s = 0.99

	X = load_zipf_compressed(round(Int, round(n / partition_count)), s, 0.1)
	update_derived_values(X)

	wa = SizeModel.calculate_wa_silt(X, float64(hash_size), hash_occupancy, hash_count)
	wa = ([round(v, 2) for v in wa[1]], [round(v, 2) for v in wa[2]])
	wa_sum = get_wa(wa_r_factor, wa)
	println(wa_sum, " ", wa)
	mem_use = mem_use_func(n, partition_count, hash_size, hash_count)
	println(mem_use / 1000000., " ", "MB")

	hash_count = 25

	wa = SizeModel.calculate_wa_silt(X, float64(hash_size), hash_occupancy, hash_count)
	wa = ([round(v, 2) for v in wa[1]], [round(v, 2) for v in wa[2]])
	wa_sum = get_wa(wa_r_factor, wa)
	println(wa_sum, " ", wa)
	mem_use = mem_use_func(n, partition_count, hash_size, hash_count)
	println(mem_use / 1000000., " ", "MB")

	hash_threshold = 0.10
	hash_count = 4

	mem_use_func_multi = (n, partition_count, hash_size, hash_count, hash_threshold) -> begin
		((6 * hash_size) + (2 * hash_size * (hash_count - 1) / 2) + (2 * round(n / partition_count) * hash_threshold) + (0.4 * round(n / partition_count))) * partition_count
	end
	wa = SizeModel.calculate_wa_silt_multi(X, float64(hash_size), hash_occupancy, hash_count, hash_threshold)

	wa = ([round(v, 2) for v in wa[1]], [round(v, 2) for v in wa[2]])
	wa_sum = get_wa(wa_r_factor, wa)
	println(wa_sum, " ", wa)
	mem_use = mem_use_func_multi(n, partition_count, hash_size, hash_count, hash_threshold)
	println(mem_use / 1000000., " ", "MB")

end

function uc_test()
	# https://github.com/facebook/rocksdb/wiki/Universal-Compaction

	# level0_file_num_compaction_trigger = 4		# default
	# level0_file_num_compaction_trigger = 6
	# level0_file_num_compaction_trigger = 8		# used in the example: https://github.com/facebook/rocksdb/wiki/Universal-Style-Compaction-Example
	# level0_file_num_compaction_trigger = 9
	level0_file_num_compaction_trigger = 12

	size_ratio = 1                      	# default
	# size_ratio = 10                   	# for skewed workloads
	# size_ratio = 100                  	# for skewed workloads
	#min_merge_width = 2                	# not implemented
	#max_merge_width = 1000000          	# not implemented
	max_size_amplification_percent = 200	# default
	#compression_size_percent = -1      	# not implemented
	stop_style = 1                      	# default; CompactionStopStyle::kCompactionStopStyleTotalSize
	#allow_trivial_move = false         	# not implemented

	# verbose = true
	verbose = false
	# simple = true
	simple = false

	merge_all_f = (X, arr) -> begin
		if simple
			min(sum(arr), 1000.)
		else
			interval_sum = 0.
			for i in 1:length(arr)
				interval_sum += unique_inv(X, arr[i])
			end
			Common.unique(X, interval_sum)
		end
	end

	uc_merge_f = (X, tables) -> begin
		# precondition
	    len = length(tables)
		if len >= level0_file_num_compaction_trigger
			@assert len >= 2

			# condition 1
			numer = sum(tables[1:end - 1])
			denom = tables[end]
			if numer / denom > max_size_amplification_percent / 100.
				source_size = sum(tables)
				merged_size = merge_all_f(X, tables)
				tables = [merged_size]
				if verbose
					println("cond 1: ", tables)
				end
				return tables, source_size, merged_size
			end

			# condition 2
			# PickCompactionUniversalReadAmp()
		    for start_i in 1:len
				candidate_count = 1
			    candidate_size = tables[start_i]
			    for i in start_i + 1:len
					sz = candidate_size * (100. + size_ratio) / 100.
					if sz < tables[i]
						break
					end
				    if stop_style == 0
						# kCompactionStopStyleSimilarSize
				        sz = (tables[i] * (100. + size_ratio)) / 100.
				        if sz < candidate_size
							break
						end
						candidate_size = tables[i]
					else
						# kCompactionStopStyleTotalSize
						candidate_size += tables[i]
					end
					candidate_count += 1
				end
				if candidate_count >= 2
					last_i = start_i + candidate_count - 1
					source_size = sum(tables[start_i:last_i])
					merged_size = merge_all_f(X, tables[start_i:last_i])
					tables = vcat(tables[1:start_i - 1], [merged_size], tables[last_i + 1:end])
					if verbose
						println("cond 2: ", tables)
					end
					return tables, source_size, merged_size
				end
			end

			# condition 3
			# PickCompactionUniversalReadAmp() with large ratio, a limited mergeable file count
			start_i = 1
			candidate_count = len - level0_file_num_compaction_trigger
			if candidate_count >= 2
			    last_i = start_i + candidate_count - 1
				source_size = sum(tables[start_i:last_i])
				merged_size = merge_all_f(X, tables[start_i:last_i])
				tables = vcat(tables[1:start_i - 1], [merged_size], tables[last_i + 1:end])
				if verbose
					println("cond 3: ", tables)
				end
				return tables, source_size, merged_size
			end
		end

		return tables, 0., 0.
	end

	eval_f = (results, prefix, n, s, item_size, log_size) -> begin
		if simple
			item_size = 1.
			log_size = 1.
		end

		X_load = load_zipf_compressed(n, 0.00, 0.1)
		update_derived_values(X_load)

		X_trans = load_zipf_compressed(n, s, 0.1)
		update_derived_values(X_trans)

		steps = 0
		inserts = 0.
		reads = 0.
		writes = 0.

		if simple
			tables = [1000.]
		else
			tables = [float64(X_load.count)]
		end

		do_inserts_f = (X, max_inserts) -> begin
			while inserts < max_inserts
				if length(tables) < level0_file_num_compaction_trigger + 2
					if simple
						new_table_size = 1.
					else
						new_table_size = Common.unique(X, log_size)
					end

					writes += log_size
					writes += new_table_size
					inserts += log_size
					tables = vcat([new_table_size], tables)

					if verbose
						println("insert: ", tables)
					end
				end

				# while true
				 	tables, new_reads, new_writes = uc_merge_f(X, tables)
				 	reads += new_reads
				 	writes += new_writes
				#	if new_writes == 0.
				#		break
				#	end
				# end

				steps += 1
				if steps % 20000 == 0
					wa_r = reads / inserts
					wa_w = writes / inserts
					println("WA_r: ", wa_r)
					println("WA_w: ", wa_w)
					# inserts = 0.
					# reads = 0.
					# writes = 0.
				end
			end
		end

		do_inserts_f(X_load, n)

		if verbose
			println("initial:", tables)
		end

		steps = 0
		inserts = 0.
		reads = 0.
		writes = 0.

		do_inserts_f(X_trans, n * 10)

		if verbose
			println("final:  ", tables)
		end

		wa_r = reads / inserts
		wa_w = writes / inserts
		println("WA_r: ", wa_r)
		println("WA_w: ", wa_w)

		wa_r = round(wa_r, 2)
		wa_w = round(wa_w, 2)
		push!(results, ["$(prefix)_wa_r", n, s, round(Int, log_size * item_size), wa_r])
		push!(results, ["$(prefix)_wa_w", n, s, round(Int, log_size * item_size), wa_w])
	end

	item_size = 1000.
	log_size = 4. * 1048576. / item_size
	# log_size = 256. * 1048576. / item_size

	results = []

	for n in [1000000, 3300000, 10000000, 33000000]
	# for n in [33000000]
		for s in [0.00, 0.99]
		# for s in [0.99]
			println("n = ", n)
			println("s = ", s)
			eval_f(results, "universal_compaction", n, s, item_size, log_size)
		end
	end

	write_output("output_universal_compaction.txt", results)
end

function run()
	#unique_data()
	sensitivity_2()
	# density_analysis()
	#cola_test()
	#uc_test()
end

end

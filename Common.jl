module Common
# using Ipopt

export Distribution

export check_validity
export update_derived_values

export unique
export unique_inv

export unique_avg

export density
export density_sum
export interval_from_density

export merge

export ccp

export load_zipf_compressed

export geom_mean

export get_wa


type Distribution
	count::Int64
	c::Array{Float64}
	p::Array{Float64}

	# derived values
	p1::Array{Float64}
	c_log_p1::Array{Float64}
end

function check_validity(X::Distribution)
	@assert X.count > 0
	@assert length(X.c) == length(X.p)

	count = 0.
	prob = 0.

	for i = 1:length(X.c)
		@assert X.c[i] != 0.
		@assert X.p[i] != 0.
		@assert X.p[i] != 1.

		count += X.c[i]
		prob += X.c[i] * X.p[i]
	end

	@assert abs(Float64(X.count) / count - 1.) < 0.001
	@assert abs(prob - 1.) < 0.001
end

function update_derived_values(X::Distribution)
	X.p1 = Array(Float64, length(X.c))
	X.c_log_p1 = Array(Float64, length(X.c))
	for i = 1:length(X.c)
		p1 = 1. - X.p[i]
		X.p1[i] = p1
		X.c_log_p1[i] = X.c[i] * log(p1)
	end
end

type UniqueParam
	X::Distribution
	c::Float64
end

function hash(x::UniqueParam)
	hash(x.c)
end

function isequal(x::UniqueParam, y::UniqueParam)
	x.c == y.c && x.X == y.X
end

# global unique_memoization = Dict{UniqueParam, Float64}()

function unique(X::Distribution, c::Float64)
	# global unique_memoization::Dict{UniqueParam, Float64}

	if c == Inf
		return Float64(X.count)
	end

	@assert c >= 0.

	# get!(unique_memoization, UniqueParam(X, c)) do
		s = Float64(X.count)
		for i = 1:length(X.c)
			# s -= X.c[i] * ((1. - X.p[i]) ^ c)
			s -= X.c[i] * (X.p1[i] ^ c)
		end

		s
	# end
end

function unique_diff(X::Distribution, c::Float64)
    s = 0.
	for i = 1:length(X.c)
        # p1 = 1. - X.p[i]
        # s -= X.c[i] * (p1 ^ c) * log(p1)
        s -= X.c_log_p1[i] * (X.p1[i] ^ c)
    end

    s
end

function unique_int(X::Distribution, c0::Float64, c1::Float64)
	@assert c0 < c1
	f = (c) -> begin
		unique(X, c)
	end
	I, E = quadgk(f, c0, c1, maxevals=10)
	I
end

function discrete_sum(f, a::Float64, b::Float64, maxevals::Int64=10)
	pq = Collections.PriorityQueue()
	vs = []

	f_a = f(a)
	push!(vs, (a, f_a))
	f_b = f(b)
	push!(vs, (b, f_b))
	Collections.enqueue!(pq, (a, b, f_a, f_b), -abs((f_a - f_b) * (a - b)))
	eval = 2

	while eval < maxevals
		try
			a, b, f_a, f_b = Collections.dequeue!(pq)
		catch y
			if isa(y, BoundsError)
				break
			end
		end
		# println("eval=", eval, " a=", a, " b=", b, " f_a=", f_a, " f_b=", f_b, " diff=", abs(f_a - f_b))
		if a + 1. < b
			m = round((a + b) / 2.)
			@assert a != m
			@assert b != m
			f_m = f(m)
			push!(vs, (m, f_m))
			Collections.enqueue!(pq, (a, m, f_a, f_m), -abs((f_a - f_m) * (a - m)))
			Collections.enqueue!(pq, (m, b, f_m, f_b), -abs((f_m - f_b) * (m - b)))
			eval += 1
		end
	end

	sort!(vs)

	sum = 0.
	len = length(vs)
	for i = 2:len
		sum += vs[i - 1][2] * (vs[i][1] - vs[i - 1][1])
	end
	sum += vs[end][2] * 1.

	sum
end

function unique_avg(X::Distribution, c0::Float64, c1::Float64)
	# remove a negative range that are not valid for unique(); quadgk() can emit DomainError otherwise
	if c0 < 0.
		c0 = 0.
	end
	if c1 < 0.
		c1 = 0.
	end

	if c0 > c1
		0.
	elseif c0 == c1
		unique(X, c0)
	else
		# if c1 / c0 < 100.
		#	sum = 0.
		#	count = 0
		#	for i = 1:int64(ceil(c1 / c0))
		#		sum += unique(X, c0 * Float64(i))
		#		count += 1
		#	end
		#	sum / count
		#	# discrete_sum(f, start_step, end_step) / (end_step - start_step + 1.)
		# else
			unique_int(X, c0, c1) / (c1 - c0)
		# end

		# start_step = 1.
		# end_step = c1 / c0
		# f = (step) -> begin
		#	unique(X, c0 * step)
		# end
		# discrete_sum(f, start_step, end_step) / (end_step - start_step + 1.)
	end
end


type UniqueInvParam
	X::Distribution
	u::Float64
end

function hash(x::UniqueInvParam)
	hash(x.u)
end

function isequal(x::UniqueInvParam, y::UniqueInvParam)
	x.u == y.u && x.X == y.X
end

# global unique_inv_memoization = Dict{UniqueInvParam, Float64}()

function unique_inv(X::Distribution, u::Float64)
	unique_inv_nt(X, u)
	# unique_inv_ipopt(X, u)
end

function unique_inv_nt(X::Distribution, u::Float64)
    # Newton's method
	# global unique_inv_memoization::Dict{UniqueInvParam, Float64}

	if u >= Float64(X.count) * (1 - 0.000001)
		return Inf
	end

	# get!(unique_inv_memoization, UniqueInvParam(X, u)) do
	    # take u as the initial c
	    c = u
	    for count = 1:100
	        u1 = unique(X, c)
	        if abs(u1 / u - 1.) < 0.001
	            break
	        end
	        c -= (u1 - u) / unique_diff(X, c)

			if c < 0.
				c = 0.
			end
	    end

	    c
	# end
end

function unique_inv_ipopt(X::Distribution, u::Float64)
	# inaccurate

	# if u >= Float64(X.count) * (1 - 0.000001)
	#	return Inf
	# end

	eval_f = (x) -> begin
		abs(unique(X, x[1]) - u) / u
	end

	eval_grad_f = (x, grad_f) -> begin
		# grad_f[1] = unique_diff(X, x[1])
		diff = x[1] * 0.001
		grad_f[1] = (abs(unique(X, x[1] + diff) - u) - abs(unique(X, x[1]) - u)) / u / diff
	end

	eval_g = (x, g) -> begin
		# g[1] = x[1]
	end

	eval_jac_g = (x, mode, rows, cols, values) -> begin
		# if mode == :Structure
		#	rows[1] = 1
		#	cols[1] = 1
		# else
		#	values[1] = 1.
		# end
	end

	v_L = [1.]
	v_U = [Float64(X.c) ^ 2.]

	# g_L = [1.]
	# # # g_U = [2.e19]
	# g_U = [Float64(X.c) ^ 2.]
	g_L = Array(Float64, 0)
	g_U = Array(Float64, 0)

	prob = createProblem(1, v_L, v_U,
						 0, g_L, g_U,
						 0, 0,
						 eval_f, eval_g, eval_grad_f, eval_jac_g)

	addOption(prob, "hessian_approximation", "limited-memory")

	# addOption(prob, "tol", 0.1)

	addOption(prob, "print_level", 2);

	prob.x = [u]
	status = solveProblem(prob)
	# ret = Ipopt.ApplicationReturnStatus[status]
	# obj_val = prob.obj_val
	# println("$obj_val in unique_inv (returned $ret)")

	prob.x[1]
end


function density(X::Distribution, interval::Float64, d::Float64)
	n = X.count
	v = unique(X, d / n * interval) / n
	#println(v)
	v
end

function density_sum(X::Distribution, interval::Float64)
	@assert interval >= 0.

	n = X.count

	# using integration
	f = (d) -> begin
		v = density(X, interval, d)
		@assert !isnan(v)
		v
	end
	#I, E = quadgk(f, 1., n, maxevals=10)
	I, E = quadgk(f, 0., n - 1., maxevals=10)
	I

	# using a geometric sum of unique() - this is fast but has a precision issue with large n due to the use of close-to-zero divisions
	# s = Float64(n)
	# for i = 1:length(X.c)
	#	s -= X.c[i] * (1. - (X.p1[i] ^ interval)) / (1. - (X.p1[i] ^ (interval / n))) / n
	# end
	# s
end

function interval_from_density(X::Distribution, u::Float64)
	# fix up an invalid u that can be created by the solver
	u = min(u, float(X.count))

	# unique_inv() * 2 is usually close to the solution
	c = unique_inv(X, u) * 2.
	for count = 1:100
	    u1 = density_sum(X, c)
	    if abs(u1 / u - 1.) < 0.001
	        break
	    end
	    diff = (u1 - density_sum(X, c * 1.01)) / (c - c * 1.01)
		if isnan(diff)
			println(diff, " ", u1, " ", density_sum(X, c * 1.1), " ", c)
			@assert false
		end
	    c -= (u1 - u) / diff

		if c < 0.
			c = 0.
		end
	end

	c
end


function merge(X::Distribution, n1::Float64, n2::Float64)
    c = unique_inv(X, n1) + unique_inv(X, n2)
    unique(X, c)
end


function ccp_subset_sum_choose(X::Distribution, q::Int64, pos::Int64, min::Int64, p_sum::Float64)
	if pos > q
		return 1. / (1. - p_sum)
	end
	m = X.count
	s = 0.
	for i = min:m
		s += ccp_subset_sum_choose(X, q, pos + 1, i + 1, p_sum + X.p[i])
	end

	s
end

function ccp_subset_sum(X::Distribution, q::Int64)
	ccp_subset_sum_choose(X, q, 1, 1, 0.)
end

function ccp(X::Distribution, j::Int64)
	# Coupon collector's problem; expected time to collect j coupons whose distribution is X
	# this is quite slow for large X (e.g., > 30)

	m = X.count
	for i = 1:m
		# ccp_subset_sum() cannot handle non-1 cardinality
		@assert X.c[i] == 1.
	end

	t = 0.
	for q = 0:(j - 1)
		t += Float64((-1) ^ (j - 1 - q) * binomial(m - q - 1, m - j)) * ccp_subset_sum(X, q)
	end

	t
end

function zipf(count::Int64, s::Float64)
	X = Distribution(count, Array(Float64, count), Array(Float64, count), Array(Float64, 0), Array(Float64, 0))
	p_sum = 0.
	for i = 1:count
		if s == 0.
			p = 1.
		elseif s == 1.
			p = 1. / (Float64(i))
		else
			p = 1. / (Float64(i) ^ s)
		end
		p_sum += p
		X.c[i] = 1.
		X.p[i] = p
	end
	X.p /= p_sum
	check_validity(X)

	X
end

function zipf_compressed(count::Int64, s::Float64, rel_diff::Float64)
	X = Distribution(count, Array(Float64, 0), Array(Float64, 0), Array(Float64, 0), Array(Float64, 0))

	p_denom = 0.

	if s == 0.
		p = 1.
	elseif s == 1.
		p = 1. / (Float64(count + 1 - 1))
	else
		p = 1. / (Float64(count + 1 - 1) ^ s)
	end
	c = 1.
	p_denom += p
	min_p = p
	c_sum = c
	p_sum = c * p
	for i = 2:count
		if s == 0.
			p = 1.
		elseif s == 1.
			p = 1. / (Float64(count + 1 - i))
		else
			p = 1. / (Float64(count + 1 - i) ^ s)
		end
		c = 1.
		p_denom += p
		@assert min_p <= p
		if p / min_p - 1. <= rel_diff
			c_sum += c
			p_sum += c * p
		else
			push!(X.c, c_sum)
			push!(X.p, p_sum / c_sum)
			min_p = p
			c_sum = c
			p_sum = c * p
		end
	end
	push!(X.c, c_sum)
	push!(X.p, p_sum / c_sum)
	X.p /= p_denom
	check_validity(X)

	X
end

function load_zipf_compressed(count::Int64, s::Float64, rel_diff::Float64)
	filename = string("data/zipf_", count, "_", s, "_", rel_diff, ".dat")

	X = Distribution(0, Array(Float64, 0), Array(Float64, 0), Array(Float64, 0), Array(Float64, 0))
	try
		f = open(filename, "r")
		X.count = deserialize(f)
		X.c = deserialize(f)
		X.p = deserialize(f)
		close(f)
	catch
		println("creating $filename")
		X = zipf_compressed(count, s, rel_diff)
		f = open(filename, "w")
		serialize(f, X.count)
		serialize(f, X.c)
		serialize(f, X.p)
		close(f)
	end
	X
end


function compress(X::Distribution, rel_diff::Float64)
	new_X = Distribution(X.count, Array(Float64, 0), Array(Float64, 0), Array(Float64, 0), Array(Float64, 0))

	perm = sortperm(X.p)

	p = X.p[perm[1]]
	c = X.c[perm[1]]
	min_p = p
	c_sum = c
	p_sum = c * p
	for idx in perm[2:end]
		p = X.p[idx]
		c = X.c[idx]
		@assert min_p <= p
		if p / min_p - 1. <= rel_diff
			c_sum += c
			p_sum += c * p
		else
			push!(new_X.c, c_sum)
			push!(new_X.p, p_sum / c_sum)
			min_p = p
			c_sum = c
			p_sum = c * p
		end
	end
	push!(new_X.c, c_sum)
	push!(new_X.p, p_sum / c_sum)
	check_validity(new_X)

	new_X
end

function geom_mean(A::Array{Float64})
	s = 0.
	for a in A
		s += 1. / a
	end
	1. / s
end

function get_wa(wa_r_factor::Float64, t)
	wa_r = t[1]
	wa_w = t[2]
	return sum(wa_w) + wa_r_factor * sum(wa_r)
end

end

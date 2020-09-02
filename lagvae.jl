
using Flux, Flux.Data.MNIST
using Flux: params, onehotbatch, onecold, crossentropy, throttle, ConvTranspose
import Flux.Tracker: Params, gradient,update!
using Base.Iterators: repeated, partition
using PyPlot
using BSON
using Statistics
using StatsBase
#using CuArrays
#using CUDAdrv
#using CUDAnative

X = MNIST.images()

const BATCH_SIZE = 64
const KL_CONST = 0.001f0
const E1 = 50
const E2 = 50
const alpha = 1
l1 = Flux.Tracker.param(1f0)
l2 = Flux.Tracker.param(1f0)

function make_minibatches(X, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:,:,:,i] = Float32.(X[idxs[i]])
    end
    return X_batch
end
function make_minibatches(X, batch_size::Int)
    idxs = partition(1:length(X), batch_size)
    return [make_minibatches(X, i) for i in idxs]
end

function save_model(sname::AbstractString)
	global InitialEncoder, mu, logvar, decoder
	print("In save model: \n")
	print("Initial encoder: $(typeof(InitialEncoder)) \n")
	enc_params = Tracker.data.(cpu.(params(InitialEncoder)))
	print("Enc params: $(typeof(enc_params)), $(size(enc_params)) \n")
	for param in enc_params
		print("Enc param: $(typeof(param)), $(size(param)) \n")
	end
	enc_sname = sname *"_InitialEncoder.bson"
	BSON.@save enc_sname enc_params
	mu_params =  Tracker.data.(cpu.(params(mu)))
	mu_sname = sname * "_mu.bson"
	BSON.@save mu_sname mu_params
	logvar_params = Tracker.data.(cpu.(params(logvar)))
	logvar_sname = sname * "_logvar.bson"
	BSON.@save logvar_sname logvar_params
	decoder_params = Tracker.data.(cpu.(params(decoder)))
	decoder_sname = sname * "_decoder.bson"
	BSON.@save  decoder_sname decoder_params
end

function load_model(sname::AbstractString)
	global InitialEncoder, mu, logvar, decoder
	print("In load model: \n")
	print("Initial encoder: $(typeof(InitialEncoder)) \n")
	enc_sname = sname *"_InitialEncoder.bson"
	BSON.@load enc_sname enc_params
	InitialEncoder = cpu(InitialEncoder)
	Flux.loadparams!(InitialEncoder, enc_params)
	InitialEncoder = gpu(InitialEncoder)
	mu_sname = sname * "_mu.bson"
	BSON.@load mu_sname mu_params
	mu = cpu(mu)
	Flux.loadparams!(mu, mu_params)
	mu = gpu(mu)
	logvar_sname = sname * "_logvar.bson"
	BSON.@load logvar_sname logvar_params
	logvar = cpu(logvar)
	Flux.loadparams!(logvar, logvar_params)
	logvar = gpu(logvar)
	decoder_sname = sname * "_decoder.bson"
	BSON.@load decoder_sname decoder_params
	decoder = cpu(decoder)
	Flux.loadparams!(decoder, decoder_params)
	decoder = gpu(decoder)
end
data = gpu.(make_minibatches(X, BATCH_SIZE))
X = 0
expand_dim(x::AbstractArray) = reshape(x, (size(x)...,1))
const H_SIZE = 100
const Z_SIZE = 32
const COLLAPSED_SIZE = 4*4*32
const LEARNING_RATE = 0.1
const SAMP_DIM = 200

InitialEncoder = gpu(Chain(Conv((3,3), 1=>16, stride=(2,2),pad=(1,1), relu),
					Conv((3,3), 16=>32, stride=(2,2),pad=(1,1),relu),
					Conv((3,3),32=>32,stride=(2,2),pad=(1,1), relu),
					x -> reshape(x, :, size(x,4)),
					Dense(COLLAPSED_SIZE,H_SIZE,relu),
					))

mu = gpu(Dense(H_SIZE, Z_SIZE))
logvar = gpu(Dense(H_SIZE, Z_SIZE))

z(mu, logvar) = gpu.(mu .+ exp.(logvar)./2f0 .* randn(Float32))
z(x) = gpu.(z(encoder(x)...))

function encoder(x)
	h = InitialEncoder(x)
	m = mu(h)
	logv = logvar(h)
	return m, logv
end
decoder = gpu(Chain(Dense(Z_SIZE, H_SIZE, relu),
	 			Dense(H_SIZE, COLLAPSED_SIZE, relu),
     			x -> reshape(x, (4,4,32,size(x,2))),
     			ConvTranspose((3,3), 32=>32, stride=(2,2),pad=(1,1), relu),
     			ConvTranspose((4,4), 32 => 16, stride=(2,2),pad=(1,1), relu),
     			ConvTranspose((4,4), 16=>1, stride=(2,2),pad=(1,1)),
     			))
function model(x)
    mu, logvar = encoder(x)
    samp = z(mu, logvar)
    y = decoder(samp)
    return (mu, logvar, samp, y)
end
model = gpu(model)
#RLoss(x,y) = Flux.mse(x,y)
RLoss(x,y) = sum((x .- y).^2)
KLLoss(mu, logvar) = 0.5f0 * sum(exp.(2f0 .* logvar) + mu.^2 .- 1f0 .+logvar.^2)

function compute_kernel(x,y)
	x_samp_dimension = size(x,2)
	y_samp_dimension = size(y,2)
	z_size_dim = size(x,1)
	xrs = repeat(reshape(x,(z_size_dim,1,x_samp_dimension)),outer=(1,y_samp_dimension,1))
	yrs = repeat(reshape(y, (z_size_dim, y_samp_dimension,1)), outer=(1,1,x_samp_dimension))
	kernel = exp.(-mean(xrs .- yrs, dims=1).^2) ./ Float32(z_size_dim)
	return kernel
end

function compute_mmd(x,y)
	xx = compute_kernel(x,x)
	yy = compute_kernel(y,y)
	xy = compute_kernel(x,y)
	#println("$(typeof(xx)), $(typeof(yy)), $(typeof(xy))")
	mmd_loss = mean(xx) + mean(yy) - 2f0*mean(xy)
	return mmd_loss
end
function mmd_loss(x)
	samp = z(x)
	yhat = decoder(samp)
	true_samps = randn(Float32,Z_SIZE, SAMP_DIM)
	rloss = RLoss(x,yhat)
	mmdloss = compute_mmd(samp, true_samps)
	print("rloss: $rloss, mmdloss: $mmdloss")
	return rloss + mmdloss
end
function loss(x,l1,l2)
	m,l,samp,yhat = model(x)
	rloss = RLoss(x,yhat)
	kloss = KLLoss(m,l)
	true_samps = randn(Float32, Z_SIZE, SAMP_DIM)
	mmdloss = compute_mmd(samp, true_samps)
	println("$(typeof(mmdloss)), $(typeof(rloss)), $(typeof(kloss))")

	#clip lambdas
	l1 = min(max(l1,0f0),100f0)
	l2 = min(max(l2,0f0),100f0)
	if alpha < 0
		total_loss = (l1 * rloss) + ((l1 * alpha) * kloss) + (l2 * mmdloss) - (l1 * E1) - (l2 * E2)
	end
	if alpha > 0
		println(l1, l2)
		total_loss = ((l1 * alpha) * rloss) + (l1 * kloss) + (l2 * mmdloss) - (l1 * E1) - (l2 * E2)
	end
	if alpha == 0
		total_loss = (l1 * rloss) + (l1 * kloss) + (l2 * mmdloss) - (l1 * E1) - (l2 * E2)
	end
	println("loss: $total_loss, rloss: $rloss, kloss: $kloss, mmdloss: $mmdloss")
	return total_loss
end
##


##
opt = ADAM()


losses = []
ps = params(InitialEncoder, mu,logvar, decoder)
train = data[1:850]
testdata = data[850:end]
println("Train Data: $(typeof(train)), $(size(train)) \n")
println("Test Data: $(typeof(testdata)), $(size(testdata)) \n")

print(l1)



d = data[1]
para = Params(ps)
l2 *
global l1, l2
function print_grads(gs)
	for g in gs
		println("size: $(size(last(g))), mean: $(mean(last(g))), std: $(std(last(g)))")
	end
end
macro summary(arg)
	println(string(arg))
	print_summary(eval(arg))
end
@summary gs
macro printnm(arg)
	s = string(arg)
	return println("$s: $(esc(arg))")
end
printnm(x) = println("$(@name x): $()")

@printnm bib

macro parse_run(arg)
    println("at Parse time, The argument is: ", arg)
	s = string(arg)
    return :(println("at Runtime, The argument is: $s ", $(esc(arg))))
end
macro printnm_2(arg)
	println("at Parse time, The argument is: ", arg)
	s = string(arg)
    return :(println("$(s): ", $(esc(arg))))
end

@printnm bib
@parse_run bib
bib = 5
@printnm bib
l1grad = 0
l2grad = 0
glib = 5
@printnm_2 glib
@macroexpand @printnm_2 glib
function my_train(data, opt, params,l1,l2,N=1)
	losses = []
	for i in 1:N
		batch_losses = []
		for d in data
			l = loss(d,l1,l2)
			push!(batch_losses, l)
			gs = gradient(()->l, ps)
			update!(opt, params, gs)
			#print_grads(gs)
			l1grad = Flux.Tracker.grad(l1)
			l2grad = Flux.Tracker.grad(l2)
			@printnm l1grad
			@printnm l2grad
			l1 = l1.data
			l1 += 0.00001 * l1grad
			l2 = l2.data
			l2 += 0.00001 * l2grad
			l1 = Flux.Tracker.param(l1)
			l2 = Flux.Tracker.param(l2)
		end
		push!(losses, mean(batch_losses))
	end
	return losses

end
my_train(data, opt, ps, l1,l2)

function t(a)
	@printnm a
end
t(ws)
l = loss(d,l1,l2)
gs = gradient(() -> l,ps)
print_grads(gs)

function summary(ws,print=true)
	means = []
	stds = []
	for w in ws
		m = mean(w)
		s = std(w)
		if print
			println("$(size(w)), $m, $s")
		end
		push!(means,m)
		push!(stds, s)
	end
	return means, stds
end
function histograms(ws,show=true)
	plots = []
	for w in ws
		plt = Plots.histogram(flatten(w))
		display(plt)
		push!(plots, plt)
	end
	return plots
end
function print_summary(xs)
	for x in xs
		println("$(size(x)), $(mean(x)), $(std(x))")
	end
end

Plots.histogram(randn(10000))
for i in 1:3
	global l1, l2
	#global InitialEncoder, mu, logvar, decoder
	#print("At beginning train loop: Initial encoder: $(typeof(InitialEncoder)) \n")
	#print("At beginning train loop: modelr: $(typeof(model)) \n")
	Flux.train!(loss, ps, zip(train), opt)
	# run test losses and so forth
	#(mu, logvar, samp, y) = model(testdata)
	#testloss = mean(loss.(testdata))
	l1grad = Flux.Tracker.grad(l1)
	l2grad = Flux.Tracker.grad(l2)
	l1 = l1.data
	l1 -= 0.001 * l1grad
	l2 = l2.data
	l2 -= 0.001 * l2grad
	l1 = Flux.Tracker.param(l1)
	l2 = Flux.Tracker.param(l2)
	println("test loss: $testloss")
	if i % 1 == 0
		(_,_,_,yhat) = model(testdata[1])
		PyPlot.imshow(reshape(yhat, (28,28)))
		PyPlot.show()
	end
end

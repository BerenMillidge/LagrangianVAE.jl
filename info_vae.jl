
using Flux
using Flux.Tracker
using Flux: params, onehotbatch, onecold, crossentropy, throttle, ConvTranspose, Conv, chunk, batchseq
using BSON
using NPZ
using Base.Iterators: repeated, partition
#using CuArrays
using Plots, Images, ImageView
using Statistics
using Random
using StatsBase

const Z_SIZE = 32
const BATCH_SIZE = 64
const LEARNING_RATE = 0.001
const KL_TOLERANCE = 0.5
const DENSE_PARAMS = 2*2*256
const L2_REG_ALPHA = 0.0001
const KL_FACTOR = 0.001f0
const ACTION_SIZE = 4
const SAMP_DIM = 200

expand_dim(x::AbstractArray) = reshape(x, (size(x)...,1))
reduce_dim(x::AbstractArray) = reshape(x, size(x,1))
make_minibatches(X, batch_size::Int) = [X[:,:,:,first(idx):last(idx)] for idx in partition(1:size(X,4), batch_size)]

function save_model(sname::AbstractString)
	print("In save model: \n")
	#print("Initial encoder: $(typeof(InitialEncoder)) \n")
	enc_params = Tracker.data.(cpu.(params(HaEncoder)))
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
	decoder_params = Tracker.data.(cpu.(params(HaDecoder)))
	decoder_sname = sname * "_decoder.bson"
	BSON.@save  decoder_sname decoder_params
end
function load_model(sname::AbstractString)
	global HaEncoder, mu, logvar, HaDecoder, Tmodel, Qmodel, opt 
	print("In load model: \n")
	#print("Initial encoder: $(typeof(InitialEncoder)) \n")
	enc_sname = sname *"_InitialEncoder.bson"
	BSON.@load enc_sname enc_params
	HaEncoder = cpu(HaEncoder)
	Flux.loadparams!(HaEncoder, enc_params)
	HaEncoder = gpu(HaEncoder)
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
	HaDecoder = cpu(HaDecoder)
	Flux.loadparams!(HaDecoder, decoder_params)
	HaDecoder = gpu(HaDecoder)
end

##
HaEncoder = Chain(Conv((4,4), 3=>32, stride=(2,2), leakyrelu),
					Conv((4,4), 32=>64, stride=(2,2), leakyrelu),
					Conv((4,4), 64=>128, stride=(2,2),leakyrelu),
					Conv((4,4), 128=>256, stride=(2,2), leakyrelu),
					x->reshape(x, (DENSE_PARAMS,size(x,4))),
					)
mu = Dense(DENSE_PARAMS, Z_SIZE)
logvar = Dense(DENSE_PARAMS, Z_SIZE)

function encoder(x)
	#println("x: $(typeof(x)), $(size(x))")
	h = HaEncoder(x)
	m = mu(h)
	logv = logvar(h)
	return (m, logv)
end
function z(mu, logvar)
	return mu .+ exp.(logvar ./2f0)  .* Float32.(randn((Z_SIZE, size(mu,2))))
end
z(x) = z(encoder(x)...)

HaDecoder = Chain(Dense(Z_SIZE, DENSE_PARAMS),
				x-> reshape(x, (1,1,DENSE_PARAMS, size(x,2))),
				ConvTranspose((5,5), DENSE_PARAMS=>128, stride=(2,2), leakyrelu),
				ConvTranspose((5,5), 128=>64, stride=(2,2), leakyrelu),
				ConvTranspose((6,6), 64=>32, stride=(2,2), leakyrelu),
				ConvTranspose((6,6), 32=>3, stride=(2,2),sigmoid))

function Vmodel(x)
	mu, logvar = encoder(x)
	samp = z(mu, logvar)
	y = HaDecoder(samp)
	#println(y)
	return (mu, logvar, samp, y)
end
RLoss(x,y) = mean(sum((x .-y).^2,dims=[1,2,3]))
#RLoss(x,y) = sum((x .-y).^2)
function KLLoss(mu, logvar)
	return -0.5f0 * sum(1f0 .+ logvar .- mu.^2 .- exp.(logvar), dims=1)
end
function v_loss(x)
	#println("in v_loss")
	(mu, logvar, samp, yhat) = Vmodel(x)
	rloss = RLoss(x,yhat)
	kloss = KLLoss(mu, logvar)
	kloss = max.(kloss, KL_TOLERANCE * Z_SIZE)
	kloss = mean(kloss)
	println("rloss $rloss, kloss $kloss")
	return rloss + kloss
end


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
	mmd_loss = mean(xx) + mean(yy) - 2*mean(xy)
	return mmd_loss
end
function mmd_loss(x)
	samp = z(x)
	yhat = HaDecoder(samp)
	true_samps = randn(Z_SIZE, SAMP_DIM)
	rloss = RLoss(x,yhat)
	mmdloss = compute_mmd(samp, true_samps)
	print("rloss: $rloss, mmdloss: $mmdloss")
	return rloss + mmdloss
end

function print_grads(tx)
	println("ENCODER:")
	gs = Tracker.gradient(()->mmd_loss(tx), params(HaEncoder))
	for g in gs
		println("size: $(size(last(g))), mean: $(mean(last(g))), std: $(std(last(g)))")
	end
	println("MU:")
	gs = Tracker.gradient(()->mmd_loss(tx), params(mu))
	for g in gs
		println("size: $(size(last(g))), mean: $(mean(last(g))), std: $(std(last(g)))")
	end
	println("LOGVAR:")
	gs = Tracker.gradient(()->mmd_loss(tx), params(logvar))
	for g in gs
		println("size: $(size(last(g))), mean: $(mean(last(g))), std: $(std(last(g)))")
	end
	println("DECODER:")
	gs = Tracker.gradient(()->mmd_loss(tx), params(HaDecoder))
	for g in gs
		println("size: $(size(last(g))), mean: $(mean(last(g))), std: $(std(last(g)))")
	end
end
##
const FNAME = "../FramesData.npy"
data = npzread(FNAME)
data = Float32.(data) ./ 255f0
data = permutedims(data, [2,3,4,1])
data = make_minibatches(data, BATCH_SIZE)
opt = ADAM(0.0001)
ps = params(HaEncoder, HaDecoder, mu, logvar)
tx = data[55]
evalcb = () -> print_grads(tx)

function save_samples(tx,save_name, i)
	(_,_,_,ohat) = Vmodel(tx)
	for j in 1:5
		o = ohat[:,:,:,j].data
		save("$save_name/epoch_$(i)_ohat_$(j).png",colorview(RGB,permutedims(o, [3,1,2])))
	end
end
##
const save_name = "infoVAE"
for i in 1:500
	println("Epoch: $i")
	save_samples(tx, save_name, i)
	if i % 50 == 0
		print_grads(tx)
	end
	Flux.train!(mmd_loss, ps, zip(data), opt)

	if i % 3 == 0
		save_model("$save_name/model")
	end

end

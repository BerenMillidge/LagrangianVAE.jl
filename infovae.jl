using Flux, Flux.Data.MNIST
using Flux: params, onehotbatch, onecold, crossentropy, throttle, ConvTranspose
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
RLoss(x,y) = Flux.mse(x,y)
KLLoss(mu, logvar) = KL_CONST *  0.5f0 * sum(exp.(2f0 .* logvar) + mu.^2 .- 1f0 .+logvar.^2)
L2Loss(alpha) = alpha * sum(x->sum(x.^2), params(model))

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
function loss(x)
	(mu, logvar, samp, y) = model(x)
	return gpu(RLoss(x,y) + KLLoss(mu, logvar))
end
function mmd_loss(x)
	samp = z(x)
	yhat = decoder(samp)
	true_samps = randn(Z_SIZE, SAMP_DIM)
	rloss = RLoss(x,yhat)
	mmdloss = compute_mmd(samp, true_samps)
	#print("rloss: $rloss, mmdloss: $mmdloss")
	return rloss + mmdloss
end


loss(x,y,mu,logvar) = RLoss(x,y) + KLLoss(mu, logvar)
function Nsample(x,N)
	mu, logvar = encoder(x)
	ys = []
	for i in 1:N
		samp = z(mu, logvar)
		push!(ys, decoder(samp))
	end
	return ys
end
##
opt = ADAM()


@info("Beginning Training")
losses = []
print("Before training loop, initial encoder: $(typeof(InitialEncoder)) \n")
print("Before training loop, model: $(typeof(model)) \n")
ps = params(InitialEncoder, mu,logvar, decoder)
train = data[1:850]
testdata = data[850:end]
println("Train Data: $(typeof(train)), $(size(train)) \n")
println("Test Data: $(typeof(testdata)), $(size(testdata)) \n")


#data = data[1:2]

for i in 1:3
	#global InitialEncoder, mu, logvar, decoder
	#print("At beginning train loop: Initial encoder: $(typeof(InitialEncoder)) \n")
	#print("At beginning train loop: modelr: $(typeof(model)) \n")
	Flux.train!(mmd_loss, ps, zip(train), opt)
	# run test losses and so forth
	#(mu, logvar, samp, y) = model(testdata)
	testloss = mean(mmd_loss.(testdata))
	println("test loss: $testloss")
	if i % 1 == 0
		(_,_,_,yhat) = model(testdata[1])
		PyPlot.imshow(reshape(yhat, (28,28)))
		PyPlot.show()
	end
end
	#print("Before save_model initial encoder: $(typeof(InitialEncoder)) \n")
	#save_model("CVAE_Model")
	#print("After save model: \n")
	#print("Initial encoder: $(typeof(InitialEncoder)) \n")
	##print("Saved, now loading")
	#print("In load model: \n")
	#print("Initial encoder: $(typeof(InitialEncoder)) \n")
	##sname = "CVAE_Model"
	#load_model(sname)

end
x = [1,2,3,4]
y = [5,6,7,8]

x = randn(5,10)
y = randn(5,12)
x = reshape(x, (5,1,10))
repeat(x, outer=(1,12,1))
repeat(x, inner=[1,1,1])

x = [1 2; 2 3; 4 5]
y = [2 1 3 4; 4 3 1 2; 1 2 3 4]
xr = reshape(x, (3,1,2))
xrs = repeat(xr, outer=(1,4,1))
yr = reshape(y, (3,4,1))
yrs = repeat(yr, outer=(1,1,2))
kernel = exp.(-mean(xrs .- yrs,dims=1).^2) ./3
function compute_kernel(x,y)
	x_samp_dimension = size(x,2)
	y_samp_dimension = size(y,2)
	z_size_dim = size(x,1)
	xrs = repeat(reshape(x,(z_size_dim,1,x_samp_dimension)),outer=(1,y_samp_dimension,1))
	yrs = repeat(reshape(y, (z_size_dim, y_samp_dimension,1)), outer=(1,1,x_samp_dimension))
	kernel = exp.(-mean(xrs .= yrs, dims=1).^2) ./ Float32(z_size_dim)
	return kernel
end

function compute_mmd(x,y)
	xx = compute_kernel(x,x)
	yy = compute_kernel(y,y)
	xy = compute_kernel(x,y)
	mmd_loss = mean(xx) + mean(yy) - 2*mean(xy)
	return mmd_loss
end

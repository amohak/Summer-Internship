require 'paths'
require 'rnn'

cmd = torch.CmdLine()

cmd:text('Options:')
cmd:option('--hiddensize', 299, 'number of hidden units used at output of each recurrent layer. When more than one is specified, RNN/LSTMs/GRUs are stacked')
cmd:option('--lr', 0.05, 'learning rate at t=0')

cmd:option('--cuda', true, 'use CUDA')
cmd:option('--device', 1, 'sets the device (GPU) to use')
cmd:option('--maxepoch', 51, 'maximum number of epochs to run')

cmd:option('--savepath', '/home/arjun/mohak/lstm', 'path to directory where experiment log (includes model) will be saved')
cmd:option('--inpsize', 7497, 'maximum number of epochs to run')
cmd:option('--weight', 9, 'maximum number of epochs to run')

splitRatio = {0.6,0.8}

local opt = cmd:parse(arg or {})

hiddenSize = opt.hiddensize
lr = opt.lr
vocabsize = 8870

if opt.cuda then
	require 'cunn'
	cutorch.setDevice(opt.device)
end

target_data = torch.load("murphy_target.dat")
input_data = torch.load("murphy_vector_index.dat")
steps_data = torch.load("murphy_steps.dat")
vec = torch.load("ind2vec.dat")
print("Data Loaded")

-- build simple recurrent neural network

local r = nn.FastLSTM(hiddenSize,hiddenSize)
local embedding = nn.LookupTable(vocabsize,hiddenSize)

for i=1,vocabsize do
	embedding.weight[i] = vec[i]
end

local rnn = nn.Sequential()
	:add(embedding)
	:add(r)
	:add(nn.Linear(hiddenSize,2))
	:add(nn.LogSoftMax())

-- internally, rnn will be wrapped into a Recursor to make it an AbstractRecurrent instance.
rnn = nn.Recursor(rnn)

print(rnn)

w = opt.weight
weights = torch.Tensor(2)
weights[2] = w*1.0/(w+1)
weights[1] = 1 - weights[1]
criterion = nn.ClassNLLCriterion(weights)

-- Using CUDA to evaluate the model
if opt.cuda then
   rnn:cuda()
   criterion:cuda()
end

local xplog = {}
xplog.opt = opt -- save all hyper-parameters and such

xplog.model = nn.Serial(rnn)
xplog.model:mediumSerial()
xplog.criterion = criterion
xplog.epoch = 0
paths.mkdir(opt.savepath)

nBatches = opt.inpsize
local trainBatches = splitRatio[1] * nBatches
local validBatches = splitRatio[2] * nBatches

local epoch = 1
local err = 0

while epoch < opt.maxepoch do
	local iteration = 1
	err = 0
	while iteration < trainBatches do
		inputs = input_data[iteration]
		targets = target_data[iteration]
		
		local rev_inputs = {}
		local rev_targets = {}

		length = steps_data[iteration]

		for i=1,length do
			rev_inputs[i] = inputs[length - i + 1]
			rev_targets[i] = targets[length - i + 1]
		end

		inputs = rev_inputs
		targets = rev_targets

		rnn:zeroGradParameters()
		rnn:forget()

		local outputs = {}
		local gradOutputs = {}
		local gradInputs = {}

		for i=1,length do
			outputs[i] = rnn:forward(inputs[i])
			err = err + criterion:forward(outputs[i], targets[i])
		end

		for i=length,1,-1 do
			gradOutputs[i] = criterion:backward(outputs[i], targets[i])
			gradInputs[i] = rnn:backward(inputs[i], gradOutputs[i])
		end

		rnn:updateParameters(lr)
		iteration = iteration + 1
	end
	
	print("Epoch "..epoch)
	print("Training Loss : "..err)

	--local pre = string.format("fwd_%d",epoch)
	local pre = string.format("murphy_vector_bwd_%d",epoch)	
	local filename = paths.concat(opt.savepath, pre..'.t7')
	xplog.epoch = epoch
	torch.save(filename, xplog)
	epoch = epoch + 1
end
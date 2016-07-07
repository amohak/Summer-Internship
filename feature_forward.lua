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
pos_tags = 35
chk_tags = 29

if opt.cuda then
	require 'cunn'
	cutorch.setDevice(opt.device)
end

target_data = torch.load("target.dat")
input_data = torch.load("vector_index.dat")
pos_data = torch.load("pos_index.dat")
chk_data = torch.load("chk_index.dat")
steps_data = torch.load("steps.dat")
vec = torch.load("ind2vec.dat")
print("Data Loaded")

-- build simple recurrent neural network

local r = nn.FastLSTM(3*hiddenSize,hiddenSize)

local word_embedding = nn.LookupTable(vocabsize,hiddenSize)
local pos_embedding = nn.LookupTable(pos_tags,hiddenSize)
local chk_embedding = nn.LookupTable(chk_tags,hiddenSize)

for i=1,vocabsize do
	word_embedding.weight[i] = vec[i]
end

local embedding = nn.ParallelTable()
embedding:add(word_embedding)
embedding:add(pos_embedding)
embedding:add(chk_embedding)

local join = nn.JoinTable(2)

local rnn = nn.Sequential()
	:add(embedding)
	:add(join)
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
		pos_inputs = pos_data[iteration]
		chk_inputs = chk_data[iteration]
		
		rnn:zeroGradParameters()
		rnn:forget()
		
		local outputs = {}
		local gradOutputs = {}
		local gradInputs = {}

		for i=1,steps_data[iteration] do
			outputs[i] = rnn:forward({inputs[i],pos_inputs[i],chk_inputs[i]})
			err = err + criterion:forward(outputs[i], targets[i])
		end

		for i=steps_data[iteration],1,-1 do
			gradOutputs[i] = criterion:backward(outputs[i], targets[i])
			gradInputs[i] = rnn:backward({inputs[i],pos_inputs[i],chk_inputs[i]}, gradOutputs[i])
		end

		rnn:updateParameters(lr)
		iteration = iteration + 1
	end
	
	print("Epoch "..epoch)
	print("Training Loss : "..err)

	--local pre = string.format("fwd_%d",epoch)
	local pre = string.format("feature_vector_fwd_%d",epoch)	
	local filename = paths.concat(opt.savepath, pre..'.t7')
	xplog.epoch = epoch
	torch.save(filename, xplog)
	epoch = epoch + 1
end

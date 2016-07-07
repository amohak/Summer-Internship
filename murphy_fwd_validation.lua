require 'paths'
require 'rnn'

cmd = torch.CmdLine()

cmd:text('Options:')
cmd:option('--hiddensize', 299, 'number of hidden units used at output of each recurrent layer. When more than one is specified, RNN/LSTMs/GRUs are stacked')
cmd:option('--lr', 0.05, 'learning rate at t=0')

cmd:option('--cuda', true, 'use CUDA')
cmd:option('--device', 1, 'sets the device (GPU) to use')
cmd:option('--maxepoch', 51, 'maximum number of epochs to run')

local opt = cmd:parse(arg or {})
filename = {}

for i=1,opt.maxepoch - 1 do
	filename[i] = string.format("/home/arjun/mohak/lstm/murphy_fwd_%d.t7",i)
end

splitRatio = {0.6,0.8}
thresholds = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}

if opt.cuda then
	require 'cunn'
	cutorch.setDevice(opt.device)
end

print("Loading data")
target_data = torch.load("murphy_target.dat")
input_data = torch.load("murphy_input.dat")
steps_data = torch.load("murphy_steps.dat")
print("Data loaded")

nBatches = 4899
local trainBatches = splitRatio[1] * nBatches
local validBatches = splitRatio[2] * nBatches
local testBatches = nBatches - validBatches

local epoch = 1

best_fscore = {0,0,0,0,0,0,0,0,0}
best_epoch = {0,0,0,0,0,0,0,0,0}

while epoch < opt.maxepoch do
	
	print("Epoch "..epoch)
	tp = {0,0,0,0,0,0,0,0,0}
	fp = {0,0,0,0,0,0,0,0,0}
	fn = {0,0,0,0,0,0,0,0,0}
	tn = {0,0,0,0,0,0,0,0,0}
	fscore = {0,0,0,0,0,0,0,0,0}
	recall = {0,0,0,0,0,0,0,0,0}
	precision = {0,0,0,0,0,0,0,0,0}

	local xplog = torch.load(filename[epoch])
	local rnn = xplog.model

	local iteration = 1

	while iteration < trainBatches do
		iteration = iteration + 1
	end

	while iteration < validBatches do
		inputs = input_data[iteration]
		targets = target_data[iteration]

		length = steps_data[iteration]

		local outputs = {}

		for i=1,length do
			outputs[i] = rnn:forward(inputs[i])
		end

		for key,value in ipairs(outputs) do
			exp_value = torch.exp(value)
			for k,v in ipairs(thresholds) do
				if targets[key][1] == 2 then
					if exp_value[2] > v then
						tp[k] = tp[k] + 1
					else
						fn[k] = fn[k] + 1
					end
				else
					if exp_value[2] > v then
						fp[k] = fp[k] + 1
					else
						tn[k] = tn[k] + 1
					end
				end
			end
		end
		iteration = iteration + 1						
	end

	for i=1,9 do
		recall[i] = tp[i]*100.0/(tp[i]+fn[i])
		precision[i] = tp[i]*100.0/(tp[i]+fp[i])
		fscore[i] = 2*recall[i]*precision[i]/(recall[i]+precision[i])
	end

	for i=1,9 do
		if fscore[i] > best_fscore[i] then
			best_fscore[i] = fscore[i]
			best_epoch[i] = epoch
		end
	end

	epoch = epoch + 1
end

print(best_epoch)

for th=1,9 do
	tp = 0
	fp = 0
	fn = 0
	tn = 0
	fscore = 0
	recall = 0
	precision = 0
	local epoch = best_epoch[th]
	filen = string.format("/home/arjun/mohak/lstm/murphy_fwd_%d.t7",epoch)
	print("Threshold "..th)
	print("Best Epoch : "..epoch)

	local xplog = torch.load(filen)
	local rnn = xplog.model

	local iteration = 1

	while iteration < trainBatches do
		iteration = iteration + 1
	end

	while iteration < validBatches do
		inputs = input_data[iteration]
		targets = target_data[iteration]

		length = steps_data[iteration]

		local outputs = {}
		pk = {}
		pnk = {}
		npnk = {}
		npk = {}
		for i=1,length do
			outputs[i] = rnn:forward(inputs[i])
		end

		for key,value in ipairs(outputs) do
			exp_value = torch.exp(value)
			if targets[key][1] == 2 then
				if exp_value[2] > thresholds[th] then
					tp = tp + 1
					pk[#pk + 1] = key
				else
					fn = fn + 1
					npk[#npk + 1] = key
				end
			else
				if exp_value[2] > thresholds[th] then
					fp = fp + 1
					pnk[#pnk + 1] = key
				else
					tn = tn + 1
					npnk[#npnk + 1] = key
				end
			end
		end	
		print(iteration)
		io.write("Predicted and keyword : ")
		for k,v in ipairs(pk) do
			io.write(v," ")
		end

		io.write("\nPredicted but not keyword : ")
		for k,v in ipairs(pnk) do
			io.write(v," ")
		end

		io.write("\nNot predicted and not keyword : ")
		for k,v in ipairs(npnk) do
			io.write(v," ")
		end

		io.write("\nNot predicted but keyword : ")
		for k,v in ipairs(npk) do
			io.write(v," ")
		end

		io.write("\n")					
		iteration = iteration + 1
	end

	print("Predicted and Keyword : "..tp)
	print("Predicted but not keyword : "..fp)
	print("Not predicted and not keyword : "..tn)
	print("Not predicted but keyword : ",tn)

	recall = tp*100.0/(tp+fn)
	precision = tp*100.0/(tp+fp)
	fscore = 2*recall*precision/(recall+precision)

	print("Recall : "..recall)
	print("Precision : "..precision)
	print("F1-Score : "..fscore)

end






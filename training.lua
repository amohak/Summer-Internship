require 'paths'

require 'cunn'
cutorch.setDevice(1)

appl_data = {}
steps_data = {}

local csvFile1 = io.open("application.csv", 'r')			-- insert filePath
local i = 0  
for line in csvFile1:lines('*l') do
	temp = {}
	i = i + 1
	local l = line:split(',')
	for key, val in ipairs(l) do
		t = torch.Tensor(1)
		t[1] = val
		temp[#temp + 1] = t:cuda()
	end
	appl_data[i] = temp
	steps_data[i] = #temp
end
csvFile1:close()

-- pos_data = {}
-- local csvFile1 = io.open("pos_target.csv", 'r')			-- insert filePath
-- local i = 0  
-- for line in csvFile1:lines('*l') do
-- 	temp = {}
-- 	i = i + 1
-- 	local l = line:split(',')
-- 	for key, val in ipairs(l) do
-- 		t = torch.Tensor(1)
-- 		t[1] = val
-- 		temp[#temp + 1] = t:cuda()
-- 	end
-- 	pos_data[i] = temp
-- end
-- csvFile1:close()

-- chk_data = {}
-- local csvFile1 = io.open("chk_target.csv", 'r')			-- insert filePath
-- local i = 0  
-- for line in csvFile1:lines('*l') do
-- 	temp = {}
-- 	i = i + 1
-- 	local l = line:split(',')
-- 	for key, val in ipairs(l) do
-- 		t = torch.Tensor(1)
-- 		t[1] = val
-- 		temp[#temp + 1] = t:cuda()
-- 	end
-- 	chk_data[i] = temp
-- end
-- csvFile1:close()

-- input_data = {}
-- -- steps_data = {}

-- local csvFile1 = io.open("murphy_sentence-index.csv", 'r')			-- insert filePath
-- local i = 0  
-- for line in csvFile1:lines('*l') do
-- 	temp = {}
-- 	i = i + 1
-- 	local l = line:split(';')
-- 	for key, val in ipairs(l) do
-- 		t = torch.Tensor(1)
-- 		t[1] = val
-- 		temp[#temp + 1] = t:cuda()
-- 	end
-- 	input_data[i] = temp
-- 	-- steps_data[i] = #temp
-- end
-- csvFile1:close()

-- vec = {}

-- local csvFile1 = io.open("ind2vec.csv", 'r')			-- insert filePath
-- local i = 0  
-- for line in csvFile1:lines('*l') do
-- 	temp = torch.Tensor(299)
-- 	i = i + 1
-- 	local l = line:split(',')
-- 	for key, val in ipairs(l) do
-- 		if key <= 299 then
-- 			temp[key] = val
-- 		end
-- 	end
-- 	vec[i] = temp
-- end
-- 	-- steps_data[i] = #temp
-- csvFile1:close()

-- print(#vec)

savepath = "/home/arjun/mohak/New"

-- file1 = string.format("ind2vec")
-- filename = paths.concat(savepath, file1..'.dat')
-- torch.save(filename,vec)

file1 = string.format("appl_input")
filename = paths.concat(savepath, file1..'.dat')
torch.save(filename,appl_data)

-- file1 = string.format("pos_index")
-- filename = paths.concat(savepath, file1..'.dat')
-- torch.save(filename,pos_data)

-- file2 = string.format("chk_index")
-- filename1 = paths.concat(savepath, file2..'.dat')
-- torch.save(filename1,chk_data)

file2 = string.format("appl_steps")
filename = paths.concat(savepath, file2..'.dat')
torch.save(filename,steps_data)

-- file3 = string.format("murphy_target")
-- filename = paths.concat(savepath, file3..'.dat')
-- torch.save(filename,target_data)

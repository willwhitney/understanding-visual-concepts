require 'nn'
require 'cutorch'
require 'cunn'

require 'UnsupervisedEncoder'
require 'Decoder'

torch.manualSeed(1)
-- torch.setdefaulttensortype('torch.CudaTensor')

-- parameters
local precision = 1e-5
local jac = nn.Jacobian

local dim_hidden = 200
local color_channels = 1
local feature_maps = 96
local filter_size = 5

local image_size = 150

iteration = 1

-- define inputs and module
local input = torch.rand(2, 1, image_size, image_size):cuda()

local network = nn.Sequential()
network:add(nn.SplitTable(1))
network:add(nn.ParallelTable():add(nn.Reshape(1, image_size, image_size)):add(nn.Reshape(1, image_size, image_size)))
network:add(UnsupervisedEncoder(dim_hidden, color_channels, feature_maps, filter_size, 1))
network:add(Decoder(dim_hidden, color_channels, feature_maps, filter_size))

network:cuda()

-- test backprop, with Jacobian
local err = jac.testJacobian(network, input)
print('==> error: ' .. err)
if err<precision then
    print('==> module OK')
else
    print('==> error too large, incorrect implementation')
end

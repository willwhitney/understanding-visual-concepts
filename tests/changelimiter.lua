require 'nn'

require 'ChangeLimiter'

torch.manualSeed(1)

-- parameters
local precision = 1e-5
local jac = nn.Jacobian
-- iteration = 100000

-- opt = {}
-- opt.sharpening_rate = 0
-- opt.batch_size = 1
-- opt.num_primitives = 8
--
-- opt.num_primitives = 8
-- timesteps = 10
-- vector_size = 10

-- define inputs and module
local input = torch.rand(3, 200)

local network = nn.Sequential()
network:add(nn.SplitTable(1))
network:add(nn.ChangeLimiter())

-- local par = nn.ConcatTable()
--
-- primitivePipe = nn.Sequential()
-- primitivePipe:add(nn.Narrow(2, 1, opt.num_primitives * timesteps))
-- primitivePipe:add(nn.Reshape(timesteps, opt.num_primitives, false))
-- par:add(primitivePipe)
--
-- par:add(nn.Narrow(2, opt.num_primitives * timesteps, vector_size))
-- network:add(par)
--
--
-- local module = nn.IIDCFNetwork({
--         num_primitives = opt.num_primitives,
--         encoded_dimension = vector_size,
--         num_functions = opt.num_primitives,
--         controller_units_per_layer = vector_size,
--         controller_num_layers = 1,
--         controller_dropout = 0,
--         steps_per_output = timesteps,
--         controller_nonlinearity = 'softmax',
--         function_nonlinearity = 'prelu',
--         controller_type = 'scheduled_sharpening',
--         controller_noise = 0,
--         -- all_metadata_controller = true,
--         metadata_only_controller = true,
--     })
-- network:add(module)

-- print(network)
-- test_input = {torch.rand(2,8), torch.rand(1,10)}
-- module:forward(test_input)
-- print(module:backward(test_input, torch.rand(1,10)))
-- print(network:forward(input))


-- test backprop, with Jacobian
local err = jac.testJacobian(network, input)
print('==> error: ' .. err)
if err<precision then
    print('==> module OK')
else
    print('==> error too large, incorrect implementation')
end

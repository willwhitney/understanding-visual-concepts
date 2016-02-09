require 'nn'
require 'optim'

local UnsupervisedEncoder = require 'UnsupervisedEncoder'
local UnsupervisedBatchNormEncoder = require 'UnsupervisedEncoder'
local Decoder = require 'Decoder'
local BatchNormDecoder = require 'Decoder'
local data_loaders = require 'data_loaders'

local cmd = torch.CmdLine()

cmd:option('--name', 'net', 'filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('--checkpoint_dir', 'networks', 'output directory where checkpoints get written')
cmd:option('-import', '', 'initialize network parameters from checkpoint at this path')

-- data
cmd:option('--datasetdir', '/om/user/wwhitney/facegen/CNN_DATASET', 'dataset source directory')

-- optimization
cmd:option('--learning_rate', 1e-4, 'learning rate')
cmd:option('--learning_rate_decay', 0.97, 'learning rate decay')
cmd:option('--learning_rate_decay_after', 18000, 'in number of examples, when to start decaying the learning rate')
cmd:option('--learning_rate_decay_interval', 4000, 'in number of examples, how often to decay the learning rate')
cmd:option('--decay_rate', 0.95, 'decay rate for rmsprop')
cmd:option('--grad_clip', 3, 'clip gradients at this value')
cmd:option('--criterion', 'MSE', 'criterion to use')
cmd:option('--batch_norm', false, 'criterion to use')


cmd:option('--dim_hidden', 200, 'dimension of the representation layer')
cmd:option('--feature_maps', 96, 'number of feature maps')
cmd:option('--sharpening_rate', 10, 'number of feature maps')
cmd:option('--noise', 0.1, 'variance of added Gaussian noise')


cmd:option('--max_epochs', 50, 'number of full passes through the training data')
cmd:option('--train_frac', 0.9, 'fraction of data that goes into train set')
cmd:option('--val_frac', 0.05, 'fraction of data that goes into validation set')

-- bookkeeping
cmd:option('--seed', 123, 'torch manual random number generator seed')
cmd:option('--print_every', 1, 'how many steps/minibatches between printing out the loss')
cmd:option('--eval_val_every', 9000, 'every how many iterations should we evaluate on validation data?')
-- cmd:option('-eval_val_every',10,'every how many iterations should we evaluate on validation data?')

-- data
cmd:option('--num_train_batches', 9000, 'number of batches to train with per epoch')
cmd:option('--num_train_batches_per_type', 3000, 'number of available train batches of each data type')
cmd:option('--num_test_batches', 1400, 'number of batches to test with')
cmd:option('--num_test_batches_per_type', 350, 'number of available test batches of each type')
-- cmd:option('--batch_size', 20, 'number of samples per batch')

-- GPU/CPU
cmd:option('--gpu', false, 'which gpu to use. -1 = use CPU')
cmd:text()


-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

if opt.gpu then
    require 'cutorch'
    require 'cunn'
end

if opt.name == 'net' then
    local name = 'unsup_'
    for _, v in ipairs(arg) do
        name = name .. tostring(v) .. '_'
    end
    opt.name = name .. os.date("%b_%d_%H_%M_%S")
end

local savedir = string.format('%s/%s', opt.checkpoint_dir, opt.name)
print("Saving output to "..savedir)
os.execute('mkdir -p '..savedir)
os.execute(string.format('rm %s/*', savedir))

-- log out the options used for creating this network to a file in the save directory.
-- super useful when you're moving folders around so you don't lose track of things.
local f = io.open(savedir .. '/opt.txt', 'w')
for key, val in pairs(opt) do
  f:write(tostring(key) .. ": " .. tostring(val) .. "\n")
end
f:flush()
f:close()

local logfile = io.open(savedir .. '/output.log', 'w')
true_print = print
print = function(...)
    for _, v in ipairs{...} do
        true_print(v)
        logfile:write(tostring(v))
    end
    logfile:write("\n")
    logfile:flush()
end

local dim_hidden = opt.dim_hidden
local feature_maps = opt.feature_maps
local color_channels = 1
local filter_size = 5
local scheduler_iteration = torch.zeros(1)
-- local image_size = 150


local model = nn.Sequential()
if opt.batch_norm then
    model:add(UnsupervisedBatchNormEncoder(dim_hidden, color_channels, feature_maps, filter_size, opt.noise, opt.sharpening_rate, scheduler_iteration))
    model:add(BatchNormDecoder(dim_hidden, color_channels, feature_maps))
else
    model:add(UnsupervisedEncoder(dim_hidden, color_channels, feature_maps, filter_size, opt.noise, opt.sharpening_rate, scheduler_iteration))
    model:add(Decoder(dim_hidden, color_channels, feature_maps))
end


if opt.criterion == 'MSE' then
    criterion = nn.MSECriterion()
elseif opt.criterion == 'BCE' then
    criterion = nn.BCECriterion()
    -- criterion.sizeAverage = false
else
    error("Invalid criterion specified!")
end

if opt.gpu then
    model:cuda()
    criterion:cuda()
end
params, grad_params = model:getParameters()


function validate()
    local loss = 0
    model:evaluate()

    local n = 0
    for _, variation in ipairs{"AZ_VARIED", "EL_VARIED", "LIGHT_AZ_VARIED"} do
        for i = 1, opt.num_test_batches_per_type do -- iterate over batches in the split
            -- fetch a batch
            local input = data_loaders.load_mv_batch(i, variation, 'FT_test')

            output = model:forward(input)

            local step_loss = criterion:forward(output, input[2])
            loss = loss + step_loss
            n = n + 1
        end
    end

    loss = loss / n
    return loss
end

-- do fwd/bwd and return loss, grad_params
function feval(x)
    if x ~= params then
        error("Params not equal to given feval argument.")
        params:copy(x)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local input = data_loaders.load_random_mv_batch('train')

    ------------------- forward pass -------------------
    model:training() -- make sure we are in correct mode


    output = model:forward(input)

    loss = criterion:forward(output, input[2])
    grad_output = criterion:backward(output, input[2]):clone()

    ------------------ backward pass -------------------
    model:backward(input, grad_output)
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)

    collectgarbage()
    return loss, grad_params
end


train_losses = {}
val_losses = {}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * opt.num_train_batches
-- local iterations_per_epoch = opt.num_train_batches
local loss0 = nil

for step = 1, iterations do
    scheduler_iteration[1] = step
    epoch = step / opt.num_train_batches

    local timer = torch.Timer()

    local _, loss = optim.rmsprop(feval, params, optim_state)

    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[step] = train_loss

    -- exponential learning rate decay
    if step % opt.learning_rate_decay_interval == 0 and opt.learning_rate_decay < 1 then
        if step >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed function learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    if step % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.2fs", step, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
    end

    -- every now and then or on last iteration
    if step % opt.eval_val_every == 0 or step == iterations then
        -- evaluate loss on validation data
        local val_loss = validate() -- 2 = validation
        val_losses[step] = val_loss
        print(string.format('[epoch %.3f] Validation loss: %6.8f', epoch, val_loss))

        local model_file = string.format('%s/epoch%.2f_%.4f.t7', savedir, epoch, val_loss)
        print('saving checkpoint to ' .. model_file)
        local checkpoint = {}
        checkpoint.model = model
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.step = step
        checkpoint.epoch = epoch
        torch.save(model_file, checkpoint)

        local val_loss_log = io.open(savedir ..'/val_loss.txt', 'a')
        val_loss_log:write(val_loss .. "\n")
        val_loss_log:flush()
        val_loss_log:close()
    end

    if step % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break -- halt
    end
    if loss0 == nil then
        loss0 = loss[1]
    end
    -- if loss[1] > loss0 * 8 then
    --     print('loss is exploding, aborting.')
    --     print("loss0:", loss0, "loss[1]:", loss[1])
    --     break -- halt
    -- end
end
--]]

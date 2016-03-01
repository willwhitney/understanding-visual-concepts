require 'nn'
require 'optim'
require 'modules/KLDCriterion'
require 'MotionBCECriterion'

schedule_weight_iter = 0  -- GLOBAL VARIABLES
schedule_weight_exp = 1  -- GLOBAL VARIABLES
local Encoder = require 'BallsEncoder'
local Decoder = require 'Decoder'

local data_loaders = require 'data_loaders'

local cmd = torch.CmdLine()

cmd:option('--name', 'netvar', 'filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('--checkpoint_dir', 'logslink', 'output directory where checkpoints get written')
cmd:option('-import', '', 'initialize network parameters from checkpoint at this path')

-- data
cmd:option('--datasetdir', '/om/data/public/mbchang/udcign-data/balls', 'dataset source directory')  -- change
cmd:option('--numballs', 1, 'dataset source directory')
cmd:option('--subsample', 3, 'subsample')  -- hard code this into data_loader
cmd:option('--frame_interval', 1, 'the number of timesteps between input[1] and input[2]')

-- optimization
cmd:option('--learning_rate', 1e-5, 'learning rate')
cmd:option('--learning_rate_decay', 0.97, 'learning rate decay')
cmd:option('--learning_rate_decay_after', 1000, 'in number of examples, when to start decaying the learning rate')
cmd:option('--learning_rate_decay_interval', 1000, 'in number of examples, how often to decay the learning rate')
cmd:option('--decay_rate', 0.95, 'decay rate for rmsprop')  -- this is momentum
cmd:option('--grad_clip', 3, 'clip gradients at this value')

cmd:option('--L2', 0.01, 'amount of L2 regularization')
cmd:option('--criterion', 'BCE', 'criterion to use')
cmd:option('--batch_norm', false, 'use model with batch normalization')

cmd:option('--heads', 1, 'how many filtering heads to use')
cmd:option('--motion_scale', 3, 'how much to accentuate loss on changing pixels')

cmd:option('--dim_hidden', 32, 'dimension of the representation layer')
cmd:option('--feature_maps', 16, 'number of feature maps')
cmd:option('--color_channels', 1, '1 for grayscale, 3 for color')
cmd:option('--sharpening_rate', 10, 'number of feature maps')
cmd:option('--noise', 0.1, 'variance of added Gaussian noise')


cmd:option('--max_epochs', 50, 'number of full passes through the training data')

-- bookkeeping
cmd:option('--seed', 123, 'torch manual random number generator seed')
cmd:option('--print_every', 1, 'how many steps/minibatches between printing out the loss')
cmd:option('--eval_val_every', 1000, 'every how many iterations should we evaluate on validation data?')  -- CHANGE

-- data
cmd:option('--num_train_batches', 9000, 'number of batches to train with per epoch')  -- CHANGE
cmd:option('--num_test_batches', 1000, 'number of batches to test with')  -- CHANGE

-- GPU/CPU
cmd:option('--gpu', true, 'which gpu to use. -1 = use CPU')
cmd:text()


-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

print(opt)
print(opt.gpu)

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


local scheduler_iteration = torch.zeros(1)

local model = nn.Sequential()
local encoder = Encoder(opt.dim_hidden, opt.color_channels, opt.feature_maps, opt.noise, opt.sharpening_rate, scheduler_iteration, opt.batch_norm, opt.heads)
local decoder = Decoder(opt.dim_hidden, opt.color_channels, opt.feature_maps, opt.batch_norm)
model:add(encoder)
model:add(decoder)

local encoder1 = encoder:findModules('nn.Sequential')[2]  -- enc1, not enc1var
local encoder2 = encoder:findModules('nn.Sequential')[4]  -- enc2, not enc2var

-- graph.dot(model.modules[1].fg, 'encoder', 'reports/encoder')

if opt.criterion == 'MSE' then
    criterion = nn.MSECriterion()
elseif opt.criterion == 'BCE' then
    criterion = nn.BCECriterion()
    -- criterion = nn.MotionBCECriterion(opt.motion_scale)
else
    error("Invalid criterion specified!")
end
KLD = nn.KLDCriterion()  -- variational

if opt.gpu then
    model:cuda()
    criterion:cuda()
    KLD:cuda()  -- variational
end
params, grad_params = model:getParameters()


function validate()
    local loss = 0
    model:evaluate()

    for i = 1, opt.num_test_batches do -- iterate over batches in the split
        -- fetch a batch
        local input = data_loaders.load_balls_batch(i, 'val')

        local output = model:forward(input)

        local err = -criterion:forward(output, input[2])

        local enc1out = encoder1.output
        local enc2out = encoder2.output

        -- encoder1
        local KLDerr1 = KLD:forward(enc1out, input[1])

        -- encoder2
        local KLDerr2 = KLD:forward(enc2out, input[2])

        local KLDerr = KLDerr1/2 + KLDerr2/2
        local lowerbound = err + KLDerr -- want to maximize this?
        loss = loss + lowerbound

        -- TODO: do variational stuff here
        -- local step_loss = criterion:forward(output, input[2])
        -- loss = loss + step_loss
    end

    loss = loss / opt.num_test_batches
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
    local input = data_loaders.load_random_balls_batch('train')

    ------------------- forward pass -------------------
    model:training() -- make sure we are in correct mode


    local output = model:forward(input)

    -- ######################################################################--
    -- Your code here for variational

    local loss = -criterion:forward(output, input[2])
    local grad_output = criterion:backward(output, input[2]):clone():mul(-1) --gradient ascent. why is this the likelihood cost function?

    -- local loss = criterion:forward(output, input[2])
    -- local grad_output = criterion:backward(output, input[2]):clone()

    ------------------ backward pass -------------------
    model:backward(input, grad_output)

    local enc1out = encoder1.output
    local enc2out = encoder2.output

    -- encoder1
    local KLDerr1 = KLD:forward(enc1out, input[1])
    local dKLD_dw1 = KLD:backward(enc1out, input[1])
    encoder1:backward(input[1],dKLD_dw1) -- does this go backward through the entire encoder1?

    -- encoder2
    local KLDerr2 = KLD:forward(enc2out, input[2])
    local dKLD_dw2 = KLD:backward(enc2out, input[2])
    encoder1:backward(input[2],dKLD_dw2)  -- does this go backward through the entire encoder2?

    local KLDerr = KLDerr1/2 + KLDerr2/2
    local lowerbound = loss + KLDerr -- want to maximize this?

    if opt.verbose then
        print("BCE",loss/input[1]:size(1))
        print("KLD", KLDerr/input[1]:size(1))
        print("lowerbound", lowerbound/input[1]:size(1))
    end

    --#######################################################################--


    ------------------ regularize -------------------
    if opt.L2 > 0 then
        -- Loss:
        loss = loss + opt.L2 * params:norm(2)^2/2
        -- Gradients:
        grad_params:add( params:clone():mul(opt.L2) )
    end

    grad_params:clamp(-opt.grad_clip, opt.grad_clip)

    collectgarbage()
    -- return loss, grad_params
    return lowerbound, grad_params  -- this is from variational autoencoder
end


train_losses = {}
val_losses = {}
-- local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local optim_state = {learningRate = -opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * opt.num_train_batches
-- local iterations_per_epoch = opt.num_train_batches
local loss0 = nil

for step = 1, iterations do
    scheduler_iteration[1] = step
    schedule_weight_iter = step
    epoch = step / opt.num_train_batches

    local timer = torch.Timer()

    local _, loss = optim.rmsprop(feval, params, optim_state)
    -- print(params:norm())  -- params are definitely getting updated

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
        print(string.format("%d/%d (epoch %.3f), lowerbound = %6.8f, grad/param norm = %6.4e, time/batch = %.2fs, sharpening exp = %2.4f, lr = %2.4e", step, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time, schedule_weight_exp, optim_state.learningRate))
        -- print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.2fs", step, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
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

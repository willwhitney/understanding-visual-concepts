require 'nn'
require 'nngraph'

require 'Print'
require 'ChangeLimiter'
require 'Noise'
require 'ScheduledWeightSharpener'

local UnsupervisedBatchNormEncoder = function(dim_hidden, color_channels, feature_maps, filter_size, noise, sharpening_rate, scheduler_iteration)

    local inputs = {
            nn.Identity()():annotate{name="input1"},
            nn.Identity()():annotate{name="input2"},
        }

    -- make two copies of an encoder

    local enc1 = nn.Sequential()
    enc1:add(nn.SpatialConvolution(color_channels, feature_maps, filter_size, filter_size))
    enc1:add(nn.SpatialMaxPooling(2,2,2,2))
    enc1:add(nn.SpatialBatchNormalization(feature_maps))
    enc1:add(nn.Threshold(0,1e-6))

    enc1:add(nn.SpatialConvolution(feature_maps, feature_maps/2, filter_size, filter_size))
    enc1:add(nn.SpatialMaxPooling(2,2,2,2))
    enc1:add(nn.SpatialBatchNormalization(feature_maps/2))
    enc1:add(nn.Threshold(0,1e-6))

    enc1:add(nn.SpatialConvolution(feature_maps/2, feature_maps/4, filter_size, filter_size))
    enc1:add(nn.SpatialMaxPooling(2,2,2,2))
    enc1:add(nn.SpatialBatchNormalization(feature_maps/4))
    enc1:add(nn.Threshold(0,1e-6))

    enc1:add(nn.Reshape((feature_maps/4) * 15*15))
    enc1:add(nn.Linear((feature_maps/4) * 15*15, dim_hidden))

    local enc2 = enc1:clone('weight', 'bias', 'gradWeight', 'gradBias')
    enc1 = enc1(inputs[1])
    enc2 = enc2(inputs[1])

    -- and join them together for analysis
    local encoded_join = nn.JoinTable(2)({enc1, enc2}):annotate{name="encoded_join"}

    -- make the "controller", which looks at the two frames and decides what's changing
    local controller_lin1 = nn.Linear(dim_hidden * 2, dim_hidden)(encoded_join):annotate{name="controller_lin1"}
    local controller_nonlin = nn.Sigmoid()(controller_lin1):annotate{name="controller_nonlin"}
    local controller_noise = nn.Noise(noise)(controller_nonlin):annotate{name="controller_noise"}
    local controller_sharpener = nn.ScheduledWeightSharpener(sharpening_rate, scheduler_iteration)(controller_noise):annotate{name="controller_sharpener"}

    local controller_addc = nn.AddConstant(1e-20)(controller_sharpener):annotate{name="controller_addc"}
    local controller_norm = nn.Normalize(1, 1e-100)(controller_addc):annotate{name="controller_norm"}

    local change_limiter = nn.ChangeLimiter()({controller_norm, enc1, enc2}):annotate{name="change_limiter"}


    local output = {change_limiter}
    -- local output = {nn.Print("End of encoder")(change_limiter)}
    return nn.gModule(inputs, output)
end

return UnsupervisedBatchNormEncoder

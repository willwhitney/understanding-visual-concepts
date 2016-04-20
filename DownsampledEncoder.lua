require 'nn'
require 'nngraph'

require 'Print'
require 'ChangeLimiter'
require 'Noise'
require 'ScheduledWeightSharpener'

local DownsampledEncoder = function(dim_hidden, color_channels, feature_maps, noise, sharpening_rate, scheduler_iteration, num_heads)

    local filter_size = 5
    local stride = 1
    local padding = 2
    local encoded_size = 10
    local inputs = {
            nn.Identity()():annotate{name="input1"},
            nn.Identity()():annotate{name="input2"},
        }

    -- make two copies of an encoder

    local enc1 = nn.Sequential()
    enc1:add(nn.SpatialConvolution(color_channels, feature_maps, filter_size, filter_size, stride, stride, padding, padding))
    enc1:add(nn.SpatialMaxPooling(2,2,2,2))
    enc1:add(nn.Threshold(0,1e-6))

    enc1:add(nn.SpatialConvolution(feature_maps, feature_maps/2, filter_size, filter_size, stride, stride, padding, padding))
    enc1:add(nn.SpatialMaxPooling(2,2,2,2))
    enc1:add(nn.Threshold(0,1e-6))

    enc1:add(nn.SpatialConvolution(feature_maps/2, feature_maps/4, filter_size, filter_size, stride, stride, padding, padding))
    enc1:add(nn.SpatialMaxPooling(2,2,2,2))
    enc1:add(nn.Threshold(0,1e-6))

    enc1:add(nn.Reshape((feature_maps/4) * encoded_size * encoded_size))
    enc1:add(nn.Linear((feature_maps/4) * encoded_size * encoded_size, dim_hidden))

    local enc2 = enc1:clone('weight', 'bias', 'gradWeight', 'gradBias')
    enc1 = enc1(inputs[1])
    enc2 = enc2(inputs[2])


    -- make the heads to analyze the encodings
    local heads = {}
    heads[1] = nn.Sequential()
    heads[1]:add(nn.JoinTable(2))
    heads[1]:add(nn.Linear(dim_hidden * 2, dim_hidden))
    heads[1]:add(nn.Sigmoid())
    heads[1]:add(nn.Noise(noise))
    heads[1]:add(nn.ScheduledWeightSharpener(sharpening_rate, scheduler_iteration))
    heads[1]:add(nn.AddConstant(1e-20))
    heads[1]:add(nn.Normalize(1, 1e-100))

    for i = 2, num_heads do
        heads[i] = heads[1]:clone()
    end

    for i = 1, num_heads do
        heads[i] = heads[i]{enc1, enc2}
    end

    local dist
    if num_heads > 1 then
        -- combine the distributions from all heads
        local dist_adder = nn.CAddTable()(heads)
        local dist_clamp = nn.Clamp(0, 1)(dist_adder)
        dist = dist_clamp
    else
        dist = heads[1]
    end

    -- and use it to filter the encodings
    local change_limiter = nn.ChangeLimiter()({dist, enc1, enc2}):annotate{name="change_limiter"}

    local output = {change_limiter}
    return nn.gModule(inputs, output)
end

return DownsampledEncoder

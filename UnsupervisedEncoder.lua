require 'nn'
require 'nngraph'

require 'Print'
require 'ChangeLimiter'
require 'Noise'
require 'ScheduledWeightSharpener'

local UnsupervisedEncoder = function(dim_hidden, color_channels, feature_maps, filter_size, noise, sharpening_rate)

    local inputs = {
            nn.Identity()():annotate{name="input1"},
            nn.Identity()():annotate{name="input2"},
        }

    -- make two copies of an encoder

    -- copy #1
    local enc1_conv1 = nn.SpatialConvolution(color_channels, feature_maps, filter_size, filter_size)(inputs[1]):annotate{name="enc1_conv1"}
    local enc1_pool1 = nn.SpatialMaxPooling(2,2,2,2)(enc1_conv1):annotate{name="enc1_pool1"}
    local enc1_thresh1 = nn.Threshold(0,1e-6)(enc1_pool1):annotate{name="enc1_thresh1"}

    local enc1_conv2 = nn.SpatialConvolution(feature_maps,feature_maps/2,filter_size,filter_size)(enc1_thresh1):annotate{name="enc1_conv2"}
    local enc1_pool2 = nn.SpatialMaxPooling(2,2,2,2)(enc1_conv2):annotate{name="enc1_pool2"}
    local enc1_thresh2 = nn.Threshold(0,1e-6)(enc1_pool2):annotate{name="enc1_thresh2"}

    local enc1_conv3 = nn.SpatialConvolution(feature_maps/2,feature_maps/4,filter_size,filter_size)(enc1_thresh2):annotate{name="enc1_conv3"}
    local enc1_pool3 = nn.SpatialMaxPooling(2,2,2,2)(enc1_conv3):annotate{name="enc1_pool3"}
    local enc1_thresh3 = nn.Threshold(0,1e-6)(enc1_pool3):annotate{name="enc1_thresh3"}

    local enc1_reshape = nn.Reshape((feature_maps/4) * 15*15)(enc1_thresh3):annotate{name="enc1_reshape"}
    local enc1_out = nn.Linear((feature_maps/4) * 15*15, dim_hidden)(enc1_reshape):annotate{name="enc1_out"}

    -- copy #2
    local enc2_conv1 = nn.SpatialConvolution(color_channels, feature_maps, filter_size, filter_size)(inputs[2]):annotate{name="enc2_conv1"}
    local enc2_pool1 = nn.SpatialMaxPooling(2,2,2,2)(enc2_conv1):annotate{name="enc2_pool1"}
    local enc2_thresh1 = nn.Threshold(0,1e-6)(enc2_pool1):annotate{name="enc2_thresh1"}

    local enc2_conv2 = nn.SpatialConvolution(feature_maps,feature_maps/2,filter_size,filter_size)(enc2_thresh1):annotate{name="enc2_conv2"}
    local enc2_pool2 = nn.SpatialMaxPooling(2,2,2,2)(enc2_conv2):annotate{name="enc2_pool2"}
    local enc2_thresh2 = nn.Threshold(0,1e-6)(enc2_pool2):annotate{name="enc2_thresh2"}

    local enc2_conv3 = nn.SpatialConvolution(feature_maps/2,feature_maps/4,filter_size,filter_size)(enc2_thresh2):annotate{name="enc2_conv3"}
    local enc2_pool3 = nn.SpatialMaxPooling(2,2,2,2)(enc2_conv3):annotate{name="enc2_pool3"}
    local enc2_thresh3 = nn.Threshold(0,1e-6)(enc2_pool3):annotate{name="enc2_thresh3"}

    local enc2_reshape = nn.Reshape((feature_maps/4) * 15*15)(enc2_thresh3):annotate{name="enc2_reshape"}
    local enc2_out = nn.Linear((feature_maps/4) * 15*15, dim_hidden)(enc2_reshape):annotate{name="enc2_out"}

    -- tie their parameters together
    -- print(enc2_conv1.data.module)
    enc2_conv1.data.module:share(enc1_conv1.data.module, 'weight', 'bias', 'gradWeight', 'gradBias')
    enc2_conv2.data.module:share(enc1_conv2.data.module, 'weight', 'bias', 'gradWeight', 'gradBias')
    enc2_conv3.data.module:share(enc1_conv3.data.module, 'weight', 'bias', 'gradWeight', 'gradBias')
    enc2_out.data.module:share(enc1_out.data.module, 'weight', 'bias', 'gradWeight', 'gradBias')

    -- and join them together for analysis
    local encoded_join = nn.JoinTable(2)({enc1_out, enc2_out}):annotate{name="encoded_join"}

    -- make the "controller", which looks at the two frames and decides what's changing
    local controller_lin1 = nn.Linear(dim_hidden * 2, dim_hidden)(encoded_join):annotate{name="controller_lin1"}
    local controller_nonlin = nn.Sigmoid()(controller_lin1):annotate{name="controller_nonlin"}
    local controller_noise = nn.Noise(noise)(controller_nonlin):annotate{name="controller_noise"}
    local controller_sharpener = nn.ScheduledWeightSharpener(sharpening_rate)(controller_noise):annotate{name="controller_sharpener"}

    local controller_addc = nn.AddConstant(1e-20)(controller_sharpener):annotate{name="controller_addc"}
    local controller_norm = nn.Normalize(1, 1e-100)(controller_addc):annotate{name="controller_norm"}

    local change_limiter = nn.ChangeLimiter()({controller_norm, enc1_out, enc2_out}):annotate{name="change_limiter"}


    local output = {change_limiter}
    -- local output = {nn.Print("End of encoder")(change_limiter)}
    return nn.gModule(inputs, output)
end

return UnsupervisedEncoder

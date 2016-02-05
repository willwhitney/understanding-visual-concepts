require 'nn'
require 'nngraph'

require 'Print'
require 'ChangeLimiter'
require 'ScheduledWeightSharpener'

function UnsupervisedEncoder(dim_hidden, color_channels, feature_maps, filter_size)

    local inputs = {
            nn.Identity()(),
            nn.Identity()(),
        }

    -- make two copies of an encoder

    -- copy #1
    local enc1_conv1 = nn.SpatialConvolution(color_channels, feature_maps, filter_size, filter_size)(inputs[1])
    local enc1_pool1 = nn.SpatialMaxPooling(2,2,2,2)(enc1_conv1)
    local enc1_thresh1 = nn.Threshold(0,1e-6)(enc1_pool1)

    local enc1_conv2 = nn.SpatialConvolution(feature_maps,feature_maps/2,filter_size,filter_size)(enc1_thresh1)
    local enc1_pool2 = nn.SpatialMaxPooling(2,2,2,2)(enc1_conv2)
    local enc1_thresh2 = nn.Threshold(0,1e-6)(enc1_pool2)

    local enc1_conv3 = nn.SpatialConvolution(feature_maps/2,feature_maps/4,filter_size,filter_size)(enc1_thresh2)
    local enc1_pool3 = nn.SpatialMaxPooling(2,2,2,2)(enc1_conv3)
    local enc1_thresh3 = nn.Threshold(0,1e-6)(enc1_pool3)

    local enc1_reshape = nn.Reshape((feature_maps/4) * 15*15)(enc1_thresh3)
    local enc1_out = nn.Linear((feature_maps/4) * 15*15, dim_hidden)(enc1_reshape)

    -- copy #2
    local enc2_conv1 = nn.SpatialConvolution(color_channels, feature_maps, filter_size, filter_size)(inputs[2])
    local enc2_pool1 = nn.SpatialMaxPooling(2,2,2,2)(enc2_conv1)
    local enc2_thresh1 = nn.Threshold(0,1e-6)(enc2_pool1)

    local enc2_conv2 = nn.SpatialConvolution(feature_maps,feature_maps/2,filter_size,filter_size)(enc2_thresh1)
    local enc2_pool2 = nn.SpatialMaxPooling(2,2,2,2)(enc2_conv2)
    local enc2_thresh2 = nn.Threshold(0,1e-6)(enc2_pool2)

    local enc2_conv3 = nn.SpatialConvolution(feature_maps/2,feature_maps/4,filter_size,filter_size)(enc2_thresh2)
    local enc2_pool3 = nn.SpatialMaxPooling(2,2,2,2)(enc2_conv3)
    local enc2_thresh3 = nn.Threshold(0,1e-6)(enc2_pool3)

    local enc2_reshape = nn.Reshape((feature_maps/4) * 15*15)(enc2_thresh3)
    local enc2_out = nn.Linear((feature_maps/4) * 15*15, dim_hidden)(enc2_reshape)

    -- tie their parameters together
    -- print(enc2_conv1.data.module)
    enc2_conv1.data.module:share(enc1_conv1.data.module, 'weight', 'bias')
    enc2_conv2.data.module:share(enc1_conv2.data.module, 'weight', 'bias')
    enc2_conv3.data.module:share(enc1_conv3.data.module, 'weight', 'bias')

    -- and join them together for analysis
    local encoded_join = nn.JoinTable(2)({enc1_out, enc2_out})

    -- make the "controller", which looks at the two frames and decides what's changing
    local controller_lin1 = nn.Linear(dim_hidden * 2, dim_hidden)(encoded_join)
    local controller_nonlin = nn.Sigmoid()(controller_lin1)
    local controller_sharpener = nn.ScheduledWeightSharpener(1)(controller_nonlin)

    -- local controller_addc = nn.AddConstant(1e-20)(controller_sharpener)
    -- local controller_norm = nn.Normalize(1, 1e-100)(controller_addc)

    local change_limiter = nn.ChangeLimiter()({controller_sharpener, enc1_out, enc2_out})


    local output = {change_limiter}
    -- local output = {nn.Print("End of encoder")(change_limiter)}
    return nn.gModule(inputs, output)
end

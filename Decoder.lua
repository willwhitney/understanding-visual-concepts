require 'nn'

local Decoder = function(dim_hidden, color_channels, feature_maps)
    local decoder = nn.Sequential()
    decoder:add(nn.Linear(dim_hidden, (feature_maps/4)*15*15 ))
    decoder:add(nn.Threshold(0,1e-6))

    decoder:add(nn.Reshape((feature_maps/4),15,15))

    decoder:add(nn.SpatialUpSamplingNearest(2))
    decoder:add(nn.SpatialConvolution(feature_maps/4,feature_maps/2, 7, 7))
    decoder:add(nn.Threshold(0,1e-6))

    decoder:add(nn.SpatialUpSamplingNearest(2))
    decoder:add(nn.SpatialConvolution(feature_maps/2,feature_maps,7,7))
    decoder:add(nn.Threshold(0,1e-6))

    decoder:add(nn.SpatialUpSamplingNearest(2))
    decoder:add(nn.SpatialConvolution(feature_maps,feature_maps,7,7))
    decoder:add(nn.Threshold(0,1e-6))

    decoder:add(nn.SpatialUpSamplingNearest(2))
    decoder:add(nn.SpatialConvolution(feature_maps,color_channels,7,7))
    decoder:add(nn.Sigmoid())
    return decoder
end

return Decoder

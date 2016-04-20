require 'nn'

-- transforms ()
local DownsampledDecoder = function(dim_hidden, color_channels, feature_maps)
    local filter_size = 6
    local encoded_size = 10
    local decoder = nn.Sequential()
    decoder:add(nn.Linear(dim_hidden, (feature_maps/4)*encoded_size*encoded_size ))
    decoder:add(nn.Threshold(0,1e-6))

    decoder:add(nn.Reshape((feature_maps/4),encoded_size,encoded_size))

    decoder:add(nn.SpatialUpSamplingNearest(2))
    decoder:add(nn.SpatialConvolution(feature_maps/4,feature_maps/2, filter_size, filter_size))
    decoder:add(nn.Threshold(0,1e-6))

    decoder:add(nn.SpatialUpSamplingNearest(2))
    decoder:add(nn.SpatialConvolution(feature_maps/2,feature_maps,filter_size,filter_size))
    decoder:add(nn.Threshold(0,1e-6))

    decoder:add(nn.SpatialUpSamplingNearest(2))
    decoder:add(nn.SpatialConvolution(feature_maps,feature_maps,filter_size,filter_size))
    decoder:add(nn.Threshold(0,1e-6))

    decoder:add(nn.SpatialUpSamplingNearest(2))
    decoder:add(nn.SpatialConvolution(feature_maps,color_channels,filter_size+1,filter_size+1))
    decoder:add(nn.Sigmoid())
    return decoder
end

return DownsampledDecoder

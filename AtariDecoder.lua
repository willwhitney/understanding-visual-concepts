require 'nn'

local AtariDecoder = function(dim_hidden, color_channels, feature_maps, batch_norm)
    local decoder = nn.Sequential()
    decoder:add(nn.Linear(dim_hidden, (feature_maps/4)*19*16 ))
    decoder:add(nn.Threshold(0,1e-6))

    decoder:add(nn.Reshape((feature_maps/4),19,16))

    decoder:add(nn.SpatialUpSamplingNearest(2))
    decoder:add(nn.SpatialConvolution(feature_maps/4,feature_maps/2, 7, 7))
    if batch_norm then
        decoder:add(nn.SpatialBatchNormalization(feature_maps/2))
    end
    decoder:add(nn.Threshold(0,1e-6))

    decoder:add(nn.SpatialUpSamplingNearest(2))
    decoder:add(nn.SpatialConvolution(feature_maps/2,feature_maps,8,8))
    if batch_norm then
        decoder:add(nn.SpatialBatchNormalization(feature_maps))
    end
    decoder:add(nn.Threshold(0,1e-6))

    decoder:add(nn.SpatialUpSamplingNearest(2))
    decoder:add(nn.SpatialConvolution(feature_maps,feature_maps,8,7))
    if batch_norm then
        decoder:add(nn.SpatialBatchNormalization(feature_maps))
    end
    decoder:add(nn.Threshold(0,1e-6))

    decoder:add(nn.SpatialUpSamplingNearest(2))
    decoder:add(nn.SpatialConvolution(feature_maps,color_channels,7,7))
    decoder:add(nn.Sigmoid())
    return decoder
end

return AtariDecoder

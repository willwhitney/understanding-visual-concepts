require 'nn'

-- transforms ()
local DownsampledAutoencoder = function(dim_hidden, color_channels, feature_maps)
    local encoder_filter_size = 5
    local stride = 1
    local padding = 2
    local encoded_size = 10

    local model = nn.Sequential()
    local encoder = nn.Sequential()

    encoder:add(nn.SpatialConvolution(color_channels, feature_maps, encoder_filter_size, encoder_filter_size, stride, stride, padding, padding))
    encoder:add(nn.SpatialMaxPooling(2,2,2,2))
    encoder:add(nn.Threshold(0,1e-6))

    encoder:add(nn.SpatialConvolution(feature_maps, feature_maps/2, encoder_filter_size, encoder_filter_size, stride, stride, padding, padding))
    encoder:add(nn.SpatialMaxPooling(2,2,2,2))
    encoder:add(nn.Threshold(0,1e-6))

    encoder:add(nn.SpatialConvolution(feature_maps/2, feature_maps/4, encoder_filter_size, encoder_filter_size, stride, stride, padding, padding))
    encoder:add(nn.SpatialMaxPooling(2,2,2,2))
    encoder:add(nn.Threshold(0,1e-6))

    encoder:add(nn.Reshape((feature_maps/4) * encoded_size * encoded_size))
    encoder:add(nn.Linear((feature_maps/4) * encoded_size * encoded_size, dim_hidden))

    local decoder_filter_size = 6
    local decoder = nn.Sequential()
    decoder:add(nn.Linear(dim_hidden, (feature_maps/4)*encoded_size*encoded_size ))
    decoder:add(nn.Threshold(0,1e-6))

    decoder:add(nn.Reshape((feature_maps/4),encoded_size,encoded_size))

    decoder:add(nn.SpatialUpSamplingNearest(2))
    decoder:add(nn.SpatialConvolution(feature_maps/4,feature_maps/2, decoder_filter_size, decoder_filter_size))
    decoder:add(nn.Threshold(0,1e-6))

    decoder:add(nn.SpatialUpSamplingNearest(2))
    decoder:add(nn.SpatialConvolution(feature_maps/2,feature_maps,decoder_filter_size,decoder_filter_size))
    decoder:add(nn.Threshold(0,1e-6))

    decoder:add(nn.SpatialUpSamplingNearest(2))
    decoder:add(nn.SpatialConvolution(feature_maps,feature_maps,decoder_filter_size,decoder_filter_size))
    decoder:add(nn.Threshold(0,1e-6))

    decoder:add(nn.SpatialUpSamplingNearest(2))
    decoder:add(nn.SpatialConvolution(feature_maps,color_channels,decoder_filter_size+1,decoder_filter_size+1))
    decoder:add(nn.Sigmoid())

    model:add(encoder)
    model:add(decoder)

    return model
end

return DownsampledAutoencoder

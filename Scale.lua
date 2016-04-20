--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require "nn"
local image = require "image"

local scale = torch.class('nn.Scale', 'nn.Module')


function scale:__init(height, width)
    self.height = height
    self.width = width
end

function scale:scale_one(input)
    -- output:zero():add(0.299, input[1]):add(0.587, input[2]):add(0.114, input[3])
    local output = image.rgb2y(input:float()) -- turn it into grayscale (luminance)
    output = (image.scale(output, self.width, self.height, 'bilinear'))
    return output
end

function scale:forward(x)
    local is_cuda = (x:type() == "torch.CudaTensor")
    -- self.output = x
    if x:dim() > 3 then
        self.output = torch.Tensor(x:size(1), 1, self.width, self.height):float()
        for i = 1, x:size(1) do
            -- puts the scaled version directly in output
            self.output[i] = self:scale_one(x[i])
        end
    else
        -- self.output = torch.Tensor(1, self.width, self.height)
        self.output = self:scale_one(x)
    end
    if is_cuda then
        self.output = self.output:cuda()
    end

    return self.output
end

function scale:updateOutput(input)
    return self:forward(input)
end

function scale:float()
end

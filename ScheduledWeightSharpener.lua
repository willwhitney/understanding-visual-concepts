--[[
Input: A table {x, y} of a Tensor x and a scalar y.
Output: x^y (elementwise)

taken from https://github.com/kaishengtai/torch-ntm/blob/master/layers/PowTable.lua
--]]

local ScheduledWeightSharpener, parent = torch.class('nn.ScheduledWeightSharpener', 'nn.Module')

function ScheduledWeightSharpener:__init(sharpening_rate)
    parent.__init(self)
    self.slope = sharpening_rate
end

function ScheduledWeightSharpener:getP(iter)
    return math.min(1 + (iter / 10000) * self.slope, 100)
end

function ScheduledWeightSharpener:updateOutput(input)
    local v = input:clone()
    v:clamp(0,1000000)

    -- smoothly increase the sharpening from 1 to 100
    -- iteration is defined globally in the training loop
    local p = self:getP(iteration)
    -- print('v:', v)
    -- print('p:', p)
    self.output = torch.pow(v, p)
    if self.output[1][1] ~= self.output[1][1] then
        print('Made a nan set of weights.')
        print('v:', v)
        print('p:', p)
        os.exit(1)
    end
    -- print(self.output)
    return self.output
end

function ScheduledWeightSharpener:updateGradInput(input, gradOutput)
    local v = input:clone()
    v:clamp(0,1000000)
    local p = self:getP(iteration)

    self.gradInput = torch.cmul(gradOutput, torch.pow(v, p - 1)) * p

    if self.gradInput[1][1] ~= self.gradInput[1][1] then
        print('Made a nan set of gradients.')
        print('v:', v)
        print('p:', p)
        print('gradInput:', self.gradInput)
        print('gradOutput:', gradOutput)
        os.exit(1)
    end
    -- local pgrad = 0
    -- for i = 1, v:size(1) do
    --     if v[i] > 0 then
    --         pgrad = pgrad + math.log(v[i]) * self.output[1][i] * gradOutput[1][i]
    --     end
    -- end
    -- pgrad = pgrad + 0.001
    -- print('pgrad: ', pgrad, 'modified pgrad: ', )
    -- self.gradInput[2][1] = pgrad
    return self.gradInput
end

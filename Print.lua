-- require 'nn'
local Print = torch.class('nn.Print', 'nn.Module')

function Print:__init(name, just_dimensions)
    self.name = name
    self.just_dimensions = just_dimensions or false
end

function Print:updateOutput(input)
    if self.just_dimensions then
        print(self.name.." input dimensions: ")
        if type(input) == 'table' then
            print("table:", input)
        else
            print(input:size())
        end
    else
        print(self.name.." input: ")
        print(input)
    end
    self.output = input
    return input
end

function Print:updateGradInput(_, gradOutput)
    if self.just_dimensions then
        print(self.name.." gradOutput dimensions:")
        if type(gradOutput) == 'table' then
            print("table:", #gradOutput)
        else
            print(gradOutput:size())
        end
    else
        print(self.name.." gradOutput:")
        print(gradOutput)
    end
    self.gradInput = gradOutput
    return self.gradInput
end

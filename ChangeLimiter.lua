require 'nn'

ChangeLimiter = torch.class('nn.ChangeLimiter', 'nn.Module')

function ChangeLimiter:updateOutput(input)
    -- print(input)
    local distribution, input1, input2 = table.unpack(input)
    self.output = torch.cmul(input1, ((distribution * -1) + 1)) + torch.cmul(input2, distribution)   -- why don't you do an AddTable?
    -- print(self.output)
    return self.output
end

function ChangeLimiter:updateGradInput(input, gradOutput)
    local distribution, input1, input2 = table.unpack(input)
    self.gradInput = {
            torch.cmul(gradOutput, (input2 - input1)),
            torch.cmul(gradOutput, ((distribution * -1) + 1)),
            torch.cmul(gradOutput, distribution),
        }

    return self.gradInput
end

--[[
This class increases the error function's sensitivity to elements of the
target which change from frame to frame within a batch.

It assumes that each batch takes the form of successive
frames of video.

After calculating the pointwise cross-entropy, it applies a multiplicative
mask, causing points which have changed from frame to frame to have a much
greater impact on the summed BCE.

We treat the first and last frame in each batch as only having motion
relative to the second and next-to-last frames, respectively.
All other frames have increased sensitivity in regions which differ either
from the previous frame or from the following frame.
--]]

local MotionBCECriterion, parent = torch.class('nn.MotionBCECriterion', 'nn.Criterion')

local eps = 1e-12

function MotionBCECriterion:__init(motionScale)
   parent.__init(self)
   self.sizeAverage = true
   self.motionScale = motionScale
   self.mask = torch.Tensor()
end

function MotionBCECriterion:updateOutput(input, target)
   -- print("input")
   -- print(input:size())
   -- print("target")
   -- print(target:size())
   -- log(input) * target + log(1 - input) * (1 - target)

   self.term1 = self.term1 or input.new()
   self.term2 = self.term2 or input.new()
   self.term3 = self.term3 or input.new()

   self.term1:resizeAs(input)
   self.term2:resizeAs(input)
   self.term3:resizeAs(input)

   self.term1:fill(1):add(-1,target)
   self.term2:fill(1):add(-1,input):add(eps):log():cmul(self.term1)

   self.term3:copy(input):add(eps):log():cmul(target)
   self.term3:add(self.term2)

   if self.sizeAverage then
      self.term3:div(target:nElement())
   end

   -- the error is Sum[(error at each point) * (importance of that point)]
   self:updateScalingMask(target)
   self.term3:cmul(self.mask)
   self.output = - self.term3:sum()

   return self.output
end

function MotionBCECriterion:updateGradInput(input, target)
   -- target / input - (1 - target) / (1 - input)

   self.term1 = self.term1 or input.new()
   self.term2 = self.term2 or input.new()
   self.term3 = self.term3 or input.new()

   self.term1:resizeAs(input)
   self.term2:resizeAs(input)
   self.term3:resizeAs(input)

   self.term1:fill(1):add(-1,target)
   self.term2:fill(1):add(-1,input)

   self.term2:add(eps)
   self.term1:cdiv(self.term2)

   self.term3:copy(input):add(eps)

   self.gradInput:resizeAs(input)
   self.gradInput:copy(target):cdiv(self.term3)

   self.gradInput:add(-1,self.term1)

   if self.sizeAverage then
      self.gradInput:div(target:nElement())
   end

   self.gradInput:mul(-1)

   -- as stated above,
   -- the error is Sum[(error at each point) * (importance of that point)]
   -- so the gradient is grad(error at each point) * (importance of that point)
   self:updateScalingMask(target)
   self.gradInput:cmul(self.mask)

   return self.gradInput
end

function MotionBCECriterion:updateScalingMask(target)
   self.mask:resizeAs(target):fill(0)
   local nBatches = target:size(1)

   -- find all the places in each frame that changed since the frame before
   -- all of the "forward in time" changes
   self.mask[{{2, nBatches}}] = target[{{2, nBatches}}] - target[{{1, nBatches - 1}}]
   self.mask:abs()

   -- also find all the "backward in time" changes
   -- these are the same places in the frames;
   -- we want to highlight the importance of regions that **will** change too
   self.mask[{{1, nBatches - 1}}] = self.mask[{{1, nBatches - 1}}]
                                    + self.mask[{{2, nBatches}}]--:clone()

   self.mask:abs():sign()
   -- normalize the tmask to be 1 where things changed, 0 otherwise
   -- self.mask:apply(function(el)
   --    if el > 0 then
   --       return 1
   --    else
   --       return 0
   --       end
   --    end)

   -- scale by the importance we assign to motion
   self.mask = self.mask * self.motionScale

   -- add 1 so we can just do self.mask * BCE
   self.mask = self.mask + 1
   return self.mask
end

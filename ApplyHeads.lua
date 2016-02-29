require 'nn'
require 'nngraph'

require 'Print'
require 'ChangeLimiter'
require 'Noise'
require 'ScheduledWeightSharpener'

local ApplyHeads = function(num_heads, dim_hidden, noise, sharpening_rate, scheduler_iteration)
    -- make the heads to analyze the encodings
    local enc1 = nn.Identity()()
    local enc2 = nn.Identity()()

    local heads = {}
    heads[1] = nn.Sequential()
    heads[1]:add(nn.JoinTable(2))
    heads[1]:add(nn.Linear(dim_hidden * 2, dim_hidden))
    heads[1]:add(nn.Sigmoid())
    heads[1]:add(nn.Noise(noise))
    heads[1]:add(nn.ScheduledWeightSharpener(sharpening_rate, scheduler_iteration))
    heads[1]:add(nn.AddConstant(1e-20))
    heads[1]:add(nn.Normalize(1, 1e-100))

    for i = 2, num_heads do
        heads[i] = heads[1]:clone()
    end

    for i = 1, num_heads do
        heads[i] = heads[i]{enc1, enc2}
    end

    local dist
    if num_heads > 1 then
        -- combine the distributions from all heads
        local dist_adder = nn.CAddTable()(heads)
        local dist_clamp = nn.Clamp(0, 1)(dist_adder)
        dist = dist_clamp
    else
        dist = heads[1]
    end

    -- and use it to filter the encodings
    local change_limiter = nn.ChangeLimiter()({dist, enc1, enc2}):annotate{name="change_limiter"}

    return nn.gModule({enc1,enc2},{change_limiter})
end

return ApplyHeads

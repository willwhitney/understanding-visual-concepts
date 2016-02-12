require 'hdf5'
require 'paths'
require 'math'
require 'xlua'

-- train-val-test: 70-15-15 split
function split_batches(examples, bsize)
    local num_test = math.floor(#examples * 0.15)
    local num_val = num_test
    local num_train = #examples - 2*num_test

    local test = {}
    local val = {}
    local train = {}

    -- shuffle examples
    local ridxs = torch.randperm(#examples)
    for i = 1, ridxs:size(1) do
        xlua.progress(i, ridxs:size(1))
        local batch = examples[ridxs[i]]
        if i <= num_train then
            table.insert(train, batch)
        elseif i <= num_train + num_val then
            table.insert(val, batch)
        else
            table.insert(test, batch)
        end
    end
    return {train, val, test}
end

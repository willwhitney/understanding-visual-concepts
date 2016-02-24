require 'hdf5'
require 'paths'
require 'math'
require 'xlua'

--[[
    Usage: Need a .h5 file <dataset_name> below, generated from
        action_data_converter.py, which has been saved in <dataset_folder>

    The difference between this and action_data_converter.lua is that this
    works with h5 files in which each h5 file is an action, rather than all
    the actions inside one h5 file
--]]


function load_data(dataset_name, dataset_folder, bsize)
    local dataset_file = hdf5.open(dataset_folder .. '/' .. dataset_name, 'r')
    local examples = {}
    for action,data in pairs(dataset_file:all()) do
        for j = 1,data:size(1)-bsize,bsize do  -- the frames in data re
            local batch = data[{{j,j+bsize-1}}]
            table.insert(examples, batch)  -- each batch is a contiguous sequence (bsize, height, width)
        end
    end
    return examples
end

function split_batches(examples)
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

function save_batches(datasets, savefolder, idxs)
    local train, val, test = unpack(datasets)
    local data_table = {train=train, val=val, test=test}
    for dname,data in pairs(data_table) do
        local subfolder = paths.concat(savefolder,dname)
        if not paths.dirp(subfolder) then paths.mkdir(subfolder) end
        for _,b in pairs(data) do
            -- xlua.progress(idxs[dname], #data)
            local batchname = paths.concat(subfolder, 'batch'..idxs[dname])
            print(dname..': '..batchname)
            torch.save(batchname, b:float())
            idxs[dname] = idxs[dname] + 1
        end
    end
    return idxs
end

function main_all()
    local scenario = 'd4'
    local actions = {'running', 'jogging', 'walking', 'handclapping', 'handwaving', 'boxing'}
    local dataset_folder = '/om/data/public/mbchang/udcign-data/action/raw/hdf5'
    local to_save_folder = '/om/data/public/mbchang/udcign-data/action/allactions'..scenario
    if not paths.dirp(to_save_folder) then paths.mkdir(to_save_folder) end
    local bsize = 30
    local idxs = {train=1,val=1,test=1}  -- train, val, test
    print(idxs)

    for _,action in pairs(actions) do
        local dataset_name = action .. '_subsamp=1_scenario=d4.h5'
        print('dataset:'..dataset_name)
        print('to save:'..to_save_folder)

        -- main
        print('loading data')
        local ex = load_data(dataset_name, dataset_folder, bsize)
        print('splitting batches')
        local train, val, test = unpack(split_batches(ex))
        print('saving batches')
        idxs = save_batches({train, val, test}, to_save_folder, idxs)
        print(idxs)
    end
end


main_all()

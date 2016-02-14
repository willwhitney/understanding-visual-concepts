require 'paths'
require 'image'
require 'data_utils'
require 'utils'

local raw_root = '/om/data/public/mbchang/udcign-data/kitti/raw/videos'
local out_root = '/om/data/public/mbchang/udcign-data/kitti/data/all'
local bsize = 30
local ch = 'image_02'
local data_root = '/om/data/public/mbchang/udcign-data/kitti/data_bsize'..bsize
local dim = 150
local id = 0
local subsample = 1

-- create all the batches for one setting
-- a setting can be 'road', 'campus', etc
function create_batches(setting_folder, bsize, idxs)
    print(idxs)
    local setting_ex = {}
    for group in paths.iterdirs(setting_folder) do
        local img_folder = paths.concat(setting_folder,group,ch,'data_resize')
        local ex = get_examples(img_folder)
        local groups2 = duplicate(ex)  -- now in groups of 2
        assert(#groups2 == #ex - 1)
        setting_ex = extend(setting_ex,groups2)
        print(#groups2,#setting_ex)
    end
    print(#setting_ex)
    -- setting_ex is now a huge table of 2-tables for this particular setting
    local setting_batches = group2batches(setting_ex,bsize)
    -- tables of examples of (1,150,150)
    print('split batches: '..#setting_batches)
    local train, val, test = unpack(split_batches(setting_batches, bsize))

    -- now save
    print('save batches: train '..#train..' val '..#val..' test '..#test)
    local new_idxs = save_batches({train, val, test}, out_root, idxs)

    return new_idxs
end

function group2batches(examples, bsize)
    local batches = {}
    for j = 1, #examples-bsize, bsize do
        local batch = subrange(examples, j, j+bsize-1)
        table.insert(batches, batch)  -- each batch is table of 2-tables
    end
    return batches
end


function get_examples(folder)
    -- first get the imgs inside this folder
    print(folder)
    local imgs = {}
    for img in paths.iterfiles(folder) do
        imgpath = paths.concat(folder, img)
        imgs[#imgs+1] = imgpath
    end
    table.sort(imgs)  -- consecutive

    -- then subsample these images
    local subsampled_imgs = {}
    for k,img_file in pairs(imgs) do
        if k % subsample == 0 then
            local img = image.load(img_file)
            subsampled_imgs[#subsampled_imgs+1] = img:float()
        end
    end
    return subsampled_imgs
end

-- turns n examples into n-1 example pairs
-- return a table of size n-1 of tables of size 2
function duplicate(examples)
    local tm1s = subrange(examples,1,#examples-1)
    local ts = subrange(examples,2,#examples)
    local g2 = {}
    for i = 1, #tm1s do
        g2[#g2+1] = {tm1s[i],ts[i]}  -- t is second element
    end
    return g2
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
            -- torch.save(batchname, b)
            idxs[dname] = idxs[dname] + 1
        end
    end
    return idxs
end


-- main
function main()
    local idxs = {train=1,val=1,test=1}  -- train, val, test
    for setting in paths.iterdirs(raw_root) do
        print('setting '..setting)
        idxs = create_batches(paths.concat(raw_root,setting), bsize, idxs)

        --
        -- for group in paths.iterdirs(paths.concat(raw_root,setting)) do
        --     print(idxs)
        --     local img_folder = paths.concat(raw_root,setting,group,ch,'data_resize')
        --     idxs = create_batches(img_folder, bsize, idxs)  -- create train, val, test for this setting
        -- end

    end
    print(idxs)
end

main()


-- to use: first call resize.py, then call this file

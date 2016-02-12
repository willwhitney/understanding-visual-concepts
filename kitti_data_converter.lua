require 'paths'
require 'image'
require 'data_utils'

local raw_root = '/om/data/public/mbchang/udcign-data/kitti/raw/videos'
local out_root = '/om/data/public/mbchang/udcign-data/kitti/data/all'
local bsize = 30
local ch = 'image_00'
local data_root = '/om/data/public/mbchang/udcign-data/kitti/data_bsize'..bsize
local dim = 150
local id = 0
local subsample = 5

function create_batches(folder, bsize, idxs)
    print(folder)
    local imgs = {}
    for img in paths.iterfiles(folder) do  -- is this sorted?
        imgpath = paths.concat(folder, img)
        imgs[#imgs+1] = imgpath
    end
    table.sort(imgs)  -- consecutive

    local subsampled_imgs = {}
    for k,img_file in pairs(imgs) do
        if k % subsample == 0 then
            local img = image.load(img_file)
            subsampled_imgs[#subsampled_imgs+1] = img
        end
    end

    -- TODO you should split on the level of the setting

    -- tables of examples of (1,150,150)
    print('split batches: '..#subsampled_imgs)
    local train, val, test = unpack(split_batches(subsampled_imgs, bsize))

    -- now save
    print('save batches: train '..#train..' val '..#val..' test '..#test)
    local new_idxs = save_batches({train, val, test}, out_root, idxs)

    return new_idxs
end


function save_batches(datasets, savefolder, idxs)
    local train, val, test = unpack(datasets)
    local data_table = {train=train, val=val, test=test}
    for dname,data in pairs(data_table) do
        local subfolder = paths.concat(savefolder,dname)
        if not paths.dirp(subfolder) then paths.mkdir(subfolder) end
        for _,b in pairs(data) do
            xlua.progress(idxs[dname], #data)
            b = b:float()
            local batchname = paths.concat(subfolder, 'batch'..idxs[dname])
            -- print(dname..': '..batchname)
            torch.save(batchname, b)
            idxs[dname] = idxs[dname] + 1
        end
    end
    return idxs
end


-- main
function main()
    local idxs = {train=1,val=1,test=1}  -- train, val, test
    for setting in paths.iterdirs(raw_root) do
        print('setting')
        for group in paths.iterdirs(paths.concat(raw_root,setting)) do
            print(idxs)
            local img_folder = paths.concat(raw_root,setting,group,ch,'data_resize')
            idxs = create_batches(img_folder, bsize, idxs)  -- create train, val, test for this setting
        end
        -- assert(false)
    end
end

main()

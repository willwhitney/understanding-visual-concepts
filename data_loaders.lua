local data_loaders = {}

function data_loaders.load_mv_batch(id, dataset_name, mode)
    local data = torch.load(opt.datasetdir .. '/th_' .. dataset_name .. '/' .. mode .. '/batch' .. id)

    local input1s = torch.zeros(19, 1, 150, 150)
    local input2s = torch.zeros(19, 1, 150, 150)

    if opt.gpu then
        data = data:cuda()
    	input1s = input1s:cuda()
    	input2s = input2s:cuda()
    end

    for i = 1, 19 do
        input1s[i] = data[i]
        input2s[i] = data[i + 1]
    end
    return {input1s, input2s}
end

function data_loaders.load_random_mv_batch(mode)
    local variation_type = math.random(3)
    local variation_name = ""
    if variation_type == 1 then
        variation_name = "AZ_VARIED"
    elseif variation_type == 2 then
        variation_name = "EL_VARIED"
    elseif variation_type == 3 then
        variation_name = "LIGHT_AZ_VARIED"
    end

    local id, mode_name
    if mode == 'train' then
        mode_name = 'FT_training'
        id = math.random(opt.num_train_batches_per_type)
    elseif mode == 'test' then
    	mode_name = 'FT_test'
        id = math.random(opt.num_test_batches_per_type)
    end
    return data_loaders.load_mv_batch(id, variation_name, mode_name), variation_type
end



function data_loaders.load_atari_batch(id, mode)
    local data = torch.load(opt.datasetdir .. '/' .. opt.dataset_name .. '/' .. mode .. '/images_batch_' .. id)

    local input1s = torch.zeros(29, 3, 210, 160)
    local input2s = torch.zeros(29, 3, 210, 160)

    if opt.gpu then
        data = data:cuda()
    	input1s = input1s:cuda()
    	input2s = input2s:cuda()
    end

    for i = 1, 29 do
        input1s[i] = data[i]
        input2s[i] = data[i + 1]
    end
    return {input1s, input2s}
end

function data_loaders.load_random_atari_batch(mode)
    local id
    if mode == 'train' then
        id = math.random(opt.num_train_batches)
    elseif mode == 'test' then
        id = math.random(opt.num_train_batches)
    end
    return data_loaders.load_atari_batch(id, mode)
end



function data_loaders.load_action_batch(id, mode)
    local data = torch.load(opt.datasetdir .. '/' .. opt.dataset_name .. '/' .. mode .. '/images_batch_' .. id)
    data = data:reshape(data:size(1),1,data:size(2),data:size(3))  -- one channel

    local input1s = torch.zeros(29, 1, 120, 160)
    local input2s = torch.zeros(29, 1, 120, 160)

    if opt.gpu then
        data = data:cuda()
    	input1s = input1s:cuda()
    	input2s = input2s:cuda()
    end

    for i = 1, 29 do
        input1s[i] = data[i]
        input2s[i] = data[i + 1]
    end
    return {input1s, input2s}
end

function data_loaders.load_random_action_batch(mode)
    local id
    if mode == 'train' then
        id = math.random(opt.num_train_batches)
    elseif mode == 'test' then
        id = math.random(opt.num_train_batches)
    end
    return data_loaders.load_action_batch(id, mode)
end

return data_loaders

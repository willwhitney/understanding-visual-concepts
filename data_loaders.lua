function load_mv_batch(id, dataset_name, mode)
    local data = torch.load(opt.datasetdir .. '/th_' .. dataset_name .. '/' .. mode .. '/batch' .. id):cuda()

    input1s = torch.zeros(19, 1, 150, 150)
    input2s = torch.zeros(19, 1, 150, 150)
    if opt.gpuid then
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

function load_random_mv_batch(mode)
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
    return load_mv_batch(id, variation_name, mode_name), variation_type
end

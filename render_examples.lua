require 'nn'
require 'cutorch'
require 'cunn'
require 'paths'
require 'lfs'

require 'vis'
require 'UnsupervisedEncoder'
require 'Decoder'
require 'data_loaders'

opt = {
        datasetdir = '/om/user/wwhitney/facegen/CNN_DATASET',
        gpu = true,
    }

-- networks whose names contain this string will be rendered
search_str = "unsup"

if true then
    base_directory = "/om/user/wwhitney/unsupervised-dcign/networks"
else
    base_directory = lfs.currentdir()
end

local jobname = search_str ..'_'.. os.date("%b_%d_%H_%M")
local output_path = 'reports/renderings/'..jobname
os.execute('mkdir -p '..output_path)

local dataset_types = {"AZ_VARIED", "EL_VARIED", "LIGHT_AZ_VARIED"}

function getMatchingNetworkNames(search_string)
    local results = {}
    for network_name in lfs.dir(base_directory) do
        local network_path = base_directory .. '/' .. network_name
        if lfs.attributes(network_path).mode == 'directory' then
            if string.find(network_name, search_str) then
                table.insert(results, network_name)
            end
        end
    end
    return results
end

function getLastSnapshot(network_name)
    local res_file = io.popen("ls -t "..paths.concat(base_directory, network_name).." | grep -i epoch | head -n 1")
    local result = res_file:read():match( "^%s*(.-)%s*$" )
    res_file:close()
    return result
end

for _, network in ipairs(getMatchingNetworkNames(search_string)) do
    print('')
    print(network)
    local checkpoint = torch.load(paths.concat(base_directory, network, getLastSnapshot(network)))
    local model = checkpoint.model
    iteration = checkpoint.step
    model:evaluate()

    local encoder = model.modules[1]
    local change_limiter = encoder:listModules()[32]
    for _, variation in ipairs{"AZ_VARIED", "EL_VARIED", "LIGHT_AZ_VARIED"} do
        local images = {}
        for i = 1, 1 do -- for now only render one batch
            -- fetch a batch
            local input = load_mv_batch(i, variation, 'FT_test')
            output = model:forward(input)

            local weight_norms = torch.zeros(output:size(1))
            for input_index = 1, output:size(1) do
                local weights = change_limiter.output[input_index]
                local max_weight, varying_index = weights:max(1)
                print("Varying index: " .. vis.simplestr(varying_index), "Weight: " .. vis.simplestr(max_weight))
                weight_norms[input_index] = weights:norm()

                local image_row = {}
                table.insert(image_row, input[1][input_index]:float())
                table.insert(image_row, input[2][input_index]:float())
                table.insert(image_row, output[input_index]:float())
                table.insert(images, image_row)
            end
            print("Mean independence of weights: ", weight_norms:mean())

            collectgarbage()
        end
        vis.save_image_grid(paths.concat(output_path, network .."-"..variation..'.png'), images)
    end
end


print("done")

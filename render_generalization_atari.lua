require 'nn'
require 'cutorch'
require 'cunn'
require 'paths'
require 'lfs'

vis = require 'vis'
require 'AtariEncoder'
require 'AtariDecoder'
local data_loaders = require 'data_loaders'

name = arg[1]
-- dataset_name = arg[2] or name
networks = {}
while true do
    local line = io.read()
    if line == nil then break end

    -- strip whitespace
    line = string.gsub(line, "%s+", "")

    table.insert(networks, line)
end

-- opt = {
--         datasetdir = '/om/user/wwhitney/deep-game-engine',
--         dataset_name = dataset_name,
--         gpu = true,
--     }

base_directory = "/om/user/wwhitney/unsupervised-dcign/networks"

local jobname = name ..'_'.. os.date("%b_%d_%H_%M")
local output_path = 'reports/renderings/mutation/'..jobname
os.execute('mkdir -p '..output_path)


function getLastSnapshot(network_name)
    local res_file = io.popen("ls -t "..paths.concat(base_directory, network_name).." | grep -i epoch | head -n 1")
    local status, result = pcall(function() return res_file:read():match( "^%s*(.-)%s*$" ) end)
    -- print(status, result)
    res_file:close()
    if not status then
        return false
    else
        return result
    end
end

for _, network in ipairs(networks) do
    collectgarbage()

    print('')
    print(network)
    local snapshot_name = getLastSnapshot(network)
    if snapshot_name then
        local checkpoint = torch.load(paths.concat(base_directory, network, snapshot_name))
        opt = checkpoint.opt
        local model = checkpoint.model
        local scheduler_iteration = torch.Tensor{checkpoint.step}
        model:evaluate()

        local encoder = model.modules[1]
        local sharpener = encoder:findModules('nn.ScheduledWeightSharpener')[1]
        sharpener.iteration_container = scheduler_iteration
        print("Current sharpening: ", sharpener:getP())

        local weight_predictor = encoder:findModules('nn.Normalize')[1]
        local previous_embedding = encoder:findModules('nn.Linear')[1]
        -- local current_embedding = encoder:findModules('nn.Linear')[2]
        local decoder = model.modules[2]

        for batch_index = 342, 342 do
            -- local images = {}

            -- fetch a batch
            local input = data_loaders.load_atari_batch(batch_index, 'train')
            local output = model:forward(input):clone()
            local embedding_from_previous = previous_embedding.output:clone()
            -- local embedding_from_current = current_embedding.output:clone()

            -- local reconstruction_from_previous = decoder:forward(embedding_from_previous):clone()
            -- local reconstruction_from_current = decoder:forward(embedding_from_current):clone()

            local weight_norms = torch.zeros(output:size(1))
            for input_index = 1, output:size(1) do
                local weights = weight_predictor.output[input_index]:clone()
                weight_norms[input_index] = weights:norm()
            end
            print("Mean independence of weights: ", weight_norms:mean())

            local max_indices = {}
            for input_index = 1, output:size(1) do
                local weights = weight_predictor.output[input_index]:clone()
                local _, idx = weights:max(1)
                max_indices[idx] = true
            end

            for input_index = 2, 2 do
                local base_embedding = embedding_from_previous[batch_index]:clone()
                for _, max_index in ipairs(max_indices) do
                    -- local weights = weight_predictor.output[input_index]:clone()
                    -- local max_weight, varying_index = weights:max(1)

                    local i = 0
                    for change = -4, 1.5, 0.05 do
                        local output_directory = paths.concat(
                            output_path,
                            network,
                            'batch_'..batch_index..'_input_'..input_index..'_along_'..max_index)
                        local output_filename = paths.concat(
                            output_directory,
                            'changing_'..i..'_amount_'..change..'.png')
                        os.execute('mkdir -p '..output_directory)

                        local changed_embedding = base_embedding:clone()
                        changed_embedding[max_index] = changed_embedding[max_index] + change

                        local rendering = decoder:forward(changed_embedding:reshape(1, 200))[1]:clone()
                        image.save(output_filename, rendering:float())
                        i = i + 1
                    end

                    -- weight_norms[input_index] = weights:norm()

                    -- local image_row = {}
                    -- table.insert(image_row, input[1][input_index]:float())
                    -- table.insert(image_row, input[2][input_index]:float())
                    -- table.insert(image_row, reconstruction_from_previous[input_index]:float())
                    -- table.insert(image_row, reconstruction_from_current[input_index]:float())
                    -- table.insert(image_row, output[input_index]:float())
                    -- table.insert(images, image_row)
                end
            end
            -- print("Mean independence of weights: ", weight_norms:mean())
            -- vis.save_image_grid(paths.concat(output_path, network .. '_batch_'..batch_index..'.png'), images)

            collectgarbage()
        end
    end
end


print("done")

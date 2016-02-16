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
local output_path = 'reports/renderings/'..jobname
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
        local current_embedding = encoder:findModules('nn.Linear')[2]
        local decoder = model.modules[2]

        for i = 339, 343 do
            local images = {}

            -- fetch a batch
            local input = data_loaders.load_atari_batch(i, 'train')
            local output = model:forward(input):clone()
            local embedding_from_previous = previous_embedding.output:clone()
            local embedding_from_current = current_embedding.output:clone()

            local reconstruction_from_previous = decoder:forward(embedding_from_previous):clone()
            local reconstruction_from_current = decoder:forward(embedding_from_current):clone()

            local weight_norms = torch.zeros(output:size(1))
            for input_index = 1, math.min(30, output:size(1)), 3 do
                local weights = weight_predictor.output[input_index]:clone()
                local max_weight, varying_index = weights:max(1)
                -- print("Varying index: " .. vis.simplestr(varying_index), "Weight: " .. vis.simplestr(max_weight))

                -- local embedding_change = embedding_from_current[input_index] - embedding_from_previous[input_index]
                -- local normalized_embedding_change = embedding_change / embedding_change:norm(1)
                -- print("Independence of embedding change: ", normalized_embedding_change:norm())
                -- print("Distance between timesteps: ", embedding_change:norm())

                weight_norms[input_index] = weights:norm()

                local image_row = {}
                table.insert(image_row, input[1][input_index]:float())
                table.insert(image_row, input[2][input_index]:float())
                table.insert(image_row, reconstruction_from_previous[input_index]:float())
                table.insert(image_row, reconstruction_from_current[input_index]:float())
                table.insert(image_row, output[input_index]:float())
                table.insert(images, image_row)
            end
            print("Mean independence of weights: ", weight_norms:mean())
            vis.save_image_grid(paths.concat(output_path, network .. '_batch_'..i..'.png'), images)

            collectgarbage()
        end
    end
end


print("done")

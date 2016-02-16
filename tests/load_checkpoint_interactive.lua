require 'nn'
require 'cutorch'
require 'cunn'
require 'image'

vis = require 'vis'
data_loaders = require 'data_loaders'
Encoder = require 'AtariEncoder'
Decoder = require 'AtariDecoder'

base_directory = "/om/user/wwhitney/unsupervised-dcign/networks"

function getLastSnapshot(network_name)
    local res_file = io.popen("ls -t "..paths.concat(base_directory, network_name).." | grep -i epoch | head -n 1")
    local result = res_file:read():match( "^%s*(.-)%s*$" )
    res_file:close()
    return result
end


network_name = "atari_motion_scale_3_noise_0.1_heads_1_sharpening_rate_10_gpu_learning_rate_0.0002_dataset_name_space_invaders_frame_interval_10"
epoch = getLastSnapshot(network_name)
checkpoint = torch.load('networks/'..network_name..'/'..epoch)
opt = checkpoint.opt

model = checkpoint.model
encoder = model.modules[1]
decoder = model.modules[2]

model:evaluate()

weight_predictor = encoder:findModules('nn.Normalize')[1]
previous_embedding = encoder:findModules('nn.Linear')[1]
current_embedding = encoder:findModules('nn.Linear')[2]

for i, mod in ipairs(encoder:listModules()) do print(i, mod) end

batch = data_loaders.load_atari_batch(339, 'test')
output = model:forward(batch):clone()


test_input_frame_index = 14

weights = weight_predictor.output[test_input_frame_index]
mx, idx = weights:max(1)
mx = mx[1]
idx = idx[1]

print("mx: ", mx, "idx: ", idx)

for i = 1, weights:size(1) do
    print(i, vis.simplestr(weights[i]))
end

base_embedding = previous_embedding.output[test_input_frame_index]:clone()

function render_changing_index(changing_index)
    output_dir = 'reports/renderings/mutate_'..network_name
    os.execute('mkdir -p '..output_dir)
    i = 0
    for change = -4, 1.5, 0.05 do
        changed_embedding = base_embedding:clone()
        changed_embedding[changing_index] = changed_embedding[changing_index] + change
        image.save(output_dir..'/changing_'..i..'_amount_'..change..'.png', decoder:forward(changed_embedding:reshape(1, 200))[1]:float():clone())
        i = i + 1
    end
end

-- render_changing_index(idx)

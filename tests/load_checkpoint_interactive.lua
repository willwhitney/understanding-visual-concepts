require 'nn'
require 'cutorch'
require 'cunn'
require 'image'

data_loaders = require 'data_loaders'
Encoder = require 'AtariEncoder'
Decoder = require 'AtariDecoder'

network_name = "breakout_noise_0.1_heads_3_sharpening_rate_5_learning_rate_0.0001_gpu"
epoch = 'epoch5.62_0.2013.t7'
checkpoint = torch.load('networks/'..network_name..'/'..epoch)
opt = checkpoint.opt

model = checkpoint.model
encoder = model.modules[1]
decoder = model.modules[2]

weight_predictor = encoder:findModules('nn.Normalize')[1]
previous_embedding = encoder:findModules('nn.Linear')[1]
current_embedding = encoder:findModules('nn.Linear')[2]

for i, mod in ipairs(encoder:listModules()) do print(i, mod) end

batch = data_loaders.load_atari_batch(234, 'test')
output = model:forward(batch):clone()
mx, idx = weight_predictor.output[1]:max(1)



base_embedding = previous_embedding.output[1]:clone()

function render_changing_max_index()
    output_dir = 'reports/renderings/mutate_'..network_name
    os.execute('mkdir -p '..output_dir)
    for change = -10, 10, 0.2 do
        changed_embedding = base_embedding:clone()
        changed_embedding[idx[1]] = changed_embedding[idx[1]] + change
        image.save(output_dir..'/changing_'..change..'.png', decoder:forward(changed_embedding:reshape(1, 200))[1]:float():clone())
    end
end


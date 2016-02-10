require 'nn'
require 'cutorch'
require 'cunn'
Encoder = require 'UnsupervisedEncoder'
Decoder = require 'Decoder'
checkpoint = torch.load('networks/unsup_gpu_learning_rate_0.0001_noise_0.05_criterion_BCE_sharpening_rate_10/epoch7.00_0.3695.t7')
encoder = checkpoint.model.modules[1]
for i, mod in ipairs(encoder:listModules()) do print(i, mod) end

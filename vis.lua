require 'image'

local vis = {}

function vis.save_image_grid(filepath, images)
    if images ~= nil and images[1] ~= nil then
        image_width = 150
        padding = 5
        images_across = #images[1]
        images_down = #images
        -- print(images_down, images_across)

        image_output = torch.zeros(
            image_width * images_down + (images_down - 1) * padding,
            image_width * images_across + (images_across - 1) * padding)
        for i, image_row in ipairs(images) do
            for j, image in ipairs(image_row) do
                y_index = j - 1
                y_location = y_index * image_width + y_index * padding
                x_index = i - 1
                x_location = (x_index) * image_width + x_index * padding
                -- print({{x_location + 1, x_location + image_width},
                --           {y_location + 1, y_location + image_width}})
                image_output[{{x_location + 1, x_location + image_width},
                {y_location + 1, y_location + image_width}}] = image
            end
        end
        image_output = image_output:reshape(1, image_output:size()[1], image_output:size()[2])
        image.save(filepath, image_output)
    else
        error("Invalid images:", images)
    end
end

vis.colors = {
    HEADER = '\27[95m',
    OKBLUE = '\27[94m',
    OKGREEN = '\27[92m',
    WARNING = '\27[93m',
    FAIL = '\27[91m',
    ENDC = '\27[0m',
    BOLD = '\27[1m',
    UNDERLINE = '\27[4m',
    RESET = '\27[0m'
}

vis.decimalPlaces = 4

function vis.lines(str)
    local t = {}
    local function helper(line) table.insert(t, line) return "" end
    helper((str:gsub("(.-)\r?\n", helper)))
    return t
end

function vis.flatten(tensor)
    return tensor:reshape(1, tensor:nElement())
end

function vis.round(tensor, places)
    places = places or vis.decimalPlaces
    local tensorClone = tensor:clone()
    local offset = 0
    if tensor:sum() ~= 0 then
        offset = - math.floor(math.log10(torch.abs(tensorClone):mean())) + (places - 1)
    end

    tensorClone = tensorClone * (10 ^ offset)
    tensorClone:round()
    tensorClone = tensorClone / (10 ^ offset)

    if tostring(tensorClone[1]) == tostring(0/0) then
        print(tensor)
        print(math.floor(math.log10(torch.abs(tensorClone):mean())))
        print(offset)
        error("got nan")
    end

    return tensorClone
end

function vis.simplestr(tensor)
    -- local rounded = vis.round(tensor)
    -- -- local rounded = tensor:clone()
    --
    -- local strTable = vis.lines(tostring(vis.flatten(rounded)))
    -- table.remove(strTable, #strTable)
    -- table.remove(strTable, #strTable)
    --
    -- local str = ""
    -- for i, line in ipairs(strTable) do
    --     str = str..line
    -- end
    -- return str
    -- print("tensor", tensor)
    -- print("tensor1", tensor[1])
    -- print(tensor)
    local str = string.format("%." .. vis.decimalPlaces .. "f", tensor[1])
    for i = 2, tensor:size(1) do
        str = str .. string.format(" %." .. vis.decimalPlaces .. "f", tensor[i])
    end
    return str
end

function vis.prettySingleError(number)
    local str = tostring(number)
    if math.abs(number) < 1e-10 then
        return '0.0000'
    else
        return vis.colors.FAIL..str..vis.colors.RESET
    end
end

function vis.prettyError(err)
    if type(err) == 'number' then
        return vis.prettySingleError(err)
    elseif type(err) == 'table' then
        local str = ''
        for _, val in ipairs(err) do
            str = str .. ' ' .. vis.prettySingleError(val)
        end
        return str
    elseif err.size then -- assume tensor
        local rounded = vis.round(err)
        if rounded:nDimension() ~= 1 then
            error("Only able to pretty-print 1D tensors.")
        else
            local str = ''
            for i = 1, rounded:size(1) do
                str = str .. ' ' .. vis.prettySingleError(rounded[i])
            end
            return str
        end
    else
        error("Not sure what to do with this object.")
    end
end

function vis.diff(a, b)
    local str
    if type(a) == 'number' and type(b) == 'number' then
        str = vis.prettySingleError(a - b)
    elseif type(a) == 'table' and type(b) == 'table' then
        str = ''
        for i, _ in ipairs(a) do
            str = str .. ' ' .. vis.prettySingleError(a[i] - b[i])
        end
    elseif a.size and b.size then -- assume tensor
        local rounded = vis.round(a - b)
        if rounded:nDimension() ~= 1 then
            error("Only able to pretty-print 1D tensors.")
        else
            str = ''
            for i = 1, rounded:size(1) do
                str = str .. ' ' .. vis.prettySingleError(rounded[i])
            end
        end
    else
        error("Not sure what to do with this object.")
    end
    print(str)
end

function vis.hist(a)
    tensor = a:clone()
    tensor = tensor / tensor:clone():abs():max()
    -- print(tensor:min())
    tensor = tensor + (-tensor:min())
    tensor:mul(10)
    local str = vis.simplestr(tensor)
    -- print(str)
    os.execute('spark ' .. str)
end

return vis

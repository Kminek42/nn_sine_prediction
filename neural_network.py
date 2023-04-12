import random

def extract_data(_filename):
    inputs = []
    outputs = []
    input_n = 0

    file = open(_filename, "r")
    data = file.read()
    file.close()

    data = data.split("\n")
    input_n = len(data[0].split()) - 1
    
    for line in data:
        if len(line.split()) != input_n + 1:
            break

        temp = []
        for i in range(input_n):
            temp.append(float(line.split()[i]))
        inputs.append(temp)
        outputs.append(float(line.split()[input_n]))

    return [inputs, outputs, input_n]

def calculate_output(_input, _weights):
    output = 0
    for i in range(len(_input)):
        output += _input[i] * _weights[i]
    
    return output

def training(_input, _weights, _output):
    output_predicted = calculate_output(_input, _weights)
    error = _output - output_predicted
    gradient = []
    for i in range(len(_input)):
       gradient.append(error * _input[i])
    
    return [gradient, error ** 2]

def init_nn(_t):
    _weights = []
    for i in range(_t[2]):
        _weights.append(random.normalvariate())
    return _weights
    
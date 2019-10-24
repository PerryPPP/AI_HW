import torch
import numpy as np
import torchvision.models as models
from matplotlib import pyplot
from ResNet import ResNet18

if __name__ == '__main__':
    model = ResNet18()
    checkpoint = torch.load('./model/net_200.pth')
    model.load_state_dict(checkpoint)
    params = {}
    #print("Model's state_dict:")
    # Print model's state_dict
    for param_tensor in model.state_dict():
        parameters = model.state_dict()[param_tensor]
        #print(param_tensor,"\t", parameters.size())
        params[param_tensor] = parameters.detach().numpy()
    filters = params['conv1.0.weight']
    biases = params['conv1.1.bias']
    #normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    for i in range(3):
        ix = 1
        for j in range(64):
            ax = pyplot.subplot(8, 8, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            pyplot.imshow(filters[j][i], cmap=pyplot.get_cmap('PuRd'))
            ix += 1
        # show the figure
        pyplot.show()
        pyplot.savefig("./pic/channel"+ str(i+1)+".png")
        ix = 1

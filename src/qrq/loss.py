from torch import nn
import torch


class IntentLoss(nn.Module):
    def __init__(self, input=None, target=None, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='elementwise_mean'):
        super(IntentLoss, self).__init__()
        self.__dict__.update(locals())
        self.loss = nn.CrossEntropyLoss()
    def forward(self, input, target):
          ''' 
          Input ((N x n_classes x 2) torch.Tensor): Predicated class\intent, 
          Target ((N x n_classes) torch.long torch.Tensor): Real Intent from dataset. 
          input is just the last layer of the network
          target is just one hot encoding saying if that particular class exists or not.
          '''
          #  Given just five classes/intent
          # sample input = [0.9, 0.8, 0.7, 0.1, 0.2, 0.5] 
          # sample target = [1, 0, 0, 1, 0, 0]
          # print(type(input), type(target))
          # print(input.shape, target.shape)
          loss = self.loss(input.view(-1, 2), target.view(-1))
          return loss

class InformationGainLoss(nn.Module):
    def __init__(self, input=None, target=None, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='elementwise_mean'):
        super(InformationGainLoss, self).__init__()
        self.__dict__.update(locals())

    def forward(self, intent_1, intent_2):
        '''
        Args:
            intent_1 ((N x n_classes x 2) torch.Tensor): Predicted class\intent at stage 1
            intent_2 ((N x n_classes x 2) torch.long torch.Tensor): Predicted class\intent at stage 2
        '''
        # Both of them are coming from NN models
        information_gain = torch.sum(intent_2.view(-1, 2)[:, 1] - intent_1.view(-1, 2)[:, 1])
        return -information_gain
   
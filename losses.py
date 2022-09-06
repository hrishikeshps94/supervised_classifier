from math import ceil, floor
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss



class FocalLoss(_Loss):
    def __init__(self,
        weight: Optional[Tensor] = None,
        ignore_index: int = -100,
        reduction: str = 'sum',
        gamma: float = 2.0) -> Tensor:
        """Implements the focal loss from
        `"Focal Loss for Dense Object Detection" <https://arxiv.org/pdf/1708.02002.pdf>`_
        Args:
            weight (torch.Tensor[K], optional): manual rescaling of each class
            ignore_index (int, optional): specifies target value that is ignored and do not contribute to gradient
            reduction (str, optional): reduction method
            gamma (float, optional): gamma parameter of focal loss
        Returns:
            torch.Tensor: loss reduced with `reduction` method
        """
        super().__init__()
        self.reduction = reduction
        self.weight = weight
        self.ignore_index = ignore_index
        self.gamma = gamma
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: (torch.Tensor[N, K, ...]), where K is the number of classes.
                You can pass logits or probabilities as input, if pass logit, must set softmax=True
            target: 
                (torch.Tensor[N, ...]): hard target tensor, where K is the number of classes, 
                It should contain binary values
       """
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt*target

        # Compute pt and logpt only for target classes (the remaining will have a 0 coefficient)
        # Ignore index (set loss contribution to 0)
        valid_idxs = torch.ones(target.view(-1).shape[0], dtype=torch.bool, device=input.device)
        if self.ignore_index >= 0 and self.ignore_index < input.shape[1]:
            valid_idxs[target.view(-1) == self.ignore_index] = False


        # Get P(class)
        pt = logpt.exp()

        # Weight
        if self.weight is not None:
            # Tensor type
            if self.weight.type() != input.data.type():
                self.weight = self.weight.type_as(input.data)
            logpt = logpt*self.weight

        # Loss
        loss = (-1 * (1 - pt) ** self.gamma * logpt).reshape(-1)

        # Loss reduction
        if self.reduction == 'sum':
            loss = loss[valid_idxs].sum()
        elif self.reduction == 'mean':
            loss = loss[valid_idxs].mean()
        else:
            # if no reduction, reshape tensor like target
            loss = loss.view(*target.shape)

        return loss


class PolyLoss(_Loss):
    def __init__(self,
        weight: Optional[Tensor] = None,
        ignore_index: int = -100,
        reduction: str = 'mean',
        eps: float = 2.0) -> Tensor:
        """Implements the Poly1 loss from `"PolyLoss: A Polynomial Expansion Perspective of Classification Loss
        Functions" <https://arxiv.org/pdf/2204.12511.pdf>`_.
        Args:
            x (torch.Tensor[N, K, ...]): predicted probability
            target (torch.Tensor[N, K, ...]): target probability
            eps (float, optional): epsilon 1 from the paper
            weight (torch.Tensor[K], optional): manual rescaling of each class
            ignore_index (int, optional): specifies target value that is ignored and do not contribute to gradient
            reduction (str, optional): reduction method
        Returns:
            torch.Tensor: loss reduced with `reduction` method
        """
        super().__init__()
        self.reduction = reduction
        self.weight = weight
        self.ignore_index = ignore_index
        self.eps = eps
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: (torch.Tensor[N, K, ...]), where K is the number of classes.
                You can pass logits or probabilities as input, if pass logit, must set softmax=True
            target: 
                (torch.Tensor[N, ...]): hard target tensor, where K is the number of classes, 
                It should contain binary values
       """
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt*target

        # Compute pt and logpt only for target classes (the remaining will have a 0 coefficient)
        # Ignore index (set loss contribution to 0)
        valid_idxs = torch.ones(target.view(-1).shape[0], dtype=torch.bool, device=input.device)
        if self.ignore_index >= 0 and self.ignore_index < input.shape[1]:
            valid_idxs[target.view(-1) == self.ignore_index] = False

        # Weight
        if self.weight is not None:
            # Tensor type
            if self.weight.type() != input.data.type():
                self.weight = self.weight.type_as(input.data)
            logpt = logpt*self.weight

        # Loss
        loss = (-1 * logpt + self.eps * (1 - logpt.exp())).reshape(-1)

        # Loss reduction
        if self.reduction == 'sum':
            loss = loss[valid_idxs].sum()
        elif self.reduction == 'mean':
            loss = loss[valid_idxs].mean()
        else:
            # if no reduction, reshape tensor like target
            loss = loss.view(*target.shape)

        return loss


class FocalLossOrg(_Loss):
    def __init__(self,
        weight: Optional[Tensor] = None,
        ignore_index: int = -100,
        reduction: str = 'mean',
        gamma: float = 2.
                 ) -> Tensor:
        """Implements the focal loss from
        `"Focal Loss for Dense Object Detection" <https://arxiv.org/pdf/1708.02002.pdf>`_
        Args:
            weight (torch.Tensor[K], optional): manual rescaling of each class
            ignore_index (int, optional): specifies target value that is ignored and do not contribute to gradient
            reduction (str, optional): reduction method
            gamma (float, optional): gamma parameter of focal loss
        Returns:
            torch.Tensor: loss reduced with `reduction` method
        """
        super().__init__()
        self.reduction = reduction
        self.weight = weight
        self.ignore_index = ignore_index
        self.gamma = gamma
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: (torch.Tensor[N, K, ...]), where K is the number of classes.
                You can pass logits or probabilities as input, if pass logit, must set softmax=True
            target: 
                (torch.Tensor[N, ...]): hard target tensor, where K is the number of classes, 
                It should contain binary values
       """
        logpt = F.log_softmax(input, dim=1)
        target = target.argmax(dim=-1)
        # Compute pt and logpt only for target classes (the remaining will have a 0 coefficient)
        logpt = logpt.transpose(1, 0).flatten(1).gather(0, target.view(1, -1)).squeeze()
        # Ignore index (set loss contribution to 0)
        valid_idxs = torch.ones(target.view(-1).shape[0], dtype=torch.bool, device=input.device)
        if self.ignore_index >= 0 and self.ignore_index < input.shape[1]:
            valid_idxs[target.view(-1) == self.ignore_index] = False

        # Get P(class)
        pt = logpt.exp()

        # Weight
        if self.weight is not None:
            # Tensor type
            if self.weight.type() != input.data.type():
                self.weight = self.weight.type_as(input.data)
            logpt = self.weight.gather(0, target.data.view(-1)) * logpt

        # Loss
        loss = -1 * (1 - pt) ** self.gamma * logpt

        # Loss reduction
        if self.reduction == 'sum':
            loss = loss[valid_idxs].sum()
        elif self.reduction == 'mean':
            loss = loss[valid_idxs].mean()
        else:
            # if no reduction, reshape tensor like target
            loss = loss.view(*target.shape)

        return loss
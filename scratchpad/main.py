import torch.nn.functional
from torch import tensor
from torch.nn.functional import l1_loss, mse_loss
from numpy import array

one = [1,2,3]
two = [4,5,6]
three = [7,-2,-5]

allNums = [one, two, three]

tns = tensor([
    one,
    two,
    three
])

tns2 = torch.stack([tensor(nums) for nums in allNums])


# arr = array([one, two, three])

#print(f"{arr.ndim} {arr.size} {arr.shape}")
print(f"{tns.ndim} {tns.size()} {tns.shape}")
print(f"{tns2.ndim} {tns2.size()} {tns2.shape}")

#print(arr * 0.2)

#tns2 = torch.nn.functional.relu(tns)
#print(tns2)

# l1 = l1_loss(tensor([0.0, 3.0]), tensor([8.0, 4.0]))
# l2 = mse_loss(tensor([0.0, 3.0]), tensor([3.0, 4.0]))
# print(l1)
# print(l2)
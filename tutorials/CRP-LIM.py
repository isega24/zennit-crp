from captum.attr import Lime, LimeBase
from captum._utils.models.linear_model import SkLearnLasso
from torchvision.models.swin_transformer import swin_b

from torchvision.models.resnet import resnet50
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import EpsilonPlusFlat
from crp.helper import get_layer_names
from crp.attribution import CondAttribution
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
import seaborn as sns
from matplotlib import pyplot as plt
import torchvision
from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = swin_b(pretrained=True).to(device)
model.eval()
layer_names = get_layer_names(
    model, types=[torch.nn.Conv2d, torch.nn.Linear, torch.nn.AdaptiveAvgPool2d]
)

pre_last_layer_name = layer_names[-2]
def get_pre_last_layer_output(x):
    for name, layer in model.named_children():
        x = layer(x)
        if name == pre_last_layer_name:
            return x

model.get_pre_last_layer_output = get_pre_last_layer_output

imagenet_dataset = torchvision.datasets.ImageFolder(
    "/mnt/homeGPU/jlsuarez/ECML-Arantxa/codes/nnguide/data/imagenet1k/imagenet/val/",
    transform=T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
)
    #root="/mnt/homeGPU/jlsuarez/ECML-Arantxa/codes/nnguide/data/imagenet1k/imagenet/",
    
segmentation_mask = torch.zeros((224,224), dtype=torch.int64)
n_parts = 10
for i in range(n_parts):
    i_step = 224 // n_parts
    for j in range(n_parts):
        j_step = 224 // n_parts
        segmentation_mask[
            i * i_step : (i + 1) * i_step, j * j_step : (j + 1) * j_step
        ] = (i * n_parts + j)

segmentation_mask = segmentation_mask.to(device)

sigma = n_parts * n_parts / 2
def perturb_func(input, **kwargs):
    segmentation_mask = kwargs.get("feature_mask")
    num_interp_features = segmentation_mask.max().item() + 1

    binary_vector = torch.zeros(
        (input.shape[0], num_interp_features), dtype=torch.int64, device=input.device
    )

    # Generate input.shape[0] random values from an exponential distribution
    value = np.random.exponential(scale=sigma, size=input.shape[0])
    value = np.clip(value, 1, num_interp_features - 1).astype(int)

    # select value indices from the binary vector and change them to 1
    for i in range(input.shape[0]):
        indices = torch.randperm(num_interp_features)[: value[i]]
        binary_vector[i, indices] = 1

    return binary_vector


def sim_func(original_input, perturbed_input, perturbed_interpretable_input, **kwargs):
    distance = 1 - torch.cosine_similarity(
        perturbed_interpretable_input,
        1.0 * torch.zeros_like(perturbed_interpretable_input) + 1,
        dim=1,
    )
    distance = torch.exp(-1 * (distance * distance) / (2 * sigma * sigma))
    return distance.item()


def from_interp_rep_transform(curr_sample, original_input, **kwargs):

    segmentation_mask = kwargs.get("feature_mask")
    num_interp_features = segmentation_mask.max().item() + 1

    mask = torch.zeros_like(segmentation_mask, dtype=torch.float32)

    for i in range(num_interp_features):
        if curr_sample[0, i] == 1:
            segment_indices = segmentation_mask == i
            mask[segment_indices] = 1.0

    mask = mask.unsqueeze(0).unsqueeze(1)

    transformed_input = original_input * mask
    # Plot the transformed input
    if False:
        trf_input_display = transformed_input.squeeze().permute(1, 2, 0).cpu().numpy()
        trf_input_display = (trf_input_display - trf_input_display.min()) / (
            trf_input_display.max() - trf_input_display.min()
        )
        plt.imshow(trf_input_display)
        plt.axis("off")
        plt.show()
    return transformed_input

imagenet_loader = torch.utils.data.DataLoader(
    imagenet_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)
for_biased_class = 231
detected_biased_features = [943,233]



mean_acc = 0.0
total_samples = 0
mean_acc_class_biased = 0.0
total_samples_class_biased = 0
imagenet_loader = tqdm(imagenet_loader, desc="Evaluating model before on ImageNet")
imagenet_loader.set_description("Evaluating model before feat unbias on ImageNet")
for batch in imagenet_loader:
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)

    outputs = model(inputs)
    
    _, preds = torch.max(outputs, 1)
    correct = (preds == targets).sum().item()
    mean_acc += correct
    total_samples += inputs.size(0)
    mean_acc_class_biased += (preds[targets == for_biased_class] == for_biased_class).sum().item()
    total_samples_class_biased += (
        targets == for_biased_class
    ).sum().item()
    
    
    imagenet_loader.set_postfix({"accuracy": f"{mean_acc / total_samples:.4f}","accuracy_class_biased": f"{1.0 if total_samples_class_biased == 0 else mean_acc_class_biased / total_samples_class_biased:.4f}"})
mean_acc /= total_samples
mean_acc_class_biased /= total_samples_class_biased
print(f"Mean accuracy on before ImageNet: {mean_acc:.4f}")
print(f"Mean accuracy on class {for_biased_class} before ImageNet: {mean_acc_class_biased:.4f}")





pre_state_dict = model.state_dict()
for feat in detected_biased_features:
    pre_state_dict["head.weight"][for_biased_class, feat] = 0
model.load_state_dict(pre_state_dict)

imagenet_loader = torch.utils.data.DataLoader(
    imagenet_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)
mean_acc = 0.0
total_samples = 0
mean_acc_class_biased = 0.0
total_samples_class_biased = 0
imagenet_loader = tqdm(imagenet_loader, desc="Evaluating model on ImageNet")
imagenet_loader.set_description("Evaluating model after feat unbias on ImageNet")
for batch in imagenet_loader:
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)

    outputs = model(inputs)
    
    _, preds = torch.max(outputs, 1)
    correct = (preds == targets).sum().item()
    mean_acc += correct
    total_samples += inputs.size(0)
    mean_acc_class_biased += (preds[targets == for_biased_class] == for_biased_class).sum().item()
    total_samples_class_biased += (
        targets == for_biased_class
    ).sum().item()
    
    
    imagenet_loader.set_postfix({"accuracy": f"{mean_acc / total_samples:.4f}","accuracy_class_biased": f"{1.0 if total_samples_class_biased == 0 else mean_acc_class_biased / total_samples_class_biased:.4f}"})
mean_acc /= total_samples
mean_acc_class_biased /= total_samples_class_biased
print(f"Mean accuracy on after ImageNet: {mean_acc:.4f}")
print(f"Mean accuracy on class {for_biased_class} after ImageNet: {mean_acc_class_biased:.4f}")


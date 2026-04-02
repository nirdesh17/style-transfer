import numpy as np
import onnx 
from onnx import helper, TensorProto
import onnxruntime as ort 
from PIL import Image
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(image):
    if torch.is_tensor(image):
        if image.dtype != torch.uint8:
            image = (image * 255).to(torch.uint8)
        image = Image.fromarray(image.cpu().numpy())
    img = image
    img = img.resize((512,512))
    img = torch.tensor(np.array(img.convert('RGB')), device=device)
    img = img / 255.
    img = img.permute(2,0,1)
    img = img.float().unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    img = (img - mean) / std
    return img.to(device)

def add_output(layer_name,graph):
    new_output=onnx.helper.make_tensor_value_info(
        name=layer_name,
        elem_type=TensorProto.FLOAT,
        shape=None
    )
    graph.output.append(new_output)

def run_inference(session,img):
    input_name=session.get_inputs()[0].name
    outputs=session.run(None,{input_name:img})
    return outputs

def content_loss(ori,gen):
    return torch.sum((ori-gen)**2)/2

def gramm_matrix(arr):
    B, C, H, W = arr.size()
    arr = arr.reshape(C, H * W)
    return arr @ arr.t()

def style_loss(ori,gen):
    B,C,H,W=ori.size()
    ori=gramm_matrix(ori)
    gen=gramm_matrix(gen)
    M = (H*W)
    return (torch.sum((ori-gen)**2)/(4*(C**2)*(M**2)))

def total_style_loss(w,E):
    ans=0
    for i in range(len(w)):
        ans=ans+(w[i]*E[i])
    return ans

def total_loss(content_representation,styles_representation,noise_img,model):
    wl=[0.2,0.2,0.2,0.2,0.2]
    alpha=1
    beta=1e4
    noise_content,noise_style=model(noise_img)

    loss_content=content_loss(content_representation[0],noise_content[0])
    loss_style=[]
    for i in range(len(noise_style)):
        loss_style.append(style_loss(styles_representation[i],noise_style[i]))

    loss_style_total=total_style_loss(wl,loss_style)

    loss_total=(alpha*loss_content)+(beta*loss_style_total)
    return loss_total

class VGGFeature(nn.Module):
    def __init__(self,content_output,style_output):
        super(VGGFeature, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        for i,layer in enumerate(vgg):
            if isinstance(layer,nn.ReLU):
                vgg[i] = nn.ReLU(inplace=False)
        self.vgg=vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.content_output=content_output
        self.style_output=style_output

    def forward(self, x):
        features_content=[]
        features_style=[]
        for name, layer in self.vgg._modules.items():
            x=layer(x)
            if name in self.content_output:
                features_content.append(x)
            elif name in self.style_output:
                features_style.append(x)
        return features_content , features_style

def optimize_loop(noise_img,content_representation,styles_representation,model,iter):
    noise_img = noise_img.clone().detach().contiguous().requires_grad_(True)
    optimize = torch.optim.LBFGS([noise_img])

    for i in range(iter):
        def closure():
            optimize.zero_grad()
            loss = total_loss(content_representation,styles_representation,noise_img,model)
            loss.backward()
            print(f"Iteration {i}, Loss={loss.item():.2f}")
            return loss
        optimize.step(closure)
    return noise_img

def tensor_to_image(tensor):
    img = tensor.clone().detach().squeeze(0)

    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3,1,1)

    img = img * std + mean
    img = torch.clamp(img,0,1)

    img = img.cpu().permute(1,2,0).numpy()
    img = (img * 255).astype("uint8")
    return Image.fromarray(img)



def main():
    p=argparse.ArgumentParser()
    p.add_argument("--content", required=True, help="pass content image")
    p.add_argument("--style", required=True, help="pass style image")
    p.add_argument("--output", required=False, help="output image path", default="../images/outputs/output.jpg")
    args=p.parse_args()

    try:
        content_image = preprocess(Image.open(args.content)).to(device)
        style_image = preprocess(Image.open(args.style)).to(device)
    except Exception as e:
        print(f"Error opening image: {e}")
        exit(1)

    content_output=["22"]
    style_output=["1","6","11","20","29"]

    model=VGGFeature(content_output,style_output).to(device)

    with torch.no_grad():
        outputs_content,_=model(content_image)
        _,outputs_style = model(style_image)


    content_representation = outputs_content
    styles_representation = outputs_style


    noise_img = torch.clone(style_image)
    result = optimize_loop(noise_img,content_representation,styles_representation,model,500)

    final_img=tensor_to_image(result)

    final_img.save(args.output)
    print(f"Output image saved at {args.output}")

if __name__ == "__main__":
    main()
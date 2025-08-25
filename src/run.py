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

def preprocess(image):
    if torch.is_tensor(image):
        if image.dtype != torch.uint8:
            image = (image * 255).to(torch.uint8)
        image = Image.fromarray(image.cpu().numpy())
    img = image
    img = img.resize((256,256))
    img = torch.tensor(np.array(img.convert('RGB')))
    img = img / 255.
    h, w = img.shape[0], img.shape[1]
    y0 = (h - 224) // 2
    x0 = (w - 224) // 2
    img = img[y0 : y0+224, x0 : x0+224, :]
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    img = (img - mean) / std 
    img = img.permute(2,0,1)
    img = img.float().unsqueeze(0)
    return img

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
    arr = arr.view(C, H * W)
    return arr @ arr.t()

def style_loss(ori,gen):
    B,C,H,W=ori.size()
    ori=gramm_matrix(ori)
    gen=gramm_matrix(gen)
    return (torch.sum((ori-gen)**2)/(4*(C*C)*(H*H*W*W)))

def total_style_loss(w,E):
    ans=0
    for i in range(len(w)):
        ans=ans+(w[i]*E[i])
    return ans

def total_loss(content_representation,styles_representation,noise_img,model):
    wl=[0.5,0.5,0.5,0.5,0.5]
    alpha=8
    beta=10000
    noise_output=model(noise_img)
    noise_content=noise_output[4:5]
    noise_style = noise_output[:4] + noise_output[5:]

    loss_content=content_loss(content_representation[0],noise_content[0])
    loss_style=[]
    for i in range(len(noise_style)):
        loss_style.append(style_loss(styles_representation[i],noise_style[i]))

    loss_style_total=total_style_loss(wl,loss_style)

    loss_total=(alpha*loss_content)+(beta*loss_style_total)
    return loss_total

class VGGFeature(nn.Module):
    def __init__(self,selected_layers):
        super(VGGFeature, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.vgg=vgg.eval()
        self.selected_layers=selected_layers

    def forward(self, x):
        features=[]
        for name, layer in self.vgg._modules.items():
            x=layer(x)
            if name in self.selected_layers:
                features.append(x)
        return features


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--content", required=True, help="pass content image")
    p.add_argument("--style", required=True, help="pass style image")
    args=p.parse_args()

    try:
        content_image = preprocess(Image.open(args.content))
        style_image = preprocess(Image.open(args.style))
    except Exception as e:
        print(f"Error opening image: {e}")
        exit(1)

    content_output=["21"]
    style_output=["0","5","10","19","28"]

    model=VGGFeature(selected_layers=content_output+style_output)

    with torch.no_grad():
        outputs_content=model(content_image)
        outputs_style = model(style_image)


    content_representation = outputs_content[4:5]
    styles_representation = outputs_style[:4] + outputs_style[5:]
    
    # noise_img=torch.rand(256,256,3)
    # noise_img=preprocess(noise_img)

    # print(total_loss(content_representation,styles_representation,noise_img,model))

if __name__ == "__main__":
    main()
import numpy as np
import onnx 
from onnx import helper, TensorProto
import onnxruntime as ort 
from PIL import Image
import argparse
import matplotlib.pyplot as plt

def preprocess(image):
    img = image
    img = img.resize((256,256))
    img = np.array(img.convert('RGB'))
    img = img / 255.
    h, w = img.shape[0], img.shape[1]
    y0 = (h - 224) // 2
    x0 = (w - 224) // 2
    img = img[y0 : y0+224, x0 : x0+224, :]
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = np.transpose(img, axes=[2, 0, 1])
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

def add_output(layer_name,graph):
    new_output=onnx.helper.make_tensor_value_info(
        name=layer_name,
        elem_type=TensorProto.FLOAT,
        shape=None
    )
    graph.output.append(new_output)

def run_inference(model_bytes,img):
    session = ort.InferenceSession(model_bytes)
    input_name=session.get_inputs()[0].name
    outputs=session.run(None,{input_name:img})
    return outputs

def content_loss(ori,gen):
    return np.sum((ori-gen)**2)//2

def gramm_matrix(arr):
    arr=np.squeeze(arr, axis=0)
    C, H, W = arr.shape
    arr=arr.reshape(C,H*W)
    return arr @ arr.T

def style_loss(ori,gen):
    B,C,H,W=ori.shape
    ori=gramm_matrix(ori)
    gen=gramm_matrix(gen)
    return (np.sum((ori-gen)**2)/(4*(B*B)(H*H*W*W)))

def total_style_loss(w,E):
    return np.sum(w*E)

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

    onnx_path=(f"../models/vgg19_Opset16.onnx")
    model=onnx.load(onnx_path)

    graph=model.graph
    content_output="/features/features.19/Conv_output_0"
    style_output=["/features/features.0/Conv_output_0","/features/features.5/Conv_output_0","/features/features.10/Conv_output_0","/features/features.19/Conv_output_0","/features/features.28/Conv_output_0"]

    add_output(content_output,graph)
    for out in style_output:
        add_output(out,graph)

    model_bytes=model.SerializePartialToString()

    content_representation=run_inference(model_bytes,content_image)[1]
    styles_representation=run_inference(model_bytes,style_image)[2:]
    
    # print(len(content_representation))
    # print(len(styles_representation))

if __name__ == "__main__":
    main()
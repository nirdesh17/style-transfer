import numpy as np 
import onnxruntime as ort 
from PIL import Image
import os

def preprocess(image):
    if not os.path.exists(image):
        raise OSError("File not found: {}".format(image))
    img = Image.open(image)
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

img=preprocess(f"../images/content/dog.jpg")
# print(img.shape)
onnx_path=(f"../models/vgg19_Opset16.onnx")

session = ort.InferenceSession(onnx_path)

input_name=session.get_inputs()[0].name
# print(input_name)
outputs=session.run(None,{input_name:img})


# /features/features.21/Conv

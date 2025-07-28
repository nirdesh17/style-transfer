import numpy as np
import onnx 
from onnx import helper, TensorProto
import onnxruntime as ort 
from PIL import Image
import os

onnx_path=(f"../models/vgg19_Opset16.onnx")

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

model=onnx.load(onnx_path)
graph=model.graph

content_output="/features/features.19/Conv_output_0"
style_output=["/features/features.0/Conv_output_0","/features/features.5/Conv_output_0","/features/features.10/Conv_output_0","/features/features.19/Conv_output_0","/features/features.28/Conv_output_0"]

# outputs=graph.output
# # print(outputs)
# node=graph.node
# print(node)


new_output=onnx.helper.make_tensor_value_info(
    name=content_output,
    elem_type=TensorProto.FLOAT,
    shape=None
)

graph.output.append(new_output)


for out in style_output:
    new_output=onnx.helper.make_tensor_value_info(
        name=out,
        elem_type=TensorProto.FLOAT,
        shape=None
    )
    graph.output.append(new_output)

# onnx.save(model,"test.onnx")


model_bytes=model.SerializePartialToString()
session = ort.InferenceSession(model_bytes)
input_name=session.get_inputs()[0].name
outputs=session.run(None,{input_name:img})
print(len(outputs))
for i in outputs:
    print(i.shape)


content_reprentation=outputs[1]
style_1=outputs[2]
style_2=outputs[3]
style_3=outputs[4]
style_4=outputs[5]
style_5=outputs[6]


print(content_reprentation)

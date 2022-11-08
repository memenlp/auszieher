# python -m transformers.onnx --model=./checkpoint-1500/ --feature=sequence-classification onnx/ 模型导出
import time

import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from onnxruntime import InferenceSession

model_path = './data/checkpoint-1500'
tokenizer = AutoTokenizer.from_pretrained(model_path)

COUNT = 5
query = "Using DistilBERT with ONNX Runtime!"

model = AutoModelForSequenceClassification.from_pretrained(model_path)
inputs = tokenizer(query, return_tensors="pt")
model.eval()
start_time = time.time()
for i in range(COUNT):
    torch_out = model(**inputs)
end_time = time.time()
print(f'torch cost time:{end_time - start_time}')

onnx_path = './data/hypo/en/model.onnx'
session = InferenceSession(onnx_path)
inputs = tokenizer(query, return_tensors="np")
start_time = time.time()
for i in range(COUNT):
    ort_outs = session.run(output_names=["logits"], input_feed=dict(inputs))
end_time = time.time()
print(f'torch cost time:{end_time - start_time}')


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


np.testing.assert_allclose(to_numpy(torch_out.logits), ort_outs[0], rtol=1e-03, atol=1e-05)

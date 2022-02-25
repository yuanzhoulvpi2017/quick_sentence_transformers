##################################################################################
# 分为三个部分
# 1. tokenizer部分
# 2. transformer部分
# 3. pooling部分


from multiprocessing.pool import Pool
import numpy as np
import onnxruntime
import psutil
from sympy import im
from transformers import (AutoConfig, AutoModel, AutoTokenizer)
import os
import json
from sentence_transformers.models import Pooling

##################################################################################
# 处理transformer和 tokenizer部分

big_model_path = "../models/paraphrase-multilingual-MiniLM-L12-v2"

modules_json_path = os.path.join(big_model_path, 'modules.json')
with open(modules_json_path) as fIn:
    modules_config = json.load(fIn)

tf_from_s_path = os.path.join(big_model_path, modules_config[0].get('path'))


# 基本参数

max_seq_length = 128
doc_stride = 128
max_query_length = 64
# Enable overwrite to export onnx model and download latest script each time when running this notebook.
enable_overwrite = True
# Total samples to inference. It shall be large enough to get stable latency measurement.
total_samples = 1000


# # Load pretrained model and tokenizer
# Load pretrained model and tokenizer
config_class, model_class, tokenizer_class = (
    AutoConfig, AutoModel, AutoTokenizer)

cache_dir = os.path.join(".", "cache_models")
config = config_class.from_pretrained(tf_from_s_path, cache_dir=cache_dir)
tokenizer = tokenizer_class.from_pretrained(
    tf_from_s_path, do_lower_case=True, cache_dir=cache_dir)
model_transformer = model_class.from_pretrained(
    tf_from_s_path, from_tf=False, config=config, cache_dir=cache_dir)


##################################################################################
# 使用onnx 和cuda推理部分

output_dir = os.path.join("..", "onnx_models")
export_model_path = os.path.join(output_dir, 'Multilingual_MiniLM_L12.onnx')

device_name = 'gpu'
sess_options = onnxruntime.SessionOptions()
sess_options.optimized_model_filepath = os.path.join(
    output_dir, "optimized_model_{}.onnx".format(device_name))
# Please change the value according to best setting in Performance Test Tool result.
sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)
session = onnxruntime.InferenceSession(
    export_model_path, sess_options, providers=['CUDAExecutionProvider'])

##################################################################################
# 处理pooling部分



pooling_model_path = os.path.join(big_model_path, modules_config[0].get('path'))

pooling_model = Pooling.load(input_path=pooling_model_path)




##################################################################################
# 生成inputs数据

st = ['您好']
inputs = tokenizer(
    st,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)
inputs

ort_inputs = {k:v.cpu().numpy() for k, v in inputs.items()}
ort_outputs_gpu = session.run(None, ort_inputs)


if __name__ =='__main__':
    print(ort_outputs_gpu)

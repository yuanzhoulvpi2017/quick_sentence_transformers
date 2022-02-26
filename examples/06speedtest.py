
##################################################################################
# 分为三个部分
# 1. tokenizer部分
# 2. transformer部分
# 3. pooling部分


import time
import torch
from torch import Tensor
from torch import nn
from typing import Union, Tuple, List, Iterable, Dict
import os
import json


class Pooling(nn.Module):
    def __init__(self,
                 word_embedding_dimension: int,
                 pooling_mode: str = None,
                 pooling_mode_cls_token: bool = False,
                 pooling_mode_max_tokens: bool = False,
                 pooling_mode_mean_tokens: bool = True,
                 pooling_mode_mean_sqrt_len_tokens: bool = False,
                 ):
        super(Pooling, self).__init__()

        self.config_keys = ['word_embedding_dimension',  'pooling_mode_cls_token', 'pooling_mode_mean_tokens', 'pooling_mode_max_tokens', 'pooling_mode_mean_sqrt_len_tokens']

        if pooling_mode is not None:        #Set pooling mode by string
            pooling_mode = pooling_mode.lower()
            assert pooling_mode in ['mean', 'max', 'cls']
            pooling_mode_cls_token = (pooling_mode == 'cls')
            pooling_mode_max_tokens = (pooling_mode == 'max')
            pooling_mode_mean_tokens = (pooling_mode == 'mean')

        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens

        pooling_mode_multiplier = sum([pooling_mode_cls_token, pooling_mode_max_tokens, pooling_mode_mean_tokens, pooling_mode_mean_sqrt_len_tokens])
        self.pooling_output_dimension = (pooling_mode_multiplier * word_embedding_dimension)


    def __repr__(self):
        return "Pooling({})".format(self.get_config_dict())

    def get_pooling_mode_str(self) -> str:
        """
        Returns the pooling mode as string
        """
        modes = []
        if self.pooling_mode_cls_token:
            modes.append('cls')
        if self.pooling_mode_mean_tokens:
            modes.append('mean')
        if self.pooling_mode_max_tokens:
            modes.append('max')
        if self.pooling_mode_mean_sqrt_len_tokens:
            modes.append('mean_sqrt_len_tokens')

        return "+".join(modes)

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features['token_embeddings']
        attention_mask = features['attention_mask']

        ## Pooling strategy
        output_vectors = []
        if self.pooling_mode_cls_token:
            cls_token = features.get('cls_token_embeddings', token_embeddings[:, 0])  # Take first token by default
            output_vectors.append(cls_token)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            #If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        output_vector = torch.cat(output_vectors, 1)
        features.update({'sentence_embedding': output_vector})
        return features

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        return Pooling(**config)


import numpy as np
import onnxruntime
import psutil
from sympy import im
from transformers import (AutoConfig, AutoModel, AutoTokenizer)
import os
import json
# from sentence_transformers.models import Pooling

from sentence_transformers import SentenceTransformer as SBert


from tqdm import tqdm
import torch as t


# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# # from sentence_transformers.util import cos_sim
# import uvicorn
# from asgiref.sync import sync_to_async
# from pathlib import Path
# import hashlib
# import time



big_model_path = "../models/paraphrase-multilingual-MiniLM-L12-v2"

# max_seq_length = 128
doc_stride = 128
max_query_length = 64
# enable_overwrite = True
total_samples = 1000

config_class, model_class, tokenizer_class = (
    AutoConfig, AutoModel, AutoTokenizer)


class OnnxInfer:
    def __init__(self, big_model_path,device_name = 'gpu', max_seq_length=128):
        self.max_seq_length = max_seq_length
        self.device_name = device_name
        self.big_model_path = big_model_path
        modules_json_path = os.path.join(self.big_model_path, 'modules.json')
        with open(modules_json_path) as fIn:
            self.modules_config = json.load(fIn)

        self.tf_from_s_path = os.path.join(big_model_path, self.modules_config[0].get('path'))

        cache_dir = os.path.join(".", "cache_models")
        config = config_class.from_pretrained(self.tf_from_s_path, cache_dir=cache_dir)
        self.tokenizer = tokenizer_class.from_pretrained(
            self.tf_from_s_path, do_lower_case=True, cache_dir=cache_dir)
        self.model_transformer = model_class.from_pretrained(
            self.tf_from_s_path, from_tf=False, config=config, cache_dir=cache_dir)


        self.output_dir = os.path.join("..", "onnx_models")
        self.export_model_path = os.path.join(self.output_dir, 'optimized_model_gpu.onnx')
        
        sess_options = onnxruntime.SessionOptions()
        # sess_options.optimized_model_filepath = os.path.join(
        #     output_dir, "optimized_model_{}.onnx".format(device_name))
        sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)


        self.session = onnxruntime.InferenceSession(
            self.export_model_path, sess_options, providers=['CUDAExecutionProvider'])

        pooling_model_path = os.path.join(
            big_model_path, self.modules_config[1].get('path'))
        self.pooling_model = Pooling.load(pooling_model_path)


    def encode(self,sentences):
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'):
            sentences = [sentences]

        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        ort_inputs = {k: v.cpu().numpy() for k, v in inputs.items()}
        ort_outputs_gpu = self.session.run(None, ort_inputs)
        ort_result = self.pooling_model.forward(features={'token_embeddings': t.Tensor(ort_outputs_gpu[0]),
                                                    'attention_mask': inputs.get('attention_mask')})
        result = ort_result.get('sentence_embedding')
        return result


# _ = [inferpart(session=session, sentences = ['您好'], pooling_model=pooling_model) for i in tqdm(range(2000))]

# 使用原生的sentence transformer代码
model_sbert_raw = SBert(big_model_path, device='cuda')

# _ = [model_sbert_raw.encode(['您好'],device='cuda') for i in tqdm(range(2000))]


onnxinfer = OnnxInfer(big_model_path=big_model_path)





##############################################################################


if __name__ =='__main__':
    sentencs = '您好, 我来着中国，我很喜欢中国，晚上可以电话吗，好饿啊，我想下班，不想工作了，真的好烦男奶奶拿啊啊啊啊'

    print(f"总字数: {len(sentencs)+1}")
    # 测试本地性能
    total_samples = 5000
    latency = []
    for i in tqdm(range(total_samples)):
        start = time.time()
        ort_result = onnxinfer.encode(sentences=[f"{sentencs}_{i}"])
        latency.append(time.time() - start)
    
    print('*'* 40)
    print("type: {} Inference time = {} ms".format('onnx infer', format(sum(latency) * 1000 / len(latency), '.2f')))
    print('*'* 40)
    latency = []
    for i in tqdm(range(total_samples)):
        start = time.time()
        raw_result = model_sbert_raw.encode([f"{sentencs}_{i}"],device='cuda')
        latency.append(time.time() - start)
    
    
    print('*'* 40)
    print("type: {} Inference time = {} ms".format('raw infer', format(sum(latency) * 1000 / len(latency), '.2f')))
    print('*'* 40)








# app = FastAPI()
# origins = ["*"]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get('/nlp_service/ov')
# async def sentenc2vector_async(sentence: str):
#     """
#     onnx version
#     """
#     result = await sync_to_async(onnxinfer.encode)(sentence)
#     # result = inferpart(session, sentence, pooling_model, tokenizer)
#     # print(result)
#     return {'vector':result.numpy().tolist()}





# # nohup /root/anaconda3/envs/mynet/bin/python main.py 1>>nlp_api_log.out &


# if __name__ == '__main__':
#     uvicorn.run(app='06speedtest:app', 
#                 host="0.0.0.0", 
#                 workers=8,
#                 port=8001)

"""
参考代码: https://github.com/UKPLab/sentence-transformers/tree/master/sentence_transformers
代码结构：
-. 检查sbert模型文件是否存在 并且进行相关操作 保证模型已经保存在计算机本地
-. 将sbert模型的transformer部分从pytorch转换成onnx文件
-. 提供一个encode接口
"""
from optparse import Option
from statistics import mode
from tkinter.messagebox import NO
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
import psutil
import time
import torch
from torch import Tensor
from torch import nn
from typing import Union, Tuple, List, Iterable, Dict
import os
import json
import pandas as pd
import numpy as np
import onnxruntime
import psutil
from sympy import im
from transformers import AutoConfig, AutoModel, AutoTokenizer
import os
import json
from tqdm import tqdm
import torch as t
from sentence_transformers import __MODEL_HUB_ORGANIZATION__
from sentence_transformers import __version__
from sentence_transformers.util import snapshot_download
from sentence_transformers.models import Pooling


class InferSentenceTransformer(object):
    def __init__(self, model_name_or_path: Optional[str] = None,
                 device: Optional[str] = None,
                 cache_folder: Optional[str] = None,
                 onnx_folder: Optional[str] = None,
                 onnx_model_name: Optional[str] = None,
                 enable_overwrite: Optional[bool] = True
                 ):
        """
        model_name_or_path:特指sentence-transformer模型的名称或者路径
        device: 使用的是cpu还是cuda。默认为gpu的第0个显卡, 使用的是onnxruntime而不是tenosrrt
        cache_folder:transformers的缓冲路径
        onnx_folder: onnx文件的的路径
        onnx_model_name: onnx文件名称
        enable_overwrite: 是否让onnx文件被新的文件给覆盖掉
        """
        if onnx_folder is None:
            onnx_folder = os.path.join(
                os.getcwd(), "InferSentenceTransformer_onnx")
            os.makedirs(onnx_folder, exist_ok=True)

        if cache_folder is None:
            cache_folder = os.getenv('SENTENCE_TRANSFORMERS_HOME')
            if cache_folder is None:
                try:
                    from torch.hub import _get_torch_home

                    torch_cache_home = _get_torch_home()
                except ImportError:
                    torch_cache_home = os.path.expanduser(os.getenv(
                        'TORCH_HOME', os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))

                cache_folder = os.path.join(
                    torch_cache_home, 'sentence_transformers')

        if model_name_or_path is not None and model_name_or_path != "":

            # Old models that don't belong to any organization
            basic_transformer_models = ['albert-base-v1', 'albert-base-v2', 'albert-large-v1', 'albert-large-v2', 'albert-xlarge-v1', 'albert-xlarge-v2', 'albert-xxlarge-v1', 'albert-xxlarge-v2', 'bert-base-cased-finetuned-mrpc', 'bert-base-cased', 'bert-base-chinese', 'bert-base-german-cased', 'bert-base-german-dbmdz-cased', 'bert-base-german-dbmdz-uncased', 'bert-base-multilingual-cased', 'bert-base-multilingual-uncased', 'bert-base-uncased', 'bert-large-cased-whole-word-masking-finetuned-squad', 'bert-large-cased-whole-word-masking', 'bert-large-cased', 'bert-large-uncased-whole-word-masking-finetuned-squad', 'bert-large-uncased-whole-word-masking', 'bert-large-uncased', 'camembert-base', 'ctrl', 'distilbert-base-cased-distilled-squad', 'distilbert-base-cased', 'distilbert-base-german-cased', 'distilbert-base-multilingual-cased', 'distilbert-base-uncased-distilled-squad',
                                        'distilbert-base-uncased-finetuned-sst-2-english', 'distilbert-base-uncased', 'distilgpt2', 'distilroberta-base', 'gpt2-large', 'gpt2-medium', 'gpt2-xl', 'gpt2', 'openai-gpt', 'roberta-base-openai-detector', 'roberta-base', 'roberta-large-mnli', 'roberta-large-openai-detector', 'roberta-large', 't5-11b', 't5-3b', 't5-base', 't5-large', 't5-small', 'transfo-xl-wt103', 'xlm-clm-ende-1024', 'xlm-clm-enfr-1024', 'xlm-mlm-100-1280', 'xlm-mlm-17-1280', 'xlm-mlm-en-2048', 'xlm-mlm-ende-1024', 'xlm-mlm-enfr-1024', 'xlm-mlm-enro-1024', 'xlm-mlm-tlm-xnli15-1024', 'xlm-mlm-xnli15-1024', 'xlm-roberta-base', 'xlm-roberta-large-finetuned-conll02-dutch', 'xlm-roberta-large-finetuned-conll02-spanish', 'xlm-roberta-large-finetuned-conll03-english', 'xlm-roberta-large-finetuned-conll03-german', 'xlm-roberta-large', 'xlnet-base-cased', 'xlnet-large-cased']

            if os.path.exists(model_name_or_path):
                # Load from path
                model_path = model_name_or_path
            else:
                # Not a path, load from hub
                if '\\' in model_name_or_path or model_name_or_path.count('/') > 1:
                    raise ValueError(
                        "Path {} not found".format(model_name_or_path))

                if '/' not in model_name_or_path and model_name_or_path.lower() not in basic_transformer_models:
                    # A model from sentence-transformers
                    model_name_or_path = __MODEL_HUB_ORGANIZATION__ + "/" + model_name_or_path

                model_path = os.path.join(
                    cache_folder, model_name_or_path.replace("/", "_"))

                # Download from hub with caching
                snapshot_download(model_name_or_path,
                                  cache_dir=cache_folder,
                                  library_name='sentence-transformers',
                                  library_version=__version__,
                                  ignore_files=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5'])

        onnxproviders = onnxruntime.get_available_providers()
        # ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']

        if device == 'cpu':
            fast_onnxprovider = 'CPUExecutionProvider'
        else:
            if 'CUDAExecutionProvider' not in onnxproviders:
                fast_onnxprovider = 'CPUExecutionProvider'
            else:
                fast_onnxprovider = 'CUDAExecutionProvider'

        self.onnx_folder = onnx_folder  
        self.model_path = model_path # sbert模型文件夹
        self.fast_onnxprovider = fast_onnxprovider # onnxruntime的后端
        self.cache_folder = cache_folder # tranformer的缓冲文件夹

        # self.onnx_model_anme = onnx_model_name
        self.enable_overwrite = enable_overwrite

        self.export_model_name = os.path.join(self.onnx_folder, f"{onnx_model_name}.onnx") # 要保存使用的最终文件名称

        if os.path.exists(os.path.join(self.model_path, 'modules.json')):    #Load as SentenceTransformer model
            # 注释： 这里基本上sbert模型已经没有问题了，但是sbert还支持一些非sbert模型，这一部分还要考虑。

            # 接下来是做模型转换部分
            self.sbertmodel2onnx()

            # 接下来是做推理部分
            # onnx infer 部分
            self.session = self._load_sbert_session()
            # pooling 部分
            self.pooling_model = self._load_sbert_pooling()
        else:   #Load with AutoModel
            # 待补充
            # modules = self._load_auto_model(model_path)
            pass 


    def sbertmodel2onnx(self):
        """
        将sbert的第一个transformer模型转换成onnx格式文件
        并且保存在onnx_folder文件夹中
        """
        # 加载第一个模型
        model_json_path = os.path.join(self.model_path, 'modules.json')
        with open(model_json_path) as fIn:
            modules_config = json.load(fIn)
        tf_from_s_path = os.path.join(
            self.model_path, modules_config[0].get('path'))

        # Load pretrained model and tokenizer
        config_class, model_class, tokenizer_class = (
            AutoConfig, AutoModel, AutoTokenizer)

        config = config_class.from_pretrained(
            tf_from_s_path, cache_dir=self.cache_folder)
        tokenizer = tokenizer_class.from_pretrained(
            tf_from_s_path, do_lower_case=True, cache_dir=self.cache_folder)
        model = model_class.from_pretrained(
            tf_from_s_path, from_tf=False, config=config, cache_dir=self.cache_folder)

        self.tokenizer = tokenizer

        # onnx_model_name = self.onnx_model_anme

        model.eval()
        device = t.device('cpu')

        st = ['您好 hello']
        inputs = tokenizer(
            st,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt")

        if self.enable_overwrite or not os.path.exists(self.export_model_name):
            with torch.no_grad():
                symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
                torch.onnx.export(model,                                            # model being run
                                  # model input (or a tuple for multiple inputs)
                                  args=tuple(inputs.values()),
                                  # where to save the model (can be a file or file-like object)
                                  f=self.export_model_name,
                                  # the ONNX version to export the model to
                                  opset_version=11,
                                  # whether to execute constant folding for optimization
                                  do_constant_folding=True,
                                  input_names=['input_ids',                         # the model's input names
                                               'attention_mask',
                                               'token_type_ids'],
                                  # the model's output names
                                  output_names=['start', 'end'],
                                  dynamic_axes={'input_ids': symbolic_names,        # variable length axes
                                                'attention_mask': symbolic_names,
                                                'token_type_ids': symbolic_names,
                                                'start': symbolic_names,
                                                'end': symbolic_names})
            print(f"Model exported at: {self.export_model_name}")

    def encode(self, sentences):
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

    def _load_sbert_pooling(self):
        """
        只是处理sbert模型的pooling
        """
        model_json_path = os.path.join(self.model_path, 'modules.json')
        with open(model_json_path) as fIn:
            modules_config = json.load(fIn)

        pooling_model_path = os.path.join(
            self.model_path, modules_config[1].get('path')
        )
        pooling_model = Pooling.load(pooling_model_path)
        return pooling_model
    


    def _load_sbert_session(self):
        """
        只是加载sbert的transformer部分转换的onnx文件
        """
        # self.output_dir = os.path.join("..", "onnx_models")
        # self.export_model_path = self.export_model_name#os.path.join(self.output_dir, 'optimized_model_gpu.onnx')

        sess_options = onnxruntime.SessionOptions()
        # sess_options.optimized_model_filepath = os.path.join(
        #     output_dir, "optimized_model_{}.onnx".format(device_name))
        sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)

        session = onnxruntime.InferenceSession(
            self.export_model_name, sess_options, providers=[self.fast_onnxprovider])
        return session


if __name__ == '__main__':
    ifsbert = InferSentenceTransformer(
        # "/home/user_huzheng/documents/quick_sentence_transformers/models/paraphrase-multilingual-MiniLM-L12-v2",
        model_name_or_path="nli-bert-base",
        device='cuda',
        onnx_model_name="test_onnxmodel02",
        enable_overwrite=False
    )
    # ifsbert.pytorchmodel2onnx()
    data = ifsbert.encode(sentences=['hello', 'ok'])
    print(data.shape)
    # ifsbert

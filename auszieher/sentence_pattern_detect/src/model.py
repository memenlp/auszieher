from transformers import AutoTokenizer
import torch
import numpy as np
import onnxruntime
import pandas as pd
import os

class Model:
    def __init__(self, data_dir, batch_size=8, logger = None):
        if logger is None:
            from .commons.logger import logger
            self.logger = logger
        else:
            self.logger = logger

        self.lang2model = {}
        self.lang2tokenizer = {}
        self.lang2label_id2label = {}
        self.lang2label2threshold = {}
        self.batch_size = batch_size

        for lang in ['zh', 'en']:  # 根据语种区分文件夹
            dirname = lang
            if not os.path.isdir(os.path.join(data_dir, dirname)):
                continue

            tmp_tokenizer_path = os.path.join(data_dir, dirname)
            tokenizer = AutoTokenizer.from_pretrained(tmp_tokenizer_path)

            tmp_label_threshold_path = os.path.join(data_dir, dirname, 'label_threshold.csv')
            label_threshold_df = pd.read_csv(tmp_label_threshold_path, header=None)  # 兼容没有label.csv的情况
            label_list = label_threshold_df[0].tolist()
            label_id2label = {}
            label2label_id = {}
            label2threshold = {}
            for i, label in enumerate(label_list):
                label_id2label[i] = label
                label2label_id[label] = i
                label2threshold[label] = label_threshold_df[1][i]

            # 验证onnx模型是否一致
            tmp_model_path = os.path.join(data_dir, dirname, 'model.onnx')
            model = onnxruntime.InferenceSession(tmp_model_path)

            self.lang2model[lang] = model
            self.lang2label_id2label[lang] = label_id2label
            self.lang2tokenizer[lang] = tokenizer
            self.lang2label2threshold[lang] = label2threshold
            self.logger.info(f'load {lang} model success.')

    def process(self, text_list, lang):
        result = []  # 返回结果

        total_proba_list = []
        for i in range(0, len(text_list), self.batch_size):
            tmp_text_list = text_list[i:i + self.batch_size]
            inputs = self.lang2tokenizer[lang](tmp_text_list, padding=True, max_length=128, truncation=True,
                                               return_tensors='np')
            onnx_outputs = self.lang2model[lang].run(output_names=["logits"], input_feed=dict(inputs))
            proba_list = torch.softmax(torch.tensor(onnx_outputs[0]), dim=1).cpu().tolist()
            total_proba_list.extend(proba_list)
            label2threshold = self.lang2label2threshold[lang]
            label_id2label = self.lang2label_id2label[lang]

            # 0是其他, 1是假设型
            for proba in proba_list:
                index = np.argmax(proba)
                if label_id2label[index] == '其他':
                    result.append((0, proba[index]))
                else:
                    if proba[index] >= label2threshold[label_id2label[index]]:
                        result.append((1, proba[index]))
                    else:
                        result.append((0, 1 - proba[index]))
        assert len(text_list) == len(total_proba_list), 'inputted sample number not equal returned sample number'
        self.logger.info(f'proba result:{total_proba_list}')
        return result

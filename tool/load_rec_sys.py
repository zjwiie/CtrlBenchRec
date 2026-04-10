# # Load model directly
# from transformers import AutoTokenizer, AutoModelForMaskedLM
#
# tokenizer = AutoTokenizer.from_pretrained("../twhin-bert-base")
# model = AutoModelForMaskedLM.from_pretrained("../twhin-bert-base")
from sentence_transformers import SentenceTransformer
from sympy.printing.pytorch import torch
# Load model directly
from transformers import AutoTokenizer, AutoModel

from model.narm_recommender import NarmRecommender
from model.sasrec import SASRec
from model.sasrec_recommender import SasRecRecommender
from tool.path_constants import PathConstants


tokenizer = AutoTokenizer.from_pretrained(PathConstants.bge_model_path)
model = AutoModel.from_pretrained(PathConstants.bge_model_path)

bge_tokenizer = None
bge_model = None
twhin_tokenizer = None  # <--- 添加这一行
twhin_model = None

RAW_DATASET_ROOT_FOLDER = 'data'
GEN_DATASET_ROOT_FOLDER = 'gen_data'

STATE_DICT_KEY = 'model_state_dict'
OPTIMIZER_STATE_DICT_KEY = 'optimizer_state_dict'

import pickle


def get_bge_tokenizer():
    global bge_tokenizer
    if bge_tokenizer is None:
        from transformers import AutoTokenizer
        bge_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=PathConstants.bge_model_path,
            model_max_length=512)
    return bge_tokenizer


def get_bge_model(device):
    global bge_model
    if bge_model is None:
        # 加载 BGE 模型用于实际推荐
        bge_model = SentenceTransformer(PathConstants.bge_model_path, device='cpu')
    return bge_model

def get_twhin_tokenizer():
    global twhin_tokenizer
    if twhin_tokenizer is None:
        from transformers import AutoTokenizer
        twhin_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=PathConstants.twhin_bert_base_model_path,
            model_max_length=512)
    return twhin_tokenizer

def get_sasrec():
    class InferenceArgs:
        def __init__(self):
            # 基础结构参数
            self.num_items = 3952  # 注意：这里需要填入你 ml-1m 处理后的真实物品数量
            self.bert_hidden_units = 64
            self.bert_num_blocks = 2
            self.bert_num_heads = 2
            self.bert_head_size = None  # 对应代码里的设置
            self.bert_max_len = 200
            self.bert_dropout = 0.1
            self.bert_attn_dropout = 0.1
            self.device = 'cpu'

    # 使用方法
    args = InferenceArgs()
    model = SASRec(args)
    # 添加 map_location 参数，强制将显存里的权重映射到内存（CPU）中
    model.load_state_dict(torch.load('../encoder/narm/best_acc_model.pth', map_location=torch.device('cpu')))
    model.eval()

def run_narm():
    narm_engine = NarmRecommender()
    test_ids = ['1046', '2213', '45', '668','0']
    blacklist = ['2720']

    # 现在的调用非常简洁：只需提供原始ID和黑名单ID
    items, scores = narm_engine.recommend(test_ids, neg_ids=blacklist, topk=3)

    print("\n>>>> Custom Recommendation Results <<<<")
    print(f"Items  : {items}")
    print(f"Scores : {scores}")

def run_sasrec():
    sasrec_engine = SasRecRecommender()
    test_ids = ['1046', '2213', '45', '668','0']
    blacklist = ['2720']

    # 现在的调用非常简洁：只需提供原始ID和黑名单ID
    items, scores = sasrec_engine.recommend(test_ids, neg_ids=blacklist, topk=3)

    print("\n>>>> Custom Recommendation Results <<<<")
    print(f"Items  : {items}")
    print(f"Scores : {scores}")

if __name__ == "__main__":

    run_sasrec()

    # run_narm()

    # 以“二进制读取”模式（rb）打开文件
    # with open('../model/dataset.pkl', 'rb') as f:
    #     data = pickle.load(f)
    #
    # print(data['smap'])
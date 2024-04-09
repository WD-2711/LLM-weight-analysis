import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def random_tensor(param):
    rows, cols = param.size()
    mean = param.mean()
    std = param.std()
    random_tensor = torch.randn(rows, cols)
    return random_tensor*std+mean

def model_change(layer_num, model_weights):
    old_param = {}
    for (name, param) in model_params:
        st = "." + str(layer_num) + "."
        if st in name and "self_attn" in name:
            old_param[name] = param
            new_weight = random_tensor(param)
            model_weights[name] = new_weight
    print("[+] ", old_param.keys())
    return old_param, model_weights

def model_revert(old_param, model_weights):
    for name in old_param.keys():
        model_weights[name] = old_param[name]    
    return model_weights

def text_similarity(text1, text2):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return cosine_sim[0][0]

def model_evaluate(model, tokenizer):
    ques_and_ans = {
        "Q: Where is the capital of China?\n":"A: The capital of China is Beijing.",
        "Q: What is the longest river in china?\n":"A: The longest river in China is the Yangtze River.",
        "Q: What is the highest mountain in the world?\n":"A: The highest mountain in the world is Mount Everest.",
        "Q: Where is the capital of France?\n":"A: The capital of France is Paris.",
        "Q: What is the largest continent in the world?\n":"A: The largest continent in the world is the continent of Antarctica."
    }

    ret = []
    for q in ques_and_ans.keys():
        input_ids = tokenizer(q, return_tensors="pt").input_ids
        tokens = model.generate(input_ids, max_length=50)
        result = tokenizer.decode(tokens[0].tolist(), skip_special_tokens=True)
        if "A:" in result:
            result = result[result.index("A:"):]
        if "B:" in result:
            result = result[:result.index("B:")]
        if "\n" in result:
            result = result[:result.index("\n")]
        ret.append(text_similarity(result, ques_and_ans[q]))
    
    return ret

if __name__ == "__main__":
    model_path = './LiteLlama-460M-1T'
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model_weights = model.state_dict()
    model_params = [(name, param) for name, param in model_weights.items()]

    total_layer = 24
    for layer_num in range(total_layer):
        old_param, changed_model_weights = model_change(layer_num, model_weights)
        model.load_state_dict(changed_model_weights)

        ret = model_evaluate(model, tokenizer)
        print('[+] Layer {:2d} {}'.format(layer_num, ret))

        origin_model_weights = model_revert(old_param, changed_model_weights)
        model.load_state_dict(origin_model_weights)





from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from data_generator import make_datasets

def model_process(model_path, dataset):
    ret = {"succ":0, "fail":0}
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    t_bar = tqdm(total=len(dataset))
    for question, answer in dataset:
        input_ids = tokenizer(question, return_tensors="pt").input_ids
        tokens = model.generate(input_ids, max_length=300)
        result = tokenizer.decode(tokens[0].tolist(), skip_special_tokens=True)[:30]
        if answer in result:
            ret['succ'] += 1
        else:
            ret['fail'] += 1
        
        t_bar.update(1)
        t_bar.set_description("Accuracy:{:.2f}".format(ret['succ']/(ret['succ']+ret['fail'])))
        t_bar.refresh()
    
    print("[+] succ:{:4d} | fail:{:4d}".format(ret['succ'], ret['fail']))
    return ret

if __name__ == "__main__":
    model_path = './LiteLlama-460M-1T'
    dirs = ["./data/bbh", "./data/boolq"]
    dataset = make_datasets(dirs)
    model_process(model_path, dataset)
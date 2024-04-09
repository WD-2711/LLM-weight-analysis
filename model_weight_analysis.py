import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

def weight_analysis(name, param):
    ret = {"name":name,
           "size":tuple(param.size()),
           "mean":param.mean().item(), 
           "std":param.std().item(),
           "max":param.max().item(), 
           "min":param.min().item()}

    # Draw hot image
    if param.dim() == 2:
        min_val = param.min()
        max_val = param.max()
        normalized_tensor = (param-min_val)/(max_val-min_val)    
        plt.imshow(normalized_tensor, cmap='hot')
        plt.title(name)
        plt.colorbar()
        plt.savefig('./weight_analysis/' + name.replace(".", "_") + ".png")
        plt.clf()
    
    return ret

if __name__ == "__main__":
    # Print model structure
    model_path = './LiteLlama-460M-1T'
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("[+] model structure")
    print("-"*100)
    print(model)
    print("-"*100)

    # Analysis model weight
    csv_name = "./weight_analysis/weight.csv"
    model_weights = model.state_dict()
    with open(csv_name, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["name", "size", "mean", "std", "max", "min"])
        writer.writeheader()
        for name, param in tqdm(model_weights.items()):
            writer.writerow(weight_analysis(name, param))


    

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, torch.nn.functional as F

tok  = AutoTokenizer.from_pretrained("gpt2")
gpt2 = AutoModelForCausalLM.from_pretrained("gpt2").to("cpu").eval()
txt  = "This is a quick sanity-check sentence."
ids  = tok(txt, return_tensors="pt")
logits = gpt2(**ids).logits[:, :-1]          # next-token dists
probs  = torch.softmax(logits, dim=-1)
entropy = (-probs * torch.log2(probs)).sum(-1).mean().item()
print("Entropy bits =", entropy)

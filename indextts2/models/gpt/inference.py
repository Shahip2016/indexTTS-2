import torch
import torch.nn.functional as F

class GPTInference:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device

    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50, top_p=0.9):
        self.model.eval()
        generated = input_ids
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(generated)
                next_token_logits = outputs[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Apply top-p filtering
                if top_p > 0.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)
                
                if next_token.item() == self.model.config.eos_token_id: # Assuming eos_token_id is available
                    break
                    
        return generated

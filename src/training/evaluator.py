import torch
import torch.nn.functional as F

class Evaluator:
    def __init__(self, model, dataloader, criterion, device):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device

    def evaluate(self, max_new_tokens):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        n_samples = len(self.dataloader.dataset)

        with torch.no_grad():
            for batch in self.dataloader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                if self.model.architecture == 'encoder_only':
                    logits = self.model(src=inputs) # logits are only used to compute loss
                    preds = logits.argmax(dim=-1)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                elif self.model.architecture == 'encoder_decoder':
                    logits = self.model(src=inputs, tgt=targets[:, :-1])
                    tgt_ids = targets[:, :-max_new_tokens]
                    preds = self.model.generate(src=inputs, tgt=tgt_ids, max_new_tokens=max_new_tokens)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), targets[:, 1:].contiguous().view(-1)) # a shifted loss
                else: # decoder only
                    full_sequence = torch.cat([inputs, targets], dim=1)
                    preds = self.model.generate(src=None, tgt=full_sequence[:, :-max_new_tokens], max_new_tokens=max_new_tokens)
                    logits = self.model(tgt=full_sequence[:, :-1])
                    input_len = inputs.size(1)
                    pred_targets = logits[:, input_len-1:]  # Start from the first target toke
                    loss = self.criterion(
                        pred_targets.contiguous().view(-1, pred_targets.size(-1)),
                        targets.view(-1)
                    )

                total_loss += loss.item()

                preds = preds[:, -max_new_tokens:]
                total_correct += (preds == targets[:, -max_new_tokens:]).all(dim=1).sum().item()

        avg_loss = total_loss / len(self.dataloader)
        accuracy = total_correct / n_samples
        return avg_loss, accuracy
import torch
import torch.nn.functional as F

class Evaluator:
    '''Evaluate a transformer model on a dataloader.

    Supports encoder-only, decoder-only, and encoder-decoder architectures.
    Computes average loss and sequence-level accuracy for the last generated steps.
    '''
    def __init__(self, model, dataloader, criterion, device):
        '''Create an Evaluator.

        Args:
            model: A model with `.architecture` and a forward compatible with the code paths.
            dataloader: Yields `(inputs, targets)` batches.
            criterion: Loss function taking `(logits, targets)`.
            device: Torch device to move batches to.
        '''
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device

    def evaluate(self, max_new_tokens):
        '''Evaluate on the full dataloader.

        Args:
            max_new_tokens: Number of final target tokens to score for accuracy.

        Returns:
            tuple[float, float]: `(avg_loss, accuracy)` where accuracy is sequence-level
            exact match over the last `max_new_tokens` tokens.
        '''
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

    def evaluate_by_steps(self, max_new_tokens, eval_steps = list(range(1,10)), n_batches=16):
        '''Evaluate accuracy as a function of how many steps are matched.
        Serves a specific focus for model analysis: how well does the model do on the first generated token, 
        the first 2 tokens, etc.

        For each `step` in `eval_steps`, counts a sample correct if the first `step`
        predicted tokens (within the last `max_new_tokens` window) exactly match.

        Args:
            max_new_tokens: Number of final target tokens to generate/compare.
            eval_steps: Iterable of step counts to evaluate (e.g. 1..9).
            n_batches: If not None, limit evaluation to the first `n_batches` batches.

        Returns:
            tuple[dict[int, float], float]: `(accuracies, avg_loss)` where accuracies
            maps each step to its exact-match accuracy.
        '''
        self.model.eval()
        total_loss = 0
        total_samples = 0
        total_correct_by_steps = {step : 0 for step in eval_steps}

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                if n_batches is not None and batch_idx >= n_batches:
                    break
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
                total_samples += inputs.size(0)

                preds = preds[:, -max_new_tokens:]
                for step in eval_steps:
                    total_correct_by_steps[step] += (preds[:, :step] == targets[:, -max_new_tokens:][:, :step]).all(dim=1).sum().item()

        avg_loss = total_loss / len(self.dataloader)
        accuracies = {step: total_correct_by_steps[step] / total_samples for step in eval_steps}
        return accuracies, avg_loss
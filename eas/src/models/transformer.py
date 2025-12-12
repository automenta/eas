"""
Pre-trained Transformer Wrapper for EAS
Wraps Hugging Face transformers to expose the EAS hook interface
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Callable

class PretrainedTransformer(nn.Module):
    """
    Wrapper for Hugging Face Causal LM models.
    Exposes intervention hooks similar to the toy AutoregressiveTransformer.
    """
    def __init__(self, model_name: str = "EleutherAI/pythia-70m", device: str = "cpu"):
        super().__init__()
        self.device = device
        self.model_name = model_name
        
        print(f"Loading pre-trained model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval() # We usually keep the base model frozen/eval mode

        # Hook storage
        self.layer_activations = {}
        self.intervention_hooks = {}
        self.hook_handles = []

        # Determine model structure and hidden dimension
        self.d_model = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers

        # We target a middle layer for intervention by default.
        # For Pythia-70m (6 layers), layer 3 is middle.
        self.middle_layer_idx = self.num_layers // 2

        self._register_hooks()

    def _register_hooks(self):
        """
        Registers forward hooks on the transformer layers.
        We need to identify the specific module list that contains the layers.
        Common names: `gpt_neox.layers`, `model.layers`, `h`, `transformer.h`
        """
        # Attempt to find the layer list
        if hasattr(self.model, "gpt_neox"):
            self.layers = self.model.gpt_neox.layers
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"): # Llama-like
            self.layers = self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"): # GPT-2 like
            self.layers = self.model.transformer.h
        elif hasattr(self.model, "bert") and hasattr(self.model.bert, "encoder"): # BERT
            self.layers = self.model.bert.encoder.layer
        else:
            # Fallback: try to find a ModuleList by inspecting
            for name, module in self.model.named_modules():
                if isinstance(module, nn.ModuleList) and len(module) > 0:
                    self.layers = module
                    break
            else:
                raise ValueError(f"Could not find layer list in model {self.model_name}")

        for i, layer in enumerate(self.layers):
            handle = layer.register_forward_hook(self._make_layer_hook(i))
            self.hook_handles.append(handle)

    def _make_layer_hook(self, layer_idx):
        """Make a hook function to capture activations at a specific layer"""
        def hook(module, input, output):
            # Input to the layer is usually a tuple (hidden_states, ...)
            # We want to intercept hidden_states.
            hidden_states = input[0]

            # Store activation (clone to be safe)
            # We might want to store it detached if we are not backpropping,
            # but for intervention we need to modify it in-place or return modified.

            # Note: For intervention to work, we must return the modified output
            # But here we are hooking the layer *input*.
            # Wait, `register_forward_hook` runs *after* the forward pass of the module.
            # If we want to modify the input to the layer, we should use `register_forward_pre_hook`.
            # OR we can modify the output of the *previous* layer.

            # The toy model used `register_forward_hook` on the layer and modified `input[0]`.
            # But modifying `input` in a forward hook (post-forward) doesn't affect the computation of that layer
            # because it has already happened!

            # Checking toy model code:
            # def hook(module, input, output):
            #     self.layer_activations[layer_idx] = input[0].detach().clone()
            #     if layer_idx in self.intervention_hooks:
            #          return intervention_func(input[0])

            # In PyTorch, if a forward hook returns a value, it replaces the *output* of that layer.
            # But the toy model hook was using `input[0]`. This is weird.
            # If the toy hook returns something, it replaces the OUTPUT of the layer.
            # But the toy hook was calculating intervention based on INPUT.
            # So `layer(x)` returns `intervention(x)`.
            # This means the layer's actual computation is skipped/replaced by the intervention?
            # No, the layer computation happened, `output` was produced, but then ignored because the hook returned something else.
            # AND the replacement was `intervention(input)`.
            # So effectively, the layer was replaced by the intervention function?

            # Let's look at the toy model again.
            # `x = layer(x, attention_mask)` in `AutoregressiveTransformer.forward`.
            # If the hook returns `intervention(input)`, then `x` becomes that for the next layer.
            # So the layer transformation itself is BYPASSED if the hook returns.
            # That seems like a bug or a very specific design in the toy model.
            # EAS description says: "Exposes a read/write hook at the middle layer (Layer 1) to allow the Watcher to intercept and modify the hidden state tensor H."
            # "The hook captures the output of Layer 1 before it's passed to Layer 2, processes it through the Watcher intervention, then injects the modified version as input to Layer 2."

            # If we hook Layer 1, and return modified activation, that becomes the input to Layer 2.
            # So hooking the *output* of Layer 1 is the correct way.

            # In the toy model:
            # `self.layer_activations[layer_idx] = input[0].detach().clone()` -> Captures input to layer i.
            # `return intervention_func(input[0])` -> Returns modified input as the output of layer i.
            # This effectively makes Layer i an identity function (plus intervention) and skips the actual TransformerBlock logic for Layer i.
            # This seems wrong if we want the layer to do its job.

            # Correct approach for Pretrained Model:
            # We want to intervene on the *output* of Layer i, which becomes the input to Layer i+1.
            # So we look at `output` (which is a tuple, usually (hidden_states, ...)).

            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Store it
            self.layer_activations[layer_idx] = hidden_states.detach()

            # Apply intervention
            if layer_idx in self.intervention_hooks:
                intervention_func = self.intervention_hooks[layer_idx]
                modified_hidden = intervention_func(hidden_states)

                # We need to return the full output tuple structure
                if isinstance(output, tuple):
                    return (modified_hidden,) + output[1:]
                else:
                    return modified_hidden

        return hook

    def register_intervention_hook(self, layer_idx, intervention_func):
        """Register an intervention function for a specific layer"""
        self.intervention_hooks[layer_idx] = intervention_func

    def remove_intervention_hook(self, layer_idx):
        """Remove an intervention hook from a specific layer"""
        if layer_idx in self.intervention_hooks:
            del self.intervention_hooks[layer_idx]

    def get_layer_activation(self, layer_idx):
        """Get the activation from a specific layer"""
        return self.layer_activations.get(layer_idx)

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask=attention_mask).logits

    def generate(self, input_ids, max_new_tokens=10, **kwargs):
        return self.model.generate(input_ids, max_new_tokens=max_new_tokens, **kwargs)

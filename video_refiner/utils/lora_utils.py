import torch
import peft


def configure_lora_for_model(transformer, lora_config, is_main_process=True):
    """Configure LoRA for a transformer model.

    Targets all nn.Linear layers inside the specified block classes.
    Compatible with checkpoints from LongLive/OPSD-V (CausalWanAttentionBlock)
    and Causal-Forcing-VSR (WanAttentionBlock) since the internal Linear
    layer paths (self_attn.q/k/v/o, cross_attn.q/k/v/o, ffn.0/2) are identical.

    Args:
        transformer: The transformer model to apply LoRA to
        lora_config: LoRA configuration (OmegaConf or dict) with keys:
            type, rank, alpha, dropout, target_block_classes (optional)
        is_main_process: Whether this is the main process (for logging)

    Returns:
        lora_model: The LoRA-wrapped model
    """
    target_block_classes = list(lora_config.get(
        "target_block_classes", ["WanAttentionBlock", "CausalWanAttentionBlock"]))

    target_linear_modules = set()
    for name, module in transformer.named_modules():
        if module.__class__.__name__ in target_block_classes:
            for full_submodule_name, submodule in module.named_modules(prefix=name):
                if isinstance(submodule, torch.nn.Linear):
                    target_linear_modules.add(full_submodule_name)

    target_linear_modules = sorted(target_linear_modules)

    if not target_linear_modules:
        raise ValueError(
            f"No Linear layers found for LoRA; "
            f"target_block_classes={target_block_classes}")

    if is_main_process:
        print(f"LoRA target modules: {len(target_linear_modules)} Linear layers "
              f"(block classes: {target_block_classes})")

    adapter_type = lora_config.get("type", "lora")
    if adapter_type == "lora":
        rank = lora_config.get("rank", 16)
        peft_config = peft.LoraConfig(
            r=rank,
            lora_alpha=lora_config.get("alpha", None) or rank,
            lora_dropout=lora_config.get("dropout", 0.0),
            target_modules=target_linear_modules,
        )
    else:
        raise NotImplementedError(f"Adapter type {adapter_type} is not implemented")

    lora_model = peft.get_peft_model(transformer, peft_config)

    if is_main_process:
        lora_model.print_trainable_parameters()

    return lora_model

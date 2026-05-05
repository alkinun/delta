from unsloth import FastVisionModel
from unsloth.chat_templates import train_on_responses_only

model, processor = FastVisionModel.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B",
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
)

#################################################################################################

EPOCHS = 2

model = FastVisionModel.get_peft_model(
    model,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    r = 32,
    lora_alpha = 64,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

#################################################################################################

from datasets import load_dataset
import random
dataset = load_dataset("json", data_files="finetune.json")
random.shuffle(dataset)

#################################################################################################

from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model = model,
    tokenizer = processor,
    train_dataset = dataset["train"],
    dataset_text_field = "text",
    max_seq_length = 16384,
    args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_ratio = 0.1,
        num_train_epochs = EPOCHS,
        learning_rate = 2e-4,
        logging_steps = 1,
        save_strategy = "no",
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        report_to = "wandb",
    )
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "[INSTRUCTION]\n",
    response_part = "[OUTPUT]\n",
)

trainer_stats = trainer.train()

#################################################################################################

model.save_pretrained_merged("model_7b_7.9k", processor)

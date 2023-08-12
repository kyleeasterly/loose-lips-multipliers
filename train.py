'''
DEF CON 31 AI Village - LLMs: Loose Lips Multipliers
Kyle Easterly & Mitch Kitter
https://github.com/kyleeasterly/loose-lips-multipliers
train.py: LoRA training code
'''

import argparse
from datasets import load_dataset
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
import torch
import transformers
from transformers import BitsAndBytesConfig, LlamaForCausalLM, LlamaTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='openlm-research/open_llama_7b')
parser.add_argument('--dataset_path', type=str, default='./data/demo.json')
parser.add_argument('--output_dir', type=str, default='./loras/open_llama_7b_demo')
parser.add_argument('--bnb_load_in_4bit', type=bool, default=True)
parser.add_argument('--bnb_4bit_use_double_quant', type=bool, default=True)
parser.add_argument('--bnb_4bit_quant_type', type=str, default='nf4')
parser.add_argument('--lora_r', type=int, default=32)
parser.add_argument('--lora_alpha', type=int, default=64)
parser.add_argument('--lora_target_modules', type=str, default="q_proj,k_proj,v_proj")
parser.add_argument('--lora_dropout', type=float, default=0.05)
parser.add_argument('--lora_bias', type=str, default="none")
parser.add_argument('--lora_task_type', type=str, default="CAUSAL_LM")
parser.add_argument('--test_size', type=float, default=0.2)
parser.add_argument('--per_gpu_train_batch_size', type=int, default=1)
parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
parser.add_argument('--warmup_ratio', type=float, default=0.1)
parser.add_argument('--num_train_epochs', type=int, default=5)
parser.add_argument('--learning_rate', type=float, default=2e-4)
parser.add_argument('--fp16', type=bool, default=True)
parser.add_argument('--save_steps', type=float, default=0.1)
parser.add_argument('--optimizer', type=str, default="paged_adamw_8bit")
parser.add_argument('--report_to', type=str, default="none") # could be "wandb"
args = parser.parse_args()

args.lora_target_modules = args.lora_target_modules.split(',')

bnb_config = BitsAndBytesConfig(
    load_in_4bit=args.bnb_load_in_4bit,
    bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
    bnb_4bit_quant_type=args.bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Loading base model")

tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
model = LlamaForCausalLM.from_pretrained(args.model_path, quantization_config=bnb_config, device_map='auto')
model.config.use_cache = False

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=args.lora_target_modules,
    lora_dropout=args.lora_dropout,
    bias=args.lora_bias,
    task_type=args.lora_task_type
)

model = get_peft_model(model, config)

print("Loading dataset")

data = load_dataset('json', data_files=args.dataset_path)
data = data['train'].train_test_split(test_size=args.test_size, seed=1337)
data = data.map(lambda data_point: tokenizer(data_point['text'], max_length=tokenizer.model_max_length, truncation=True))
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

trainer = transformers.Trainer(
    model = model,  
    train_dataset = data['train'],
    eval_dataset = data['test'],
    args = transformers.TrainingArguments(
        per_device_train_batch_size = args.per_gpu_train_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        warmup_ratio = args.warmup_ratio,
        num_train_epochs = args.num_train_epochs,
        learning_rate = args.learning_rate,
        fp16 = args.fp16,
        logging_steps = 1,
        output_dir = args.output_dir,
        save_strategy = "steps",
        save_steps = args.save_steps,
        optim = args.optimizer,
        report_to = args.report_to
    ),
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("Training")

trainer.train()
trainer.save_model()

print("Finished")

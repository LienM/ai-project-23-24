from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch

mps_device = torch.device("mps")


def fine_tune_gpt2(model_name, train_file, output_dir):
    # Load GPT-2 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(mps_device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Load training dataset
    train_dataset = TextDataset(
        tokenizer=tokenizer, file_path=train_file, block_size=128
    )
    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    # Set training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        save_steps=10_000,
    )
    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    trainer.train()
    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fine_tune_gpt2("gpt2", "prompts5.txt", "output5")

    # # load model from output directory
    # model = GPT2LMHeadModel.from_pretrained("output")
    # tokenizer = GPT2Tokenizer.from_pretrained("output")

    # input_text = "User: What should I call you?"
    # input_ids = tokenizer.encode(input_text, return_tensors="pt")
    # output = model.generate(input_ids, max_length=100, temperature=0)
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(generated_text)

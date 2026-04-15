
import argparse
import random
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning Script")
    parser.add_argument('--model', type=str, required=True,
                        help="Pretrained model name or path.")
    parser.add_argument('--dataset', type=str, required=True,
                        help="Dataset name or path.")
    parser.add_argument('--learning_rate', type=float, default=1e-6,
                        help="Learning rate for the optimizer")
    parser.add_argument('--epochs', type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size for training")
    parser.add_argument('--gradient_accumulation', type=int, default=9,
                        help="how many steps you accumulate to form a 'large batch'.")
    parser.add_argument('--save_path', type=str, help="path to save the model checkpoint")
    parser.add_argument('--max_length', type=int, default=512,
                        help="Maximum sequence length")

    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.train()
    dataset = load_dataset(args.dataset)["train"]
    dataset = dataset.shuffle(seed=42).select(range(4000))


    def tokenize_fn(example):

        # Tokenize the prompt (instructions) and response separately.
        instruction = example["instruction"]
        response = example["response"]

        # Use add_special_tokens=False so we can control token concatenation
        instr_tokens = tokenizer(instruction, truncation=True, max_length=args.max_length, add_special_tokens=False)
        resp_tokens = tokenizer(response, truncation=True, max_length=args.max_length, add_special_tokens=False)

        # Tokenize a separator (here we use "\n\n")
        sep_tokens = tokenizer("\n\n", add_special_tokens=False)["input_ids"]


        input_ids = instr_tokens["input_ids"] + sep_tokens + resp_tokens["input_ids"]


        attention_mask = [1] * len(input_ids)
        attention_mask=attention_mask[:args.max_length]
        labels = []
        for i in range(len(input_ids)):
            if i<len(instr_tokens["input_ids"]) + len(sep_tokens):
                labels.append(-100)
            else:
                labels.append(input_ids[i])
        labels=labels[:args.max_length]
        input_ids=input_ids[:args.max_length]
        extra=args.max_length-len(input_ids)
        if extra>0:
            input_ids+=[tokenizer.pad_token_id] * extra
            labels+=[-100] * extra
            attention_mask+=[0] * extra








        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    tokenized_dataset = dataset.map(tokenize_fn, batched=False)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    samples = random.sample(range(len(tokenized_dataset)), 5)

    for epoch in range(args.epochs):

        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        print("\nSample generations:")
        for sample_index in samples:
            sample = dataset[sample_index]
            prompt = sample["instruction"]
            print(f"\nPrompt: {prompt}\n\n")


            with torch.no_grad():
                sent = tokenizer(prompt, return_tensors="pt").to(device)
                out=model.generate(input_ids=sent["input_ids"],attention_mask=sent["attention_mask"])
                print(tokenizer.decode(out[0]))




        total_loss = 0.0
        for index, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)




            t=model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss=t.loss/args.gradient_accumulation
            loss.backward()
            total_loss += loss.item()


            if (index+1) % args.gradient_accumulation == 0:

                optimizer.step()
                optimizer.zero_grad()





        
        model.save_pretrained(args.save_path)
        tokenizer.save_pretrained(args.save_path)

    print("\nSample generations after training:")
    for sample_index in samples:
        sample = dataset[sample_index]
        prompt = sample["instruction"]
        print(f"\nPrompt: {prompt}")

        sent = tokenizer(prompt, return_tensors="pt").to(device)
        out=model.generate(input_ids=sent["input_ids"],attention_mask=sent["attention_mask"])
        print(tokenizer.decode(out[0]))



    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print("\nFine-tuning complete. Model saved to", args.save_path)

if __name__ == "__main__":
    main()


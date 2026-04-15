
import argparse
import random
import uuid
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning Script")
    parser.add_argument('--model', type=str, required=True,
                        help="Pretrained model name or path (e.g., 'gpt2' or a local path)")
    parser.add_argument('--dataset', type=str, required=True,
                        help="Dataset name or path (e.g., 'databricks/databricks-dolly-15k')")
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help="Learning rate for the optimizer")
    parser.add_argument('--epochs', type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size for training")
    parser.add_argument('--gradient_accumulation', type=int, default=8,
                        help="how many steps you accumulate to form a 'large batch'.")
    parser.add_argument('--save_path', type=str, help="path to save the model checkpoint")
    parser.add_argument('--max_length', type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument('--beta', type=float, default=0.01,
                        help="Beta parameter for DPO loss")

    args = parser.parse_args()


    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Ensure a padding token is defined (for models that don't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.train()

    # Load dataset (using the "train" split)
    dataset = load_dataset(args.dataset)["train"]

    # Define a tokenization function that masks out the loss on the prompt
    def tokenize_fn(example):
        # Tokenize the prompt (instructions) and response separately.

        instruction = example["prompt"]
        chosen_response = example["chosen_response"]
        rejected_response = example["rejected_response"]
        id = str(uuid.uuid4())

        # Use add_special_tokens=False so we can control token concatenation
        instr_tokens = tokenizer(instruction, truncation=True, max_length=args.max_length, add_special_tokens=False)
        chosen_resp_tokens = tokenizer(chosen_response, truncation=True, max_length=args.max_length, add_special_tokens=False)
        rejected_resp_tokens = tokenizer(rejected_response, truncation=True, max_length=args.max_length, add_special_tokens=False)

        # Tokenize a separator (here we use "\n\n")
        sep_tokens = tokenizer("\n\n", add_special_tokens=False)["input_ids"]

        
        chosen_input_ids = instr_tokens["input_ids"] + sep_tokens + chosen_resp_tokens["input_ids"]
        rejected_input_ids=instr_tokens["input_ids"] + sep_tokens + rejected_resp_tokens["input_ids"]
        # Create labels: mask out (with -100) the tokens corresponding to the instruction and separator, again you need to do this for both chosen and rejected

        chosen_attention_mask = [1] * len(chosen_input_ids)
        rejected_attention_mask = [1] * len(rejected_input_ids)
        chosen_attention_mask=chosen_attention_mask[:args.max_length]
        rejected_attention_mask=rejected_attention_mask[:args.max_length]
        chosen_labels = []
        for i in range(len(chosen_input_ids)):
            if i<len(instr_tokens["input_ids"]) + len(sep_tokens):
                chosen_labels.append(-100)
            else:
                chosen_labels.append(chosen_input_ids[i])
        chosen_labels=chosen_labels[:args.max_length]
        rejected_labels = []
        for i in range(len(rejected_input_ids)):
            if i<len(instr_tokens["input_ids"]) + len(sep_tokens):
                rejected_labels.append(-100)
            else:
                rejected_labels.append(rejected_input_ids[i])
        rejected_labels=rejected_labels[:args.max_length]

        # Then trunctate the inputs / pad the inputs according to args.max_length
        chosen_input_ids=chosen_input_ids[:args.max_length]
        chosen_extra=args.max_length-len(chosen_input_ids)
        rejected_input_ids=rejected_input_ids[:args.max_length]
        rejected_extra=args.max_length-len(rejected_input_ids)
        if chosen_extra>0:
            chosen_input_ids+=[tokenizer.pad_token_id] * chosen_extra
            chosen_labels+=[-100] * chosen_extra
            chosen_attention_mask+=[0] * chosen_extra
        if rejected_extra>0:
            rejected_input_ids+=[tokenizer.pad_token_id] * rejected_extra
            rejected_labels+=[-100] * rejected_extra
            rejected_attention_mask+=[0] * rejected_extra








        return {"id": id, "chosen_input_ids": chosen_input_ids, "chosen_attention_mask": chosen_attention_mask, "chosen_labels": chosen_labels,
                "rejected_input_ids": rejected_input_ids, "rejected_attention_mask": rejected_attention_mask, "rejected_labels": rejected_labels}

    tokenized_dataset = dataset.map(tokenize_fn, batched=False)
    tokenized_dataset.set_format(type='torch',
        columns=['id', 'chosen_input_ids', 'chosen_attention_mask', 'chosen_labels', 'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels'])

    samples = [random.randint(0, len(dataset) - 1) for _ in range(3)]

    dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    
    def compute_token_logprobs(logits, input_ids, labels):
      """Compute log probabilities for tokens where labels are not -100"""
      labels = labels.to(logits.device)
      logp = torch.nn.functional.log_softmax(logits, dim=-1)
      # Shift: logits[i] predicts token[i+1]
      shift_logits = logp[:, :-1, :]
      shift_labels = labels[:, 1:].clone()
      mask = shift_labels != -100
      shift_labels[~mask] = 0  # avoid index out of bounds
      token_logprobs = shift_logits.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
      token_logprobs = token_logprobs * mask
      batch_logprobs = token_logprobs.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
      return batch_logprobs
    id2logprobs = {}
    print("Calculating reference log probabilities...")

   
    model.eval()  # Set to eval mode for reference computation
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            print(f"Processing batch {index+1}/{len(dataloader)}", end="\r")

            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)

            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)

            # Get the logits for the chosen responses
            t = model(input_ids=chosen_input_ids,attention_mask=chosen_attention_mask)
            t = t.logits

            # Get the logits for the rejected responses
            t1 = model(input_ids=rejected_input_ids,attention_mask=rejected_attention_mask)
            t1 = t1.logits

            # Compute the log probabilities

            chosen_log_probs=compute_token_logprobs(t, chosen_input_ids, chosen_labels)
            rejected_log_probs = compute_token_logprobs(t1, rejected_input_ids, rejected_labels)  # shape: [batch]

            # Store the log probabilities in a dictionary
            for i in range(len(batch["id"])):
                id = batch["id"][i]
                id2logprobs[id] = {
                    "chosen_log_probs": chosen_log_probs[i].detach(),
                    "rejected_log_probs": rejected_log_probs[i].detach(),
                }

    print("\nReference log probabilities computed.")
    model.train()  # Set back to train mode

    # TODO: DPO Training loop
    beta = args.beta  # Beta parameter for DPO loss

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        print("\nSample generations:")
        for sample_idx in samples:
            sample = dataset[sample_idx]
            prompt = sample["prompt"]
            print(f"\nPrompt: {prompt}")
            # Tokenize the prompt (without response)
            # TODO: paste the code in section 3.1
            sent = tokenizer(prompt, return_tensors="pt").to(model.device)
            out=model.generate(input_ids=sent["input_ids"],attention_mask=sent["attention_mask"])
            print(tokenizer.decode(out[0]))

        total_loss = 0.0
        optimizer.zero_grad()  # Zero gradients at the beginning of epoch

        for index, batch in enumerate(dataloader):
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)

            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)


            # Get the reference log probabilities
            ids = batch["id"]
            w = [id2logprobs[i] for i in ids]

            yrc = torch.stack([x["chosen_log_probs"] for x in w]).to(device)
            yrr = torch.stack([x["rejected_log_probs"] for x in w]).to(device)



            # Compute the logits for the chosen responses
            c = model(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask).logits

            # Compute the logits for the rejected responses
            r = model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask).logits

            # Compute token log probabilities
            # Hint: you can call the helper function compute_token_logprobs
            cp=compute_token_logprobs(c, chosen_input_ids, chosen_labels)
            rp=compute_token_logprobs(r, rejected_input_ids, rejected_labels)


            # Compute DPO loss cacluation
            # The DPO loss: -log(σ(β(log_prob_difference(x_w) - log_prob_difference(x_l))))
            # Make sure to divide the loss by number of gradient accumulation steps

            diff_c = cp-rp
            diff_ref=yrc-yrr
            dpo=beta * (diff_c - diff_ref)
            l=-torch.nn.functional.logsigmoid(dpo)
            loss = l.mean() / args.gradient_accumulation
            loss.backward()
            total_loss += loss.item()



            # Gradient Accumulation
            if (index + 1) % args.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Your code ends here.

        # Handle any remaining gradients at the end of epoch
        if total_loss > 0:
            optimizer.step()
            optimizer.zero_grad()


        # Optional: Save checkpoint at the end of each epoch
        checkpoint_path = f"{args.save_path}_epoch{epoch+1}"
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)

    # Save the final fine-tuned model and tokenizer
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)

if __name__ == "__main__":
    main()

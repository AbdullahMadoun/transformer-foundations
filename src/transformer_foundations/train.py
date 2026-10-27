"""Deliverable: Write a script that runs a training loop to train your model on user-provided input.
In particular, we recommend that your training script allow for (at least) the following:
• Ability to configure and control the various model and optimizer hyperparameters.
• Memory-efficient loading of large training and validation datasets with np.memmap.
• Serializing checkpoints to a user-provided path.
• Periodically logging training and validation performance (e.g., to console and/or an external
service like Weights and Biases).9"""
import sys
import torch
import json
import time
import numpy as np 
import wandb
from transformer_foundations.transformer import Transformer

from transformer_foundations.train_scaffold import (
    AdamW, 
    cross_entropy_loss, 
    learning_rate_schedule, 
    gradient_clipping, 
    data_loading, 
    save_checkpoint, 

)

def main(args): 
    with open('config_light.json', 'r') as file:
        config = json.load(file)
    transformer_param = config["transformer"]
    loop = config["loop"]
    epochs = loop["epochs"]
    checkpoints = loop["checkpoints"]
    model_path = loop["model_path"]
    batch_size = loop["batch_size"]
    context_length= transformer_param["context_length"]
    device= transformer_param["device"]
    transformer = Transformer(**transformer_param)
    opt = AdamW(transformer.parameters(), **config["optimizer"])
    dataset = np.memmap("/Users/abdullah/CS336/assignment1-basics/data/tokens_valid.bin", dtype=np.uint16, mode='r')
    steps = len(dataset) // (batch_size * context_length)


    global_steps = 0
    start_time = time.time()
    wandb.init(project="cs336-assignment1", config=config)

    for i in range(epochs): 

        for step in range(steps): 

            lr = learning_rate_schedule(t=global_steps, **config["lr_schedule"])
            for p in opt.param_groups: 
                p["lr"] = lr
            inputs, targets =  data_loading(dataset, batch_size, context_length, device)

            pred = transformer(inputs)
            opt.zero_grad()
            loss = cross_entropy_loss(pred, targets)
            print(loss.item())
            loss.backward()
            gradient_clipping(transformer.parameters(), **config["clipping"])
            opt.step()
            global_steps+=1
            wandb.log({
                "loss": loss.item(), 
                "learning_rate": lr, 
                "epoch": i,
                "global_step": global_steps,
                "total_time" : time.time() - start_time
            })

        if i%checkpoints ==0: 
            save_checkpoint(transformer, opt, i, model_path)
if __name__ == "__main__":
    main(sys.argv)



        

        
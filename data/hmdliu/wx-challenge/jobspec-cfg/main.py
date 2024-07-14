import logging
import os
import time
import torch
import random

from config import parse_args
from data_helper import create_dataloaders
from model import MultiModal
from util import *


def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            loss, _, pred_label_id, label = model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)

    model.train()
    return loss, results


def train_and_validate(args):
    # 1. load data
    train_dataloader, val_dataloader = create_dataloaders(args)

    # 2. build model and optimizers
    model_class = MultiModal
    model = model_class(args)
    # logging.info(f"Model: {model}")
    optimizer, scheduler = build_optimizer(args, model)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))

    # 3. training
    step = 0
    patience = 0
    best_score = 0
    prev_score = 0
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs
    for epoch in range(args.max_epochs):
        for batch in train_dataloader:
            model.train()
            loss, accuracy, _, _ = model(batch)
            loss = loss.mean()
            accuracy = accuracy.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}")

        # 4. validation
        loss, results = validate(model, val_dataloader)
        results = {k: round(v, 4) for k, v in results.items()}
        logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")

        # 5. save checkpoint
        curr_score = results['mean_f1']
        if curr_score > max(best_score, args.export_bound):
            state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
            save_dict = {
                'args': args,
                'epoch': epoch,
                'mean_f1': curr_score,
                'model_class': model_class,
                'model_state_dict': state_dict,
            }
            torch.save(save_dict, f'{args.savedmodel_path}/model_epoch_{epoch}_mean_f1_{curr_score:.4f}.bin')
        if curr_score > best_score:
            best_score = curr_score
        
        # 6. early stopping
        patience = (patience + 1) if curr_score < prev_score else 0
        prev_score = curr_score
        if patience >= args.es_patience:
            logging.info(f"Epoch {epoch} step {step}: early stopping")
            break
    
    logging.info(f"Best Pred {best_score}")


def main():
    args = parse_args()

    # random hyperparameter search
    args.seed = random.randint(0, 2022)
    args.dropout = random.choice([0.1, 0.2, 0.3])
    # args.final_dropout = random.choice([0.0, 0.5])
    args.lr_scheduler = random.choice(['cosine', 'linear'])
    args.learning_rate = log_uniform(3e-5, 7e-5)
    print('=' * 15, 'Search Config', '=' * 15)
    print(f'seed: {args.seed}')
    print(f'dropout: {args.dropout}')
    print(f'final_dropout: {args.final_dropout}')
    print(f'warmup_steps: {args.warmup_steps}')
    print(f'lr_scheduler: {args.lr_scheduler}')
    print(f'learning_rate: {args.learning_rate}')
    print(f'save_dir: {args.savedmodel_path}')
    print('=' * 45, end='\n\n')
    
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    train_and_validate(args)


if __name__ == '__main__':
    main()

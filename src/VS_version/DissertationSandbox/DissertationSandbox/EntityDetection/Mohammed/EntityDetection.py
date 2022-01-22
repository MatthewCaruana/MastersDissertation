import sys

sys.path.insert(0, "Utilities")

from EntityDetectionModel import *
from SQDataset import *
from Evaluation import *
import argparse
import torch
import torch.nn as nn
import numpy as np
from torchtext import data
import random
import os
import time

def main(args):
    np.set_printoptions()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if not args.cuda:
        args.gpu = -1
    if torch.cuda.is_available() and args.cuda:
        print("Note: You are using GPU for training")
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.cuda:
        print("Warning: You have Cuda but not use it. You are using CPU for training.")

    torchText = data.Field(lower=True)
    torchEntityDetection = data.Field()

    train, dev, test = SQdataset.splits(torchText, torchEntityDetection, args.data_dir)
    torchText.build_vocab(train,dev,test)
    torchEntityDetection.build_vocab(train,dev,test)

    match_embedding = 0
    if os.path.isfile(args.vector_cache):
        stoi, vectors, dim = torch.load(args.vector_cache)
        torchText.vocab.vectors = torch.Tensor(len(torchText.vocab), dim)
        for i, token in enumerate(torchText.vocab.itos):
            wv_index = stoi.get(token, None)
            if wv_index is not None:
                torchText.vocab.vectors[i] = vectors[wv_index]
                match_embedding += 1
            else:
                torchText.vocab.vectors[i] = torch.FloatTensor(dim).uniform_(-0.25, 0.25)
    else:
        print("Error: Need word embedding pt file")
        exit(1)

    print("Embedding match number {} out of {}".format(match_embedding, len(torchText.vocab)))

    train_iter = data.Iterator(train, batch_size=args.batch_size, device=args.gpu, train=True, repeat=False,
                                       sort=False, shuffle=True, sort_within_batch=False)
    dev_iter = data.Iterator(dev, batch_size=args.batch_size, device=args.gpu, train=False, repeat=False,
                                       sort=False, shuffle=False, sort_within_batch=False)
    test_iter = data.Iterator(test, batch_size=args.batch_size, device=args.gpu, train=False, repeat=False,
                                       sort=False, shuffle=False, sort_within_batch=False)

    config = args
    config.words_num = len(torchText.vocab)

    if args.dataset == 'EntityDetection':
        config.label = len(torchEntityDetection.vocab)
        model = EDNeuralNet(config)
    else:
        print("Error Dataset")
        exit()

    model.embed.weight.data.copy_(torchText.vocab.vectors)
    if args.cuda:
        modle = model.to(torch.device("cuda:{}".format(args.gpu)))
        print("Shift model to GPU")

    print(config)
    print("VOCAB num",len(torchText.vocab))
    print("Train instance", len(train))
    print("Dev instance", len(dev))
    print("Test instance", len(test))
    print("Entity Type", len(torchEntityDetection.vocab))
    print(model)

    parameter = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameter, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.NLLLoss()

    early_stop = False
    best_dev_F = 0
    best_dev_P = 0
    best_dev_R = 0
    iterations = 0
    iters_not_improved = 0
    num_dev_in_epoch = (len(train) // args.batch_size // args.dev_every) + 1
    patience = args.patience * num_dev_in_epoch # for early stopping
    epoch = 0
    start = time.time()
    header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
    dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
    log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{},{}'.split(','))
    save_path = os.path.join(args.save_path, args.entity_detection_mode.lower())
    os.makedirs(save_path, exist_ok=True)
    print(header)

    if args.dataset == 'EntityDetection':
        index2tag = np.array(torchEntityDetection.vocab.itos)
    else:
        print("Wrong Dataset")
        exit(1)

    while True:
        if early_stop:
            print("Early Stopping. Epoch: {}, Best Dev F1: {}".format(epoch, best_dev_F))
            break
        epoch += 1
        train_iter.init_epoch()
        n_correct, n_total = 0, 0
        n_correct_ed, n_correct_ner , n_correct_rel = 0, 0, 0

        for batch_idx, batch in enumerate(train_iter):
            # Batch size : (Sentence Length, Batch_size)
            iterations += 1
            model.train(); optimizer.zero_grad()
            scores = model(batch)
            # Entity Detection
            if args.dataset == 'EntityDetection':
                n_correct += torch.sum((torch.sum((torch.max(scores, 1)[1].view(batch.ed.size()).data == batch.ed.data), dim=0) \
                          == batch.ed.size()[0])).item()
                loss = criterion(scores, batch.ed.view(-1, 1)[:, 0])
            else:
                print("Wrong Dataset")
                exit()

            n_total += batch.batch_size
            loss.backward()
            # clip the gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
            optimizer.step()

            # evaluate performance on validation set periodically
            if iterations % args.dev_every == 0:
                model.eval()
                dev_iter.init_epoch()
                n_dev_correct = 0
                n_dev_correct_rel = 0

                gold_list = []
                pred_list = []

                for dev_batch_idx, dev_batch in enumerate(dev_iter):
                    answer = model(dev_batch)
                    if args.dataset == 'EntityDetection':
                        n_dev_correct += ((torch.max(answer, 1)[1].view(dev_batch.ed.size()).data == dev_batch.ed.data).sum(dim=0) \
                                        == dev_batch.ed.size()[0]).sum()
                        index_tag = np.transpose(torch.max(answer, 1)[1].view(dev_batch.ed.size()).cpu().data.numpy())
                        gold_list.append(np.transpose(dev_batch.ed.cpu().data.numpy()))
                        pred_list.append(index_tag)
                    else:
                        print("Wrong Dataset")
                        exit()

                if args.dataset == 'EntityDetection':
                    P, R, F = evaluation(gold_list, pred_list, index2tag, type=False)
                    print("{} Precision: {:10.6f}% Recall: {:10.6f}% F1 Score: {:10.6f}%".format("Dev", 100. * P, 100. * R,
                                                                                             100. * F))
                else:
                    print("Wrong dataset")
                    exit()

                # update model
                if args.dataset == 'EntityDetection':
                    if F > best_dev_F:
                        best_dev_F = F
                        best_dev_P = P
                        best_dev_R = R
                        iters_not_improved = 0
                        snapshot_path = os.path.join(save_path, args.specify_prefix + '_best_model.pt')
                        # save model, delete previous 'best_snapshot' files
                        torch.save(model, snapshot_path)
                    else:
                        iters_not_improved += 1
                        if iters_not_improved > patience:
                            early_stop = True
                            break
                else:
                    print("Wrong dataset")
                    exit()

            if iterations % args.log_every == 1:
                # print progress message
                print(log_template.format(time.time() - start,
                                              epoch, iterations, 1 + batch_idx, len(train_iter),
                                              100. * (1 + batch_idx) / len(train_iter), loss.item(), ' ' * 8,
                                              100. * n_correct / n_total, ' ' * 12))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment the creation of the Entity Detection Neural Network")
    parser.add_argument('--entity_detection_mode', type=str, required=True, help='options are LSTM, GRU')
    parser.add_argument('--no_cuda', action='store_false', help='do not use cuda', dest='cuda')
    parser.add_argument('--gpu', type=int, default=0) # Use -1 for CPU
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, default="EntityDetection")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=3435)
    parser.add_argument('--dev_every', type=int, default=2000)
    parser.add_argument('--log_every', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--save_path', type=str, default='saved_checkpoints')
    parser.add_argument('--specify_prefix', type=str, default='id1')
    parser.add_argument('--words_dim', type=int, default=300)
    parser.add_argument('--num_layer', type=int, default=2)
    parser.add_argument('--rnn_fc_dropout', type=float, default=0.3)
    parser.add_argument('--input_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=300)
    parser.add_argument('--rnn_dropout', type=float, default=0.3)
    parser.add_argument('--clip_gradient', type=float, default=0.6, help='gradient clipping')
    parser.add_argument('--vector_cache', type=str, default="..\\..\\..\\..\\Data\\data\\sq_glove300d.pt")
    parser.add_argument('--weight_decay',type=float, default=0)
    parser.add_argument('--fix_embed', action='store_false', dest='train_embed')
    parser.add_argument('--hits', type=int, default=100)
    # added for testing
    parser.add_argument('--trained_model', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='..\\..\\..\\..\\Data\\data\\processed_simplequestions_dataset')
    parser.add_argument('--results_path', type=str, default='query_text')

    args = parser.parse_args()

    if (args.hits != []):
        print(args.hits)

    main(args)
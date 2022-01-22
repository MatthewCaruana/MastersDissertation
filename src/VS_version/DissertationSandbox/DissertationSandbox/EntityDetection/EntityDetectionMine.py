import sys

sys.path.insert(0, "Utilities")

from SQDataset import *
from EntityDetection import *

import torch
import torch.nn as nn
import argparse

def main(args):
    #Set settings
    np.set_printoptions(threshold=1)

    # Set random seed for reproducibility
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

    #Load data
    TEXT = data.Field(lower=True)
    ED = data.Field()

    train, dev, test = SQdataset.splits(TEXT, ED, args.data_dir)
    TEXT.build_vocab(train, dev, test)
    ED.build_vocab(train, dev, test)

    #partition data iterators
    train_iter = data.Iterator(train, batch_size=args.batch_size, device=args.gpu, train=True, repeat=False,
                                   sort=False, shuffle=True, sort_within_batch=False)
    dev_iter = data.Iterator(dev, batch_size=args.batch_size, device=args.gpu, train=False, repeat=False,
                                   sort=False, shuffle=False, sort_within_batch=False)
    test_iter = data.Iterator(test, batch_size=args.batch_size, device=args.gpu, train=False, repeat=False,
                                   sort=False, shuffle=False, sort_within_batch=False)

    #set up model
    config = args
    config.words_num = len(TEXT.vocab)

    model = EDNeuralNet(config)

    print(config)
    print("VOCAB num",len(TEXT.vocab))
    print("Train instance", len(train))
    print("Dev instance", len(dev))
    print("Test instance", len(test))
    print("Entity Type", len(ED.vocab))
    print(model)

    parameter = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameter, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.NLLLoss()

    #start while loop
    while True:
        if early_stop:
            print("Early Stopping. Epoch: {}, Best Dev F1: {}".format(epoch, best_dev_F))
            break
        epoch += 1
        train_iter.init_epoch()
        n_correct, n_total = 0, 0
        n_correct_ed, n_correct_ner, n_correct_rel = 0, 0, 0

        #for each training iteration
        for batch_idx, batch in enumerate(train_iter):
            #train model
            iterations += 1
            model.train()
            optimizer.zero_grad()
            scores = model(batch)

            n_correct += torch.sum((torch.sum((torch.max(scores, 1)[1].view(batch.ed.size()).data == batch.ed.data), dim=0) == batch.ed.size()[0])).item()
            loss = criterion(scores, batch.ed.view(-1, 1)[:, 0])

            #optimize model
            n_total += batch.batch_size
            loss.backward()
            # clip the gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
            optimizer.step()

            #every now and then check validation set
            if iterations % args.dev_every == 0:
                model.eval()
                dev_iter.init_epoch()
                n_dev_correct = 0
                n_dev_correct_rel = 0

                gold_list = []
                pred_list = []

                for dev_batch_idx, dev_batch in enumerate(dev_iter):
                    answer = model(dev_batch)
                    n_dev_correct += ((torch.max(answer, 1)[1].view(dev_batch.ed.size()).data == dev_batch.ed.data).sum(dim=0) == dev_batch.ed.size()[0]).sum()
                    index_tag = np.transpose(torch.max(answer, 1)[1].view(dev_batch.ed.size()).cpu().data.numpy())
                    gold_list.append(np.transpose(dev_batch.ed.cpu().data.numpy()))
                    pred_list.append(index_tag)

                P, R, F = evaluation(gold_list, pred_list, index2tag, type=False)
                print("{} Precision: {:10.6f}% Recall: {:10.6f}% F1 Score: {:10.6f}%".format("Dev", 100. * P, 100. * R, 100. * F))

                #if current f is better than best f then save the best f model
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
            if iterations % args.log_every == 1:
            # print progress message
            print(log_template.format(time.time() - start,
                                          epoch, iterations, 1 + batch_idx, len(train_iter),
                                          100. * (1 + batch_idx) / len(train_iter), loss.item(), ' ' * 8,
                                          100. * n_correct / n_total, ' ' * 12))



if __init__ == "__main__":
    parser = argparse.ArgumentParser(description="Transforming the Freebase data to our Triple structure")
    parser.add_argument('--data_dir', type=str, default='data\\QuestionAnswering\\processed_simplequestions_dataset')
    parser.add_argument('--model_location', type=str, default="EntityDetection\\models")
    parser.add_argument('--output', type=str, default="data\\QuestionAnswering\\SimpleQuestions_v2\\freebase-subsets\\")
    parser.add_argument('--type', type=str, default="FB5M")
    parser.add_argument('--no_cuda', action='store_false', help='do not use cuda', dest='cuda')
    parser.add_argument('--gpu', type=int, default=0) # Use -1 for CPU
    parser.add_argument('--batch_size', type=int, defauly=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=3435)
    parser.add_argument('--dev_every', type=int, default= 2000)
    parser.add_argument('--clip_gradient', type=float, default=0.6, help='gradient clipping')
    parser.add_argument('--log_every', type=int, default=1000)
    parser.add_argument('--specify_prefix', type=str, default='id1')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay',type=float, default=0)

    args = parser.parse_args()

    main(args)

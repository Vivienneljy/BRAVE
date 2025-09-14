import warnings

import pandas as pd
import torch
from tqdm import tqdm

from defending.defenses import attack_detection_strategy
from preprocessing.dataloader import local_loader, global_loader
from training.aggregate import aggregate
from training.update import update_weights, inference
from preprocessing.datasets import get_dataset
from preprocessing.models import get_model
from util.log_utils import Memorandum
from util.metrics import *
from util.options import args_parser
from util.param_utils import update_to_model


def main(args):
    # Acquire parameter
    if args.num_atk > args.num_users and args.attack_type != 'no':
        exit('Error:attacker number > participant number')
    if args.num_atk == 0 and args.attack_type != 'no':
        exit('Error:num_atk==0')

    # Set random seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.manual_seed_cpu)  # cpu
    torch.cuda.manual_seed(args.manual_seed_gpu)  # gpu

    # Set logs path
    memo = Memorandum(args)
    memo.print_param_detail()

    # Load preprocessing
    train_dataset, test_dataset, user_groups = get_dataset(args)
    local_train_dataloaders, local_test_dataloaders = local_loader(train_dataset,user_groups, args)
    benign_dataloader, malicious_dataloader = global_loader(test_dataset, args)

    # Load model
    start_round = 0
    global_model = get_model(args)
    device = 'cuda:0' if args.gpu else 'cpu'
    global_model.to(device)
    if args.first_time == 0:
        memo.print_checkpoint()
        checkpoint = torch.load(args.checkpoint)
        global_model.load_state_dict(checkpoint['state_dict'])
        start_round = checkpoint['iter'] + 1
    global_model.train()
    global_weights = global_model.state_dict()


    # Begin to communicate
    defense = None
    fnrs, fprs, accs, pres, scores, mtas, btas, asrs = {}, {}, {}, {}, {}, {}, {}, {}
    m = max(int(args.frac * args.num_users), 1)
    for iter in tqdm(range(start_round, args.iteration)):
        local_grads, local_losses = {}, {}
        global_model.train()
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        memo.print_iteration(iter,idxs_users)

        # Client training
        for i in range(m):
            trainloader = local_train_dataloaders[idxs_users[i]]
            local_grads[i], local_losses[i] = update_weights(global_model, trainloader, args, idxs_users[i] < args.num_atk)

        grad_list = [torch.concat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in local_grads.values()]

        # Client inference
        if args.save and (iter + 1) % args.print_every == 0:
            result_dict = {}
            for i in range(m):
                testloader = local_test_dataloaders[idxs_users[i]]
                result_dict[i] = inference(update_to_model(global_model, grad_list[i]), testloader, args)

            for i, idx in enumerate(idxs_users):
                acc, loss, _, _, asr = result_dict[i]
                memo.print_local_performance(acc, loss, asr, idx)

        # Attack Detection Strategy
        if args.defense in ['brave', 'flame', 'xmam', 'lfighter', 'dpfla']:
            defense = attack_detection_strategy(grad_list, args, device)
            mal, ben = defense.detection(memo, global_model)
            grad_list = [grad_list[i] for i in ben]
            if args.defense in ['flame']:
                grad_list = defense.adaptive_clipping(grad_list)

            if (iter + 1) % args.print_every == 0:
                scores[iter] = classify_perform(idxs_users, args.num_atk, mal)
                tpr, tnr, fprs[iter], fnrs[iter], accs[iter], pres[iter] = \
                    cluster_perform(idxs_users, args.num_atk, mal, scores[iter])
                memo.print_defense_performance(scores[iter], tpr, fprs[iter],tnr, fnrs[iter])

        global_weights = aggregate(grad_list, global_weights, args)
        if args.defense in ['flame']:
            global_weights = defense.adaptive_noising(global_weights, args.noise)
        global_model.load_state_dict(global_weights)

        # Print the performance of global model
        if (iter + 1) % args.print_every == 0:
            memo.save_model(global_model.state_dict())
            loss_avg = sum(local_losses.values()) / len(local_losses.values())
            mtas[iter], test_loss, actuals, predictions, asrs[iter] = inference(global_model, benign_dataloader, args)
            memo.print_global_performance_benign(mtas[iter], loss_avg, actuals, predictions, asrs[iter])
            if args.num_atk > 0:
                btas[iter], test_loss, actuals, predictions, _ = inference(global_model, malicious_dataloader, args)
                memo.print_global_performance_malicious(btas[iter], test_loss, actuals, predictions)

    # Record results
    file_name = './' + args.log
    if args.defense in ['brave', 'flame', 'xmam', 'lfighter', 'dpfla']:
        df = pd.DataFrame({'ctFNR': fnrs.values(), 'ctFPR': fprs.values(), 'ctAcc': accs.values(),
                           'ctPre': pres.values(), 'cfPre': scores.values(),
                           'Main_Task_Acc': mtas.values(), 'Poison_Task_Acc': btas.values(), 'ASR': asrs.values()})
    else:
        df = pd.DataFrame({'Main_Task_Acc': mtas.values(), 'Poison_Task_Acc': btas.values(), 'ASR': asrs.values()})
    df.to_csv(file_name + '/res.csv', index=False, header=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    warnings.simplefilter('ignore')
    args = args_parser()
    main(args)

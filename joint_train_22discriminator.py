import argparse
import logging
import math
import dill
import os
import options
from collections import OrderedDict

import torch
from torch import cuda
import random
import data
import utils
from meters import AverageMeter
from discriminator_professor import Discriminator as Discriminator_h
from discriminator import Discriminator as Discriminator_s
from generator_professor_forcing import LSTMModel
from PGLoss import PGLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# torch.backends.cudnn.enabled = False

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Adversarial-NMT.")

# Load args
options.add_general_args(parser)
options.add_dataset_args(parser)
options.add_distributed_training_args(parser)
options.add_optimization_args(parser)
options.add_checkpoint_args(parser)
options.add_generator_model_args(parser)
options.add_discriminator_model_args(parser)
options.add_generation_args(parser)


def main(args):
    use_cuda = (len(args.gpuid) >= 1)
    print("{0} GPU(s) are available".format(cuda.device_count()))

    # Load dataset
    splits = ['train', 'valid']
    if data.has_binary_files(args.data, splits):
        dataset = data.load_dataset(
            args.data, splits, args.src_lang, args.trg_lang, args.fixed_max_len)
    else:
        dataset = data.load_raw_text_dataset(
            args.data, splits, args.src_lang, args.trg_lang, args.fixed_max_len)
    if args.src_lang is None or args.trg_lang is None:
        # record inferred languages in args, so that it's saved in checkpoints
        args.src_lang, args.trg_lang = dataset.src, dataset.dst

    print('| [{}] dictionary: {} types'.format(dataset.src, len(dataset.src_dict)))
    print('| [{}] dictionary: {} types'.format(dataset.dst, len(dataset.dst_dict)))

    for split in splits:
        print('| {} {} {} examples'.format(args.data, split, len(dataset.splits[split])))

    g_logging_meters = OrderedDict()
    g_logging_meters['MLE_train_loss'] = AverageMeter()
    g_logging_meters['valid_loss'] = AverageMeter()
    g_logging_meters['PG_train_loss'] = AverageMeter()
    g_logging_meters['MLE_train_acc'] = AverageMeter()
    g_logging_meters['PG_train_acc'] = AverageMeter()
    g_logging_meters['valid_acc'] = AverageMeter()
    g_logging_meters['bsz'] = AverageMeter()  # sentences per batch

    d_logging_meters = OrderedDict()
    d_logging_meters['D_h_train_loss'] = AverageMeter()
    d_logging_meters['valid_loss'] = AverageMeter()
    d_logging_meters['D_s_train_loss'] = AverageMeter()
    d_logging_meters['D_h_train_acc'] = AverageMeter()
    d_logging_meters['D_s_train_acc'] = AverageMeter()
    d_logging_meters['valid_acc'] = AverageMeter()
    d_logging_meters['bsz'] = AverageMeter()  # sentences per batch

    # Set model parameters
    args.encoder_embed_dim = 1000
    args.encoder_layers = 2  # 4
    args.encoder_dropout_out = 0.1
    args.decoder_embed_dim = 1000
    args.decoder_layers = 2  # 4
    args.decoder_out_embed_dim = 1000
    args.decoder_dropout_out = 0.1
    args.bidirectional = False

    generator = LSTMModel(args, dataset.src_dict, dataset.dst_dict, use_cuda=use_cuda)
    print("Generator loaded successfully!")
    discriminator_h = Discriminator_h(args.decoder_embed_dim, args.discriminator_hidden_size, args.discriminator_linear_size, args.discriminator_lin_dropout, use_cuda=use_cuda)
    print("Discriminator_h loaded successfully!")
    discriminator_s = Discriminator_s(args, dataset.src_dict, dataset.dst_dict, use_cuda=use_cuda)
    print("Discriminator_s loaded successfully!")

    # Load generator model
    g_model_path = 'checkpoints/zhenwarm/genev.pt'
    assert os.path.exists(g_model_path)
    # generator = LSTMModel(args, dataset.src_dict, dataset.dst_dict, use_cuda=use_cuda)
    model_dict = generator.state_dict()
    model = torch.load(g_model_path)
    pretrained_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    generator.load_state_dict(model_dict)
    print("pre-trained Generator loaded successfully!")

    # Load discriminator model
    d_model_path = 'checkpoints/zhenwarm/discri.pt'
    assert os.path.exists(d_model_path)
    # generator = LSTMModel(args, dataset.src_dict, dataset.dst_dict, use_cuda=use_cuda)
    d_model_dict = discriminator_s.state_dict()
    d_model = torch.load(d_model_path)
    d_pretrained_dict = d_model.state_dict()
    # 1. filter out unnecessary keys
    d_pretrained_dict = {k: v for k, v in d_pretrained_dict.items() if k in d_model_dict}
    # 2. overwrite entries in the existing state dict
    d_model_dict.update(d_pretrained_dict)
    # 3. load the new state dict
    discriminator_s.load_state_dict(d_model_dict)
    print("pre-trained Discriminator loaded successfully!")

    # # Load discriminatorH model
    # d_H_model_path = 'checkpoints/joint_warm/DH.pt'
    # assert os.path.exists(d_H_model_path)
    # # generator = LSTMModel(args, dataset.src_dict, dataset.dst_dict, use_cuda=use_cuda)
    # d_H_model_dict = discriminator_h.state_dict()
    # d_H_model = torch.load(d_H_model_path)
    # d_H_pretrained_dict = d_H_model.state_dict()
    # # 1. filter out unnecessary keys
    # d_H_pretrained_dict = {k: v for k, v in d_H_pretrained_dict.items() if k in d_H_model_dict}
    # # 2. overwrite entries in the existing state dict
    # d_H_model_dict.update(d_H_pretrained_dict)
    # # 3. load the new state dict
    # discriminator_h.load_state_dict(d_H_model_dict)
    # print("pre-trained Discriminator_H loaded successfully!")


    def _calcualte_discriminator_loss(tf_scores, ar_scores):
        tf_loss = torch.log(tf_scores + 1e-9) * (-1)
        ar_loss = torch.log(1 - ar_scores + 1e-9) * (-1)
        return tf_loss + ar_loss

    if use_cuda:
        if torch.cuda.device_count() > 1:
            discriminator_h = torch.nn.DataParallel(discriminator_h).cuda()
            discriminator_s = torch.nn.DataParallel(discriminator_s).cuda()
            generator = torch.nn.DataParallel(generator).cuda()
        else:
            generator.cuda()
            discriminator_h.cuda()
            discriminator_s.cuda()
    else:
        discriminator_h.cpu()
        discriminator_s.cpu()
        generator.cpu()

    # adversarial training checkpoints saving path
    if not os.path.exists('checkpoints/realmyzhenup10shrink07drop01new'):
        os.makedirs('checkpoints/realmyzhenup10shrink07drop01new')
    checkpoints_path = 'checkpoints/realmyzhenup10shrink07drop01new/'

    # define loss function
    g_criterion = torch.nn.NLLLoss(ignore_index=dataset.dst_dict.pad(), reduction='sum')
    d_criterion = torch.nn.BCELoss()
    pg_criterion = PGLoss(ignore_index=dataset.dst_dict.pad(), size_average=True, reduce=True)

    # fix discriminator_s word embedding (as Wu et al. do)
    for p in discriminator_s.embed_src_tokens.parameters():
        p.requires_grad = False
    for p in discriminator_s.embed_trg_tokens.parameters():
        p.requires_grad = False

    # define optimizer
    g_optimizer = eval("torch.optim." + args.g_optimizer)(filter(lambda x: x.requires_grad,
                                                                 generator.parameters()),
                                                          args.g_learning_rate)

    d_optimizer_h = eval("torch.optim." + args.d_optimizer)(filter(lambda x: x.requires_grad,
                                                                 discriminator_h.parameters()),
                                                          args.d_learning_rate,
                                                          momentum=args.momentum,
                                                          nesterov=True)

    d_optimizer_s = eval("torch.optim." + args.d_optimizer)(filter(lambda x: x.requires_grad,
                                                                 discriminator_s.parameters()),
                                                          args.d_learning_rate,
                                                          momentum=args.momentum,
                                                          nesterov=True)

    # start joint training
    best_dev_loss = math.inf
    num_update = 0
    # main training loop
    for epoch_i in range(1, args.epochs + 1):
        logging.info("At {0}-th epoch.".format(epoch_i))

        seed = args.seed + epoch_i
        torch.manual_seed(seed)

        max_positions_train = (args.fixed_max_len, args.fixed_max_len)

        # Initialize dataloader, starting at batch_offset
        trainloader = dataset.train_dataloader(
            'train',
            max_tokens=args.max_tokens,
            max_sentences=args.joint_batch_size,
            max_positions=max_positions_train,
            # seed=seed,
            epoch=epoch_i,
            sample_without_replacement=args.sample_without_replacement,
            sort_by_source_size=(epoch_i <= args.curriculum),
            shard_id=args.distributed_rank,
            num_shards=args.distributed_world_size,
        )

        # reset meters
        for key, val in g_logging_meters.items():
            if val is not None:
                val.reset()
        for key, val in d_logging_meters.items():
            if val is not None:
                val.reset()

        # set training mode


        for i, sample in enumerate(trainloader):
            generator.train()
            discriminator_h.train()
            discriminator_s.train()
            update_learning_rate(num_update, 8e4, args.g_learning_rate, args.lr_shrink, g_optimizer)

            if use_cuda:
                # wrap input tensors in cuda tensors
                sample = utils.make_variable(sample, cuda=cuda)
            train_MLE = 0
            train_PG = 0

            if random.random()>0.5 and i!=0:
                train_MLE = 1
                # print("MLE Training")
                sys_out_batch_MLE, p_MLE, hidden_list_MLE = generator("MLE", epoch_i, sample)

                out_batch_MLE = sys_out_batch_MLE.contiguous().view(-1, sys_out_batch_MLE.size(-1))  # (64 X 50) X 6632

                train_trg_batch_MLE = sample['target'].view(-1)  # 64*50 = 3200
                loss_MLE = g_criterion(out_batch_MLE, train_trg_batch_MLE)

                sample_size_MLE = sample['target'].size(0) if args.sentence_avg else sample['ntokens']
                nsentences = sample['target'].size(0)
                logging_loss_MLE = loss_MLE.data / sample_size_MLE / math.log(2)
                g_logging_meters['bsz'].update(nsentences)
                g_logging_meters['MLE_train_loss'].update(logging_loss_MLE, sample_size_MLE)
                logging.debug(
                    f"G MLE loss at batch {i}: {g_logging_meters['MLE_train_loss'].avg:.3f}, lr={g_optimizer.param_groups[0]['lr']}")
                g_optimizer.zero_grad()
                loss_MLE.backward()
                # all-reduce grads and rescale by grad_denom
                for p in generator.parameters():
                    if p.requires_grad:
                        p.grad.data.div_(sample_size_MLE)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), args.clip_norm)
                g_optimizer.step()
            else:
                train_PG = 1
                ## part I: use gradient policy method to train the generator
                # print("Policy Gradient Training")
                sys_out_batch_PG, p_PG, hidden_list_PG = generator('PG', epoch_i, sample)  # 64 X 50 X 6632

                out_batch_PG = sys_out_batch_PG.contiguous().view(-1, sys_out_batch_PG.size(-1))  # (64 * 50) X 6632

                _, prediction = out_batch_PG.topk(1)
                prediction = prediction.squeeze(1)  # 64*50 = 3200
                prediction = torch.reshape(prediction, sample['net_input']['src_tokens'].shape)  # 64 X 50
                # if d_logging_meters['train_acc'].avg >= 0.75:
                with torch.no_grad():
                    reward = discriminator_s(sample['net_input']['src_tokens'], prediction)  # 64 X 1
                # else:
                #     reward = torch.ones(args.joint_batch_size)
                train_trg_batch_PG = sample['target']  # 64 x 50

                pg_loss_PG = pg_criterion(sys_out_batch_PG, train_trg_batch_PG, reward, use_cuda)
                sample_size_PG = sample['target'].size(0) if args.sentence_avg else sample['ntokens']  # 64
                logging_loss_PG = pg_loss_PG / math.log(2)
                g_logging_meters['PG_train_loss'].update(logging_loss_PG.item(), sample_size_PG)
                logging.debug(
                    f"G policy gradient loss at batch {i}: {pg_loss_PG.item():.3f}, lr={g_optimizer.param_groups[0]['lr']}")
                g_optimizer.zero_grad()
                if g_logging_meters['PG_train_loss'].val < 0.5:
                    pg_loss_PG.backward()
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), args.clip_norm)
                    g_optimizer.step()

            num_update += 1

            # if g_logging_meters["MLE_train_loss"].avg < 4:
            #  part II: train the discriminator
                # discriminator_h
            if num_update % 10 == 0:
                if g_logging_meters["PG_train_loss"].val < 2:
                    assert (train_MLE == 1) != (train_PG == 1)
                    if train_MLE == 1:
                        d_MLE = discriminator_h(hidden_list_MLE.detach())
                        M_loss = torch.log(d_MLE + 1e-9) * (-1)
                        h_d_loss = M_loss.sum()
                    elif train_PG == 1:
                        d_PG = discriminator_h(hidden_list_PG.detach())
                        P_loss = torch.log(1 - d_PG + 1e-9) * (-1)
                        h_d_loss = P_loss.sum()

                    # d_loss = _calcualte_discriminator_loss(d_MLE, d_PG).sum()
                    logging.debug(f"D_h training loss {h_d_loss} at batch {i}")

                    d_optimizer_h.zero_grad()

                    h_d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(discriminator_h.parameters(), args.clip_norm)
                    d_optimizer_h.step()


                #discriminator_s
                bsz = sample['target'].size(0)  # batch_size = 64

                src_sentence = sample['net_input']['src_tokens']  # 64 x max-len i.e 64 X 50

                # now train with machine translation output i.e generator output

                with torch.no_grad():
                    sys_out_batch, p, hidden_list = generator('PG', epoch_i, sample)  # 64 X 50 X 6632

                out_batch = sys_out_batch.contiguous().view(-1, sys_out_batch.size(-1))  # (64 X 50) X 6632

                _, prediction = out_batch.topk(1)
                prediction = prediction.squeeze(1)  # 64 * 50 = 6632

                fake_labels = torch.zeros(sample['target'].size(0)).float()  # 64 length vector
                # true_labels = torch.ones(sample['target'].size(0)).float()  # 64 length vector
                fake_sentence = torch.reshape(prediction, src_sentence.shape)  # 64 X 50

                if use_cuda:
                    fake_labels = fake_labels.cuda()
                    # true_labels = true_labels.cuda()

                # if random.random() > 0.5:
                fake_disc_out = discriminator_s(src_sentence, fake_sentence)  # 64 X 1
                fake_d_loss = d_criterion(fake_disc_out.squeeze(1), fake_labels)
                acc = torch.sum(torch.round(fake_disc_out).squeeze(1) == fake_labels).float() / len(fake_labels)
                d_loss = fake_d_loss
                # else:
                #
                #     true_sentence = sample['target'].view(-1)  # 64*50 = 3200
                #
                #     true_sentence = torch.reshape(true_sentence, src_sentence.shape)
                #     true_disc_out = discriminator_s(src_sentence, true_sentence)
                #     acc = torch.sum(torch.round(true_disc_out).squeeze(1) == true_labels).float() / len(true_labels)
                #     true_d_loss = d_criterion(true_disc_out.squeeze(1), true_labels)
                #     d_loss = true_d_loss


                # acc_fake = torch.sum(torch.round(fake_disc_out).squeeze(1) == fake_labels).float() / len(fake_labels)
                # acc_true = torch.sum(torch.round(true_disc_out).squeeze(1) == true_labels).float() / len(true_labels)
                # acc = (acc_fake + acc_true) / 2
                # acc = acc_fake
                # d_loss = fake_d_loss + true_d_loss
                s_d_loss = d_loss
                d_logging_meters['D_s_train_acc'].update(acc)

                d_logging_meters['D_s_train_loss'].update(s_d_loss)
                logging.debug(
                    f"D_s training loss {d_logging_meters['D_s_train_loss'].avg:.3f}, acc {d_logging_meters['D_s_train_acc'].avg:.3f} at batch {i}")
                d_optimizer_s.zero_grad()
                s_d_loss.backward()
                d_optimizer_s.step()

            if num_update % 10000 == 0:
                # validation
                # set validation mode
                print('validation and save+++++++++++++++++++++++++++++++++++++++++++++++')
                generator.eval()
                discriminator_h.eval()
                discriminator_s.eval()
                # Initialize dataloader
                max_positions_valid = (args.fixed_max_len, args.fixed_max_len)
                valloader = dataset.eval_dataloader(
                    'valid',
                    max_tokens=args.max_tokens,
                    max_sentences=args.joint_batch_size,
                    max_positions=max_positions_valid,
                    skip_invalid_size_inputs_valid_test=True,
                    descending=True,  # largest batch first to warm the caching allocator
                    shard_id=args.distributed_rank,
                    num_shards=args.distributed_world_size,
                )

                # reset meters
                for key, val in g_logging_meters.items():
                    if val is not None:
                        val.reset()
                for key, val in d_logging_meters.items():
                    if val is not None:
                        val.reset()

                for i, sample in enumerate(valloader):

                    with torch.no_grad():
                        if use_cuda:
                            # wrap input tensors in cuda tensors
                            sample = utils.make_variable(sample, cuda=cuda)

                        # generator validation
                        sys_out_batch_test, p_test, hidden_list_test = generator('test', epoch_i, sample)
                        out_batch_test = sys_out_batch_test.contiguous().view(-1, sys_out_batch_test.size(-1))  # (64 X 50) X 6632
                        dev_trg_batch = sample['target'].view(-1)  # 64*50 = 3200

                        loss_test = g_criterion(out_batch_test, dev_trg_batch)
                        sample_size_test = sample['target'].size(0) if args.sentence_avg else sample['ntokens']
                        loss_test = loss_test / sample_size_test / math.log(2)
                        g_logging_meters['valid_loss'].update(loss_test, sample_size_test)
                        logging.debug(f"G dev loss at batch {i}: {g_logging_meters['valid_loss'].avg:.3f}")

                torch.save(generator,
                           open(checkpoints_path + f"numupdate_{num_update/10000}w.sampling_{g_logging_meters['valid_loss'].avg:.3f}.pt",
                                'wb'), pickle_module=dill)

                # if g_logging_meters['valid_loss'].avg < best_dev_loss:
                #     best_dev_loss = g_logging_meters['valid_loss'].avg
                #     torch.save(generator, open(checkpoints_path + "best_gmodel.pt", 'wb'), pickle_module=dill)


def update_learning_rate(update_times, target_times, init_lr, lr_shrink, optimizer):
    lr = init_lr * (lr_shrink ** (update_times // target_times))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning(f"unknown arguments: {parser.parse_known_args()[1]}")
    main(options)

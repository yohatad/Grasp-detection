import cv2
import numpy as np
import tensorboardX
import torch
import torch.optim as optim
import torch.utils.data
from torchsummary import summary

import datetime
import logging
import os

from hardware.device import get_device
from inference.models import get_network
from inference.post_process import post_process_output
from tools.data import get_dataset
from tools.dataset_processing import evaluation

# Choosing Values
depth_use = 1
rgb_use = 1
dropout_use = 1 
dropout_probability = 0.5
input_size = 224
iou_threshold = 0.25

# Dataset
dataset_path = 'cornell' 
splitt = 0.9
rotate = 0
num_workers = 8 

# Paramters
batch_size = 8
epochs = 30
batches_per_epoch = 1000
optimz = 'SGD'

# Logging data
description = ''
logdir = 'logs/'
cpu = False
random_seed = 1


def validate(net, device, val_data, iou):
    """
    Run validation.
    Return: Successes, Failures and Losses
    """
    net.eval()

    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {

        }
    }

    ld = len(val_data)

    with torch.no_grad():
        for x, y, didx, rot, zoom_factor in val_data:
            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            lossd = net.compute_loss(xc, yc)

            loss = lossd['loss']

            results['loss'] += loss.item() / ld
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item() / ld

            acc, ang, width = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                        lossd['pred']['sin'], lossd['pred']['width'])

            suces = evaluation.calculate_iou_match(acc,ang,
                                               val_data.dataset.get_gtbb(didx, rot, zoom_factor),
                                               no_grasps = 1,
                                               grasp_width = width,
                                               threshold = iou
                                               )

            if suces:
                results['correct'] += 1
            else:
                results['failed'] += 1

    return results


def train(epoch, net, device, train_data, optimizer, batches_per_epoch):
    """
    Run one training epoch
    :return:  Average Losses for Epoch
    """
    results = {
        'loss': 0,
        'losses': {
        }
    }

    net.train()

    batch_idx = 0
    # Use batches per epoch to make training on different sized datasets more equivalent.
    while batch_idx <= batches_per_epoch:
        for x, y, _, _, _ in train_data:
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break

            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            lossd = net.compute_loss(xc, yc)

            loss = lossd['loss']

            if batch_idx % 10 == 0:
                logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))

            results['loss'] += loss.item()
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    results['loss'] /= batch_idx
    for l in results['losses']:
        results['losses'][l] /= batch_idx

    return results

def run():
    
    # Set-up output directories
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(description.split()))

    save_folder = os.path.join(logdir, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    tb = tensorboardX.SummaryWriter(save_folder)

    # Initialize logging
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        filename="{0}/{1}.log".format(save_folder, 'log'),
        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    # Get the compute device
    device = get_device(False)
    
    # Load Dataset
    logging.info('Loading cornell Dataset...')
    Dataset = get_dataset('cornell')
    dataset = Dataset(dataset_path,
                      output_size= input_size,
                      random_rotate=True,
                      random_zoom=True,
                      include_depth= depth_use,
                      include_rgb= rgb_use)
    logging.info('Dataset size is {}'.format(dataset.length))

    # Creating data indices for training and validation splits
    indices = list(range(dataset.length))
    split = int(np.floor( splitt * dataset.length))
    train_indices, val_indices = indices[:split], indices[split:]
    logging.info('Training size: {}'.format(len(train_indices)))
    logging.info('Validation size: {}'.format(len(val_indices)))

    # Creating data samplers and loaders
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    train_data = torch.utils.data.DataLoader(
            dataset,
            batch_size= batch_size,
            num_workers = num_workers,
            sampler = train_sampler
    )
    val_data = torch.utils.data.DataLoader(
        dataset,
        batch_size = 1,
        num_workers = num_workers,
        sampler = val_sampler
    )
    logging.info('Done')
# Load the network
    input_channels = 1 * depth_use + 3 * rgb_use
    network = get_network('model3')
    net = network(
        input_channels = input_channels,
        dropout = dropout_use,
        prob = dropout_probability,
        channel_size = 32 )

    net = net.to(device)
    logging.info('Done')

    if optimz.lower() == 'adam':
        optimizer = optim.Adam(net.parameters())
    elif optimz.lower() == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    best_iou = 0.0
    for epoch in range(epochs):
        logging.info('Epoch Number {:02d}'.format(epoch))
        train_results = train(epoch, net, device, train_data, optimizer, batches_per_epoch)

        # Logging training losses to tensorboard
        tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)

        # Run Validation
        logging.info('Validating...')
        test_results = validate(net, device, val_data,iou_threshold)
        logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct'] / (test_results['correct'] + test_results['failed'])))

        # Log validation results to tensorboard
        tb.add_scalar('Accuracy/IOU', test_results['correct'] / (test_results['correct'] + test_results['failed']), epoch)
        tb.add_scalar('loss/val_loss', test_results['loss'], epoch)
        for n, l in test_results['losses'].items():
            tb.add_scalar('val_loss/' + n, l, epoch)

        # Save best performing model
        iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
        if iou > best_iou or epoch == 0 or (epoch % 10) == 0:
            torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.2f' % (epoch, iou)))
            best_iou = iou

if __name__ == '__main__':
    run()
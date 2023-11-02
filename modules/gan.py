from torch.utils.data import DataLoader
import torch.optim as optim
import os
import torch
import time
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F

from custom_datasets import GanDataset
from models import  CriticTail, WordEncoder, WordDecoder, WordCritic


def train_word_gan(args, essential, large_data):
    print("Train Word GAN")
    
    pre_dataset = essential['train_set']
    dataset = GanDataset(args, essential, large_data, pre_dataset)
    val_dataset = GanDataset(args, essential, large_data, essential['val_set'], 'val')
    num_keywords = len(essential['keywords'])
    num_features = 128 # feature size
    num_category = len(essential['all_categories'])
    word_latent = args['word_latent']
    base_cuboid = large_data['base_cuboid']
    order_encoding = large_data['order_encoding']
    critic = WordCritic(num_keywords, num_category, num_features, word_latent).cuda()
    encoder = WordEncoder(num_keywords, num_category, num_features, order_encoding, base_cuboid, word_latent).cuda()
    decoder = WordDecoder(num_keywords, num_category, num_features, order_encoding, base_cuboid, word_latent).cuda()

    e_optimizer = optim.RMSprop(encoder.parameters(), lr=0.00005)
    d_optimizer = optim.RMSprop(decoder.parameters(), lr=0.0005)
    c_optimizer = optim.RMSprop(critic.parameters(), lr=0.00005)

    e_scheduler = optim.lr_scheduler.LambdaLR(optimizer=e_optimizer, lr_lambda=lambda epoch: 0.8 ** epoch)
    d_scheduler = optim.lr_scheduler.LambdaLR(optimizer=d_optimizer, lr_lambda=lambda epoch: 0.8 ** epoch)
    c_scheduler = optim.lr_scheduler.LambdaLR(optimizer=c_optimizer, lr_lambda=lambda epoch: 0.8 ** epoch)
    max_norm = 0.01
    critic_repeat = 3

    print("AAE Phase")
    for epoch in range(args['word_epoch']):
      dataloader = DataLoader(dataset, batch_size = 32, shuffle=True, num_workers=0)
      # val_dataloader = DataLoader(val_dataset, batch_size = 32, shuffle=True, num_workers=0)
      autoencoder_loss_sum = 0
      critic_loss_sum = 0
      encoder_loss_sum = 0
      total_batches = len(dataloader)
      for num_batch, batch in enumerate(dataloader):
        start = time.time()
        label, cuboids = batch
        label = label.cuda()
        cuboids = cuboids.cuda()
        local_batch_size = len(cuboids)

        ## autoencoder
        encoder.train()
        decoder.train()
        e_optimizer.zero_grad()
        d_optimizer.zero_grad()
        latent = encoder(cuboids, label)

        reconstructed = decoder(latent, label)
        line = torch.where(torch.norm(cuboids, p=2, dim=2)>0, 1, 0).view(-1, num_keywords, 1)
        reconstructed = reconstructed*line

        auto_loss = F.mse_loss(cuboids, reconstructed) # reconst loss: l2

        auto_loss.backward()
        e_optimizer.step()
        d_optimizer.step()
        autoencoder_loss_sum += auto_loss.item()

        ## descriminator
        encoder.eval()
        critic.train()
        for _ in range(critic_repeat):
          c_optimizer.zero_grad()
          latent = latent.detach()
          random_dist = torch.normal(0, 1, latent.shape, device='cuda')
          fake_score = torch.mean(critic(latent))
          true_score = torch.mean(critic(random_dist))
          critic_loss = (fake_score - true_score)/2 # critic loss
          critic_loss.backward()
          torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm)
          c_optimizer.step()
        critic_loss_sum += critic_loss.item()

        ## generator
        encoder.train()
        critic.eval()
        e_optimizer.zero_grad()
        latent = encoder(cuboids, label)
        fake_score = torch.mean(critic(latent))
        encoder_loss = -fake_score
        encoder_loss.backward()
        e_optimizer.step()
        encoder_loss_sum += encoder_loss.item()
        
        batch_time = int((time.time()-start)*1000)/1000
        print(f"\rAAE: Epoch {epoch}: Batch {num_batch}/{total_batches}, autoencoder loss={autoencoder_loss_sum/(num_batch+1)}, critic loss={critic_loss_sum/(num_batch+1)}, encoder loss={encoder_loss_sum/(num_batch+1)}, {batch_time}s/batch"+(" "*5), end='')
      print("")
      e_scheduler.step()
      d_scheduler.step()
      c_scheduler.step()
      
      
      # evaluate_word(encoder, decoder, critic, val_dataloader, epoch, order_encoding)


    print("GAN Phase")
    encoder_tail = CriticTail()
    encoder_tail.cuda()
    encoder_params = list(encoder.parameters()) + list(encoder_tail.parameters())
    decoder_params = decoder.parameters()
    e_optimizer = optim.RMSprop(encoder_params, lr=0.00005)
    d_optimizer = optim.RMSprop(decoder_params, lr=0.0001)
    e_scheduler = optim.lr_scheduler.LambdaLR(optimizer=e_optimizer, lr_lambda=lambda epoch: 0.8 ** epoch)
    d_scheduler = optim.lr_scheduler.LambdaLR(optimizer=d_optimizer, lr_lambda=lambda epoch: 0.8 ** epoch)
    max_norm = 0.01
    critic_repeat = 1
    
    for epoch in range(args['gan_epoch']):
      dataloader = DataLoader(dataset, batch_size = 32, shuffle=True, num_workers=0)
      val_dataloader = DataLoader(val_dataset, batch_size = 32, shuffle=True, num_workers=0) 
      generator_loss_sum = 0
      total_batches = len(dataloader)
      for _ in range(critic_repeat):
        critic_loss_sum = 0
        for num_batch, batch in enumerate(dataloader):
          start = time.time()
          label, cuboids = batch
          label = label.cuda()
          cuboids = cuboids.cuda()
          local_batch_size = len(cuboids)
          line = torch.where(torch.norm(cuboids, p=2, dim=2)>0, 1, 0).view(-1, num_keywords, 1)

          ## discriminator(encoder)
          encoder.train()
          encoder_tail.train()
          decoder.eval()

          e_optimizer.zero_grad()
          random_dist = torch.normal(0, 1, (local_batch_size, args['num_keywords'], 128), device='cuda')
          reconstructed = decoder(random_dist, label) * line
          original_noised = False
          # if random.random() < 0.5: ## random gaussian noise injection
          #   cuboids += torch.normal(0, 2, cuboids.shape, device='cuda') * line
          #   original_noised = True

          fake_score = torch.mean(encoder_tail(encoder(reconstructed, label, False)))
          true_score = torch.mean(encoder_tail(encoder(cuboids, label, not original_noised)))
          critic_loss = (fake_score - true_score)/2
          critic_loss.backward()
          torch.nn.utils.clip_grad_norm_(encoder_params, max_norm)
          e_optimizer.step()
          critic_loss_sum += critic_loss.item()
          batch_time = int((time.time()-start)*1000)/1000
          print(f"\rGAN: Epoch {epoch}: Batch {num_batch}/{total_batches} critic loss={critic_loss_sum/(num_batch+1)}, generator loss={generator_loss_sum/total_batches}, {batch_time}s/batch"+(" "*5), end='')

      for num_batch, batch in enumerate(dataloader):
        start = time.time()
        label, cuboids = batch
        label = label.cuda()
        cuboids = cuboids.cuda()
        local_batch_size = len(cuboids)
        line = torch.where(torch.norm(cuboids, p=2, dim=2)>0, 1, 0).view(-1, num_keywords, 1)
        
        ## generator(decoder)
        encoder.eval()
        encoder_tail.eval()
        decoder.train()

        d_optimizer.zero_grad()
        random_dist = torch.normal(0, 1, (local_batch_size, args['num_keywords'], 128), device='cuda')
        reconstructed = decoder(random_dist, label) * line

        fake_score = torch.mean(encoder_tail(encoder(reconstructed, label, False)))
        generator_loss = -fake_score
        generator_loss.backward()
        d_optimizer.step()
        generator_loss_sum += generator_loss.item()
        
        batch_time = int((time.time()-start)*1000)/1000
        print(f"\rGAN: Epoch {epoch}: Batch {num_batch}/{total_batches} critic loss={critic_loss_sum/total_batches}, generator loss={generator_loss_sum/(num_batch+1)}, {batch_time}s/batch"+(" "*5), end='')
      print("")
      e_scheduler.step()
      d_scheduler.step()
      c_scheduler.step()
    torch.save(decoder.state_dict(), './word_decoder.pt')
    return 


def evaluate_word(encoder, decoder, critic, dataloader, epoch, order_encoding, samples = True):
  encoder.eval()
  decoder.eval()
  critic.eval()
  real_correct = 0
  real_total = 0
  fake_correct = 0
  fake_total = 0
  total_batches = len(dataloader)
  with torch.no_grad():
    for num_batch, batch in enumerate(dataloader):
      start = time.time()
      label, cuboids = batch
      label = label.cuda()
      cuboids = cuboids.cuda()
      # line = torch.where(torch.norm(cuboids, p=2, dim=2)>0, 1.0, .0)

      latent = encoder(cuboids, label)
      random_dist = torch.normal(0, 1, latent.shape, device='cuda')
      fake_reality = critic(latent)
      original_reality = critic(random_dist)


      real_correct+=sum(original_reality>0)
      real_total+=len(original_reality)
      fake_correct+=sum(fake_reality<0)
      fake_total+=len(fake_reality)
      real_acc = int(real_correct/real_total * 100)
      fake_acc = int(fake_correct/fake_total * 100)
      batch_time = int((time.time()-start)*1000)/1000

      reconstructed = decoder(latent, label)
      line = torch.where(torch.norm(cuboids, p=2, dim=2)>0, 1, 0).view(-1, args['num_keywords'], 1)
      reconstructed = reconstructed*line

      auto_loss = F.mse_loss(cuboids, reconstructed) # reconst loss: l2

      print(f"\r  Evaluate: Batch {num_batch}/{total_batches}, accuracy=({real_acc}, {fake_acc}), autoencoder_los = {auto_loss}, {batch_time}s/batch"+(" "*5), end='')
      if num_batch == 1: break
    print("")

  
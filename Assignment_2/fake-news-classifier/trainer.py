import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def trainer(model, train_dataloader, val_dataloader, num_epochs, path_to_save='/mnt/c/Users/Administrator/Desktop/playground/nnfl',
          checkpoint_path='/mnt/c/Users/Administrator/Desktop/playground/nnfl',
          checkpoint=100, train_batch=1, test_batch=1, device='cuda:0'): # 2 Marks. 
      """
      Everything by default gets shifted to the GPU. Select the device according to your system configuration
      If you do no have a GPU, change the device parameter to "device='cpu'"
      :param model: the Classification model..
      :param train_dataloader: train_dataloader
      :param val_dataloader: val_Dataloader
      :param num_epochs: num_epochs
      :param path_to_save: path to save model
      :param checkpoint_path: checkpointing path
      :param checkpoint: when to checkpoint
      :param train_batch: 1
      :param test_batch: 1
      :param device: Defaults on GPU, pass 'cpu' as parameter to run on CPU. 
      :return: None
      """
      torch.backends.cudnn.benchmark = True #Comment this if you are not using a GPU...
      # set the network to training mode.
      model.train()
      model.cuda()  # if gpu available otherwise comment this line. 

      # your code goes here. 
      optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
      criterion = nn.CrossEntropyLoss().cuda()

      max_accuracy = None
      training_loss = []
      val_loss = []
      training_acc = []
      val_acc = []

      for epoch in range(1, num_epochs):
            if epoch % checkpoint == 0:
                  torch.save({
                        'epoch':epoch,
                        'optimizer': optimizer.state_dict(),
                        'train_loss': training_loss,
                        'val_loss': val_loss,
                        'train_acc' : training_acc,
                        'val_acc' : val_acc,
                        'model': model.state_dict(),
                  }, checkpoint_path+'/checkpoint.pt')
                  exit(10)
                  
            epoch_train_loss = 0
            epoch_acc_train = 0
            model.train() # set for training

            for _, data in enumerate(train_dataloader):
                  optimizer.zero_grad()

                  statement = data["statement"].to(device)
                  justification = data["justification"].to(device)
                  credit_history = data["credit_history"].to(device)

                  output = model(statement, justification, credit_history)
                  label = data["label"]
                  loss = criterion(output, label)
                  loss.backward() # calculate gradient
                  optimizer.step() # update weights

                  epoch_train_loss += loss.item()

                  __, predicted = torch.max(output.data, 1)
                  epoch_acc_train += (predicted == label).sum().item()

                  del statement, justification, credit_history, label

            training_loss.append(epoch_train_loss / (_*train_batch)) # loss per batch
            training_acc.append(epoch_acc_train / (_*train_batch)) # acc per batch

            # evaluation mode
            with torch.no_grad():  # No propagating losses here
                  model.eval()

                  epoch_val_loss = 0
                  epoch_val_accuracy = 0

                  for _, data in enumerate(val_dataloader):
                        optimizer.zero_grad()

                        statement = data["statement"].to(device)
                        justification = data["justification"].to(device)
                        credit_history = data["credit_history"].to(device)

                        output = model(statement, justification, credit_history)
                        label = data["label"]
                        loss = criterion(output, label)

                        epoch_val_loss += loss.item()

                        __, predicted = torch.max(output.data, 1)
                        epoch_val_accuracy += (predicted == label).sum().item()

                        del statement, justification, credit_history, label
                  val_loss.append(epoch_val_loss / (_*test_batch)) # loss per batch
                  val_acc.append(epoch_val_accuracy / (_*test_batch)) # acc per batch

                  if max_accuracy is None:
                        max_accuracy = epoch_val_accuracy / (_*test_batch)
                  else:
                        if (epoch_val_accuracy / (_*test_batch)) > max_accuracy:
                              max_accuracy = epoch_val_accuracy / (_*test_batch)
                              torch.save(model.state_dict(), path_to_save+'/model.pth')



      plt.plot(training_acc)
      plt.plot(val_acc)
      plt.plot(training_loss)
      plt.plot(val_loss)
      plt.show()


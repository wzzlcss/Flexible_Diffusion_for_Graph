def sgd_step(net, optimizer, feat_data, labels, train_data, device):
    """
    Function to updated weights with a SGD backpropagation
    args : net, optimizer, train_loader, test_loader, loss function, number of inner epochs, args
    return : train_loss, test_loss, grad_norm_lb
    """
    net.train()
    epoch_loss = []
    epoch_acc = []    
    # Run over the train_loader
    mini_batches, adj = train_data
    for mini_batch in mini_batches:
        # compute current stochastic gradient
        optimizer.zero_grad()
        output = net(feat_data, adj)       
        loss = net.criterion(output[mini_batch], labels[mini_batch])
        loss.backward()        
        optimizer.step()
        epoch_loss.append(loss.item())        
        output = output.argmax(dim=1)
        acc = f1_score(output[mini_batch].detach().cpu(), 
                       labels[mini_batch].detach().cpu(), average="micro")
        epoch_acc.append(acc)
    return epoch_loss, epoch_acc

def inference(eval_model, feat_data, labels, test_data, device):
    eval_model = eval_model.to(device)
    mini_batch, adj = test_data    
    output = eval_model(feat_data, adj)
    loss = eval_model.criterion(output[mini_batch], labels[mini_batch]).item()   
    output = output.argmax(dim=1)        
    acc = f1_score(output[mini_batch].detach().cpu(), 
                   labels[mini_batch].detach().cpu(), average="micro")
    return loss, acc

def train(model, data_loader, device, note):
    train_model = copy.deepcopy(model).to(device)

    results = ResultRecorder(note=note)

    optimizer = optim.Adam(train_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    tbar = trange(args.epoch_num, desc='Training Epochs')
    for epoch in tbar:
        # fetch train data 
        
        sample_time_st = time.perf_counter()
        train_data = data_loader.get_mini_batches(batch_size=args.batch_size)
        sample_time = time.perf_counter() - sample_time_st
        
        compute_time_st = time.perf_counter()
        train_loss, train_acc = sgd_step(train_model, optimizer, feat_data_th, labels_th, train_data, device)
        compute_time = time.perf_counter() - compute_time_st
        
        epoch_train_loss = np.mean(train_loss)
        epoch_train_acc = np.mean(train_acc)

        valid_data = data_loader.get_valid_batch()
        epoch_valid_loss, epoch_valid_acc = inference(train_model, feat_data_th, labels_th, valid_data, device)
        tbar.set_postfix(loss=epoch_train_loss,
                         val_loss=epoch_valid_loss,
                         val_score=epoch_valid_acc)

        results.update(epoch_train_loss, 
                       epoch_train_acc,
                       epoch_valid_loss, 
                       epoch_valid_acc, 
                       train_model, sample_time=sample_time, compute_time=compute_time)


    test_data = data_loader.get_test_batch()
    epoch_test_loss, epoch_test_acc = inference(results.best_model, feat_data_th, labels_th, test_data, device)
    results.test_loss = epoch_test_loss
    results.test_acc = epoch_test_acc
    print('Test_loss: %.4f | test_acc: %.4f' % (epoch_test_loss, epoch_test_acc))
    
    print('Average computing time per step %.5fs'%(np.mean(results.compute_time)))
    
    return results, epoch_test_acc, np.mean(results.compute_time)
from pathlib import Path
import torch
import json
from fontTools.misc.psOperators import ps_integer

from data_processing.linearwarmup_cosineLR import LinearWarmupCosineAnnealingLR
from train_test.loss_eval import ReconstructionLoss
from data_processing.mask import sequence_order_position,masked_batch_generation,masked_batch_generation_replace

def model_training(ds_loader, net, model_name, params):
    run_dir = (Path(params.save_path) / f"{model_name[0]}_run_90")
    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    print('Checkpoint directory:', run_dir.resolve())

    model_link = run_dir / 'best.pth'

    if model_link.exists():
        raise ValueError('The model already trained ')
    else:
        # generate loader
        train_loader = ds_loader.train_loader_generation(batch_size=params.batch_size, shuffle=True)

        train(model_name=model_name[0],
              net=net,
              train_loader=train_loader,
              params=params,
              train_type=2,
              run_dir=run_dir)





def train(model_name,net,train_loader,params,train_type, run_dir):
    net = net.cuda()

    # optimizer = torch.optim.Adam(list(net.parameters()), lr=params.lr)
    optimizer = torch.optim.AdamW(list(net.parameters()), lr=params.lr)
    scheduler = None
    if params.scheduling==True:
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=params.n_epochs)

    rcloss_fn = ReconstructionLoss(loss_type=params.loss_type)

    # training model
    train_loss_list = []
    train_best_loss = 100000
    state_dict = {
        'epoch': 0,
        'best_loss': train_best_loss,
        'model': net.state_dict(),
    }

    if train_type==1:
        train_layer_based(optimizer,
                          scheduler,
                          train_loss_list,
                          train_best_loss,
                          params,
                          train_loader,
                          net,
                          rcloss_fn,
                          state_dict,
                          model_name)


    if train_type ==2:
        train_epoch_based(optimizer,
                          scheduler,
                          train_loss_list,
                          train_best_loss,
                          params,
                          train_loader,
                          net,
                          rcloss_fn,
                          state_dict,
                          model_name,
                          run_dir)

    elif train_type ==3:
        train_batch_based(optimizer,
                          scheduler,
                          train_loss_list,
                          train_best_loss,
                          params,
                          train_loader,
                          net,
                          rcloss_fn,
                          state_dict,
                          model_name)



def data_process_training(b_index,params,data):
    # print('data batch:' + str(b_index))

    # print(data.isnan().any())
    #if (b_index + 1) % 100 == 0:
    #    print('batch number: ' + str(b_index))

    # for each batch, generate a masked_batch

    mask_pos_list = sequence_order_position(window_size=params.w, data_dimension=data.shape[2])
    generated_batch, generated_labels = masked_batch_generation_replace(data_batch=data,
                                                                        mask_pos_list=mask_pos_list,
                                                                        window_size=params.w,
                                                                        mask_value=params.masked_value,
                                                                        predicted_length=params.predicted_length)
    return generated_batch,generated_labels


def train_epoch_based(optimizer,
                      scheduler,
                      train_loss_list,
                      train_best_loss,
                      params,
                      train_loader,
                      net,
                      rcloss_fn,
                      state_dict,
                      model_name,
                      run_dir):

    config = vars(params).copy()

    config['model_name'] = model_name
    config['num_channels'] = list(net.num_channels)
    config['model_class'] = net.__class__.__name__

    with open(
            run_dir / 'config.json',
            'w',
            encoding='utf-8'
    ) as file:
        json.dump(
            config,
            file,
            indent=4,
            ensure_ascii=False,
            default=str
        )

    for e in range(params.n_epochs):

       # print('Train epoch ' + str(e) + '--------')

        # Tổng loss của tất cả layer trong epoch hiện tại
        train_loss_sum = 0.0

        # Số lượng loss đã được cộng
        number_loss = 0

        for i, _ in enumerate(net.num_channels):

            #print('train layer: ' + str(i))

            # Đặt toàn bộ model về eval mode trước.
            #
            # Các layer trước layer i sẽ:
            # - không cập nhật BatchNorm
            # - không sử dụng Dropout ngẫu nhiên
            net.eval()

            # Khóa gradient của tất cả encoder layer
            for j, _ in enumerate(net.num_channels):

                for pr in net.net[j][0].parameters():
                    pr.requires_grad = False

            # Chỉ mở gradient cho encoder layer hiện tại
            for pr in net.net[i][0].parameters():
                pr.requires_grad = True

            # Layer hiện tại hoạt động ở train mode
            net.net[i][0].train()

            train_layer_loss_sum = 0.0

            for b_index, (data, _) in enumerate(
                    train_loader):

                generated_batch, generated_labels = (
                    data_process_training(
                        b_index,
                        params,
                        data
                    )
                )

                # Feed data to GPU
                generated_batch = generated_batch.cuda()
                generated_labels = generated_labels.cuda()

                # Xóa gradient của batch trước
                optimizer.zero_grad()

                # Forward đến layer i
                generated_batch, aux = (
                    net.forwardToLayer(
                        generated_batch,
                        i
                    )
                )

                # Kiểm tra output reconstruction và target
                if aux.shape != generated_labels.shape:
                    raise ValueError(
                        '\nReconstruction shape does not '
                        'match target shape.'
                        + '\nepoch: '
                        + str(e)
                        + '\nlayer: '
                        + str(i)
                        + '\nbatch: '
                        + str(b_index)
                        + '\nhidden shape: '
                        + str(generated_batch.shape)
                        + '\nreconstruction shape: '
                        + str(aux.shape)
                        + '\ntarget shape: '
                        + str(generated_labels.shape)
                    )

                loss = rcloss_fn(
                    aux,
                    generated_labels
                )

                if not torch.isfinite(loss):
                    raise ValueError(
                        'Loss is NaN or infinity at '
                        + 'epoch '
                        + str(e)
                        + ', layer '
                        + str(i)
                        + ', batch '
                        + str(b_index)
                    )

                # Backpropagation local loss
                loss.backward()

                # Chỉ layer i có gradient nên chỉ layer i
                # được optimizer cập nhật
                optimizer.step()

                cpu_loss = loss.item()

                train_layer_loss_sum += cpu_loss
                train_loss_sum += cpu_loss

                number_loss += 1


            # Loss trung bình của riêng layer i
            train_layer_loss = (
                    train_layer_loss_sum/ len(train_loader)
            )

            print(
                f"Epoch {e + 1}/{params.n_epochs} | "
                f"Layer {i + 1}/{len(net.num_channels)} | "
                f"Loss: {train_layer_loss:.6f}"
            )

        # Scheduler được cập nhật một lần
        # sau khi tất cả layer hoàn thành epoch
        if params.scheduling:
            scheduler.step()

        # Loss trung bình của:
        # tất cả layer × tất cả batch
        train_loss_epoch = (
                train_loss_sum/ number_loss
        )

        print(
            f"Epoch {e + 1}/{params.n_epochs} | "
            f"Epoch Loss: {train_loss_epoch:.6f}"
        )
        print()

        train_loss_list.append(
            train_loss_epoch
        )

        # Lưu toàn bộ model tốt nhất theo epoch
        if train_loss_epoch < train_best_loss:
            train_best_loss = train_loss_epoch
            best_epoch = e

            state_dict = {
                'epoch': best_epoch,
                'best_loss': train_best_loss,
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': (
                    scheduler.state_dict()
                    if scheduler is not None else None
                ),
                'config': config
            }

            torch.save(
                state_dict,
                run_dir / 'best.pth'
            )

           # print(
             #   f"Saved best.pth - "
             #   f"Epoch {e + 1}, "
             #   f"Loss {train_best_loss:.6f}"
           # )

        last_checkpoint = {
            'epoch': e,
            'best_loss': train_best_loss,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': (
                scheduler.state_dict()
                if scheduler is not None else None
            ),
            'config': config
        }

        torch.save(
            last_checkpoint,
            run_dir / 'last.pth'
        )

        torch.save(
            train_loss_list,
            run_dir / 'loss_history.pt'
        )

    # Khôi phục model tốt nhất vào net
    state_dict = torch.load(
        run_dir / 'best.pth',
        map_location='cuda'
    )

    net.load_state_dict(
        state_dict['model']
    )

    # Mở lại gradient cho tất cả layer
    # sau khi training kết thúc
    for i, _ in enumerate(net.num_channels):

        for pr in net.net[i][0].parameters():
            pr.requires_grad = True

    net.eval()

    #print('Training completed')

    #print(    'best epoch: ' + str(state_dict['epoch']))

   # print('best loss: '+ str(state_dict['best_loss']))

    print(
        f"Completed | Best epoch: "
        f"{state_dict['epoch'] + 1} | "
        f"Best loss: {state_dict['best_loss']:.6f}"
    )

    return state_dict, train_loss_list


def train_layer_based(optimizer,
                      scheduler,
                      train_loss_list,
                      train_best_loss,
                      params,
                      train_loader,
                      net,
                      rcloss_fn,
                      state_dict,
                      model_name):

    for i, _ in enumerate(net.num_channels):
        print('train layer: ' + str(i))


        for e in range(params.n_epochs):
            print('---epoch ' + str(e) + '--------')
            train_loss_sum = 0.0
            for b_index, (data, label) in enumerate(train_loader):

                generated_batch,generated_labels =data_process_training(b_index, params, data)

                # feed data to GPUs
                generated_batch = generated_batch.cuda()
                generated_labels = generated_labels.cuda()

                generated_batch, aux = net.forwardToLayer(generated_batch, i)
                loss = rcloss_fn(aux, generated_labels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                cpu_loss = loss.item()
                train_loss_sum += cpu_loss

            if params.scheduling:
                scheduler.step()

            print('n_epoch is: ' + str(e))
            print('train_loss_epoch is ' + str(train_loss_sum))
            train_loss_list.append(train_loss_sum)

            if train_loss_sum < train_best_loss:
                train_best_loss = train_loss_sum
                best_epoch = e
                print('New train_loss is: ' + str(train_best_loss))
                print('n_epoch is: ' + str(e))

                state_dict = {
                    'epoch': best_epoch,
                    'best_loss': train_best_loss,
                    'model': net.state_dict(),
                }
                if train_best_loss==0.0: break

        torch.save(state_dict, params.save_path + model_name +'layer '+str(i))
        torch.save(train_loss_list, params.save_path + model_name + '_loss_list_layer '+str(i))

        for pr in net.layers[i].parameters():
            pr.requires_grad = False


def train_batch_based(
        net,
        train_loader,
        optimizer,
        scheduler,
        rcloss_fn,
        params,
        run_dir
):
    num_layers = len(net.num_channels)
    train_loss_list = []

    for b_index, (data, _) in enumerate(train_loader):
        print('Train batch ' + str(b_index) + '--------')

        generated_batch, generated_labels = data_process_training(
            b_index,
            params,
            data
        )

        generated_batch = generated_batch.cuda()
        generated_labels = generated_labels.cuda()

        input_batch = generated_batch

        for e in range(params.n_epochs):
            layer_loss_list = []

            for layerid in range(num_layers):
                for pr in net.parameters():
                    pr.requires_grad = False

                for pr in net.net[layerid].parameters():
                    pr.requires_grad = True

                net.eval()
                net.net[layerid].train()

                optimizer.zero_grad()

                _, aux = net.forwardToLayer(
                    input_batch,
                    layerid
                )

                if aux.shape != generated_labels.shape:
                    raise ValueError(
                        'Shape mismatch at batch '
                        + str(b_index)
                        + ', epoch '
                        + str(e)
                        + ', layer '
                        + str(layerid)
                        + ': aux='
                        + str(aux.shape)
                        + ', labels='
                        + str(generated_labels.shape)
                    )

                loss = rcloss_fn(
                    aux,
                    generated_labels
                )

                if not torch.isfinite(loss):
                    raise ValueError(
                        'Loss is NaN or Inf at batch '
                        + str(b_index)
                        + ', epoch '
                        + str(e)
                        + ', layer '
                        + str(layerid)
                    )

                loss.backward()
                optimizer.step()

                layer_loss_list.append(loss.item())

            batch_epoch_loss = (
                sum(layer_loss_list) / num_layers
            )

            train_loss_list.append({
                'batch': b_index,
                'epoch': e,
                'layer_loss': layer_loss_list,
                'loss': batch_epoch_loss
            })

            print(
                'Batch: '
                + str(b_index)
                + ' - Epoch: '
                + str(e)
                + ' - Layer loss: '
                + str(layer_loss_list)
                + ' - Loss: '
                + str(batch_epoch_loss)
            )

            if scheduler is not None:
                scheduler.step()

    torch.save(
        train_loss_list,
        run_dir / 'loss_history.pt'
    )

    return train_loss_list
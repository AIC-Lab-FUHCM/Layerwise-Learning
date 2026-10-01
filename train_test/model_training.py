from pathlib import Path
import torch
import json
from fontTools.misc.psOperators import ps_integer

from data_processing.linearwarmup_cosineLR import LinearWarmupCosineAnnealingLR
from train_test.loss_eval import ReconstructionLoss
from data_processing.mask import sequence_order_position,masked_batch_generation,masked_batch_generation_replace

def model_training(ds_loader, net, model_name, params):
    run_dir = (Path(params.save_path) / f"{model_name[0]}_backprop_8")
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

    if train_type ==1:
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

    elif train_type == 2:
        train_backprop(optimizer,
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

def data_process_training(b_index,params,data):
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
        train_loss_sum = 0.0
        number_loss = 0

        for i, _ in enumerate(net.num_channels):
            net.eval()

            for j, _ in enumerate(net.num_channels):

                for pr in net.net[j][0].parameters():
                    pr.requires_grad = False

            for pr in net.net[i][0].parameters():
                pr.requires_grad = True

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
            
                generated_batch = generated_batch.cuda()
                generated_labels = generated_labels.cuda()
                optimizer.zero_grad()
                generated_batch, aux = (
                    net.forwardToLayer(
                        generated_batch,
                        i
                    )
                )
     
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

                loss.backward()
                optimizer.step()
                cpu_loss = loss.item()
                train_layer_loss_sum += cpu_loss
                train_loss_sum += cpu_loss
                number_loss += 1

            train_layer_loss = (
                    train_layer_loss_sum/ len(train_loader)
            )

            print(
                f"Epoch {e + 1}/{params.n_epochs} | "
                f"Layer {i + 1}/{len(net.num_channels)} | "
                f"Loss: {train_layer_loss:.6f}"
            )

        if params.scheduling:
            scheduler.step()

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
 
    state_dict = torch.load(
        run_dir / 'best.pth',
        map_location='cuda'
    )

    net.load_state_dict(
        state_dict['model']
    )

    for i, _ in enumerate(net.num_channels):

        for pr in net.net[i][0].parameters():
            pr.requires_grad = True

    net.eval()

    print(
        f"Completed | Best epoch: "
        f"{state_dict['epoch'] + 1} | "
        f"Best loss: {state_dict['best_loss']:.6f}"
    )

    return state_dict, train_loss_list


def train_backprop(optimizer,
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
    config['train_type'] = 2
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

    # Cho phép tất cả encoder TCN nhận gradient.
    for i, _ in enumerate(net.num_channels):
        for pr in net.net[i][0].parameters():
            pr.requires_grad = True

    for e in range(params.n_epochs):
        net.train()

        train_loss_sum = 0.0
        number_loss = 0

        for b_index, (data, _) in enumerate(train_loader):
            generated_batch, generated_labels = (
                data_process_training(
                    b_index,
                    params,
                    data
                )
            )

            generated_batch = generated_batch.cuda()
            generated_labels = generated_labels.cuda()

            # Xóa gradient của batch trước.
            optimizer.zero_grad(set_to_none=True)

            # Forward qua toàn bộ TCN.
            # aux phải là reconstruction của layer cuối.
            generated_batch, aux = net(generated_batch)

            if aux.shape != generated_labels.shape:
                raise ValueError(
                    f'Epoch {e + 1}, batch {b_index}: '
                    f'reconstruction shape {aux.shape} '
                    f'does not match target shape '
                    f'{generated_labels.shape}'
                )

            # Chỉ dùng loss reconstruction cuối.
            loss = rcloss_fn(
                aux,
                generated_labels
            )

            if not torch.isfinite(loss):
                raise ValueError(
                    f'Loss is NaN or infinity at '
                    f'epoch {e + 1}, batch {b_index}'
                )

            # Truyền gradient từ loss cuối qua các encoder.
            loss.backward()

            # Kiểm tra kết nối gradient ở batch đầu tiên.
            if e == 0 and b_index == 0:
                for i, _ in enumerate(net.num_channels):
                    if not any(
                        pr.grad is not None
                        for pr in net.net[i][0].parameters()
                    ):
                        raise ValueError(
                            f'No gradient reaches TCN '
                            f'encoder layer {i}. '
                            'Check forward() for detach() '
                            'or torch.no_grad().'
                        )

            # Cập nhật các tham số nhận được gradient.
            optimizer.step()

            train_loss_sum += loss.item()
            number_loss += 1

        if params.scheduling:
            scheduler.step()

        train_loss_epoch = (
            train_loss_sum / number_loss
        )

        print(
            f"Epoch {e + 1}/{params.n_epochs} | "
            f"Epoch Loss: {train_loss_epoch:.6f}"
        )
        print()

        train_loss_list.append(
            train_loss_epoch
        )

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

    state_dict = torch.load(
        run_dir / 'best.pth',
        map_location='cuda'
    )

    net.load_state_dict(
        state_dict['model']
    )

    net.eval()

    print(
        f"Completed | Best epoch: "
        f"{state_dict['epoch'] + 1} | "
        f"Best loss: {state_dict['best_loss']:.6f}"
    )

    return state_dict, train_loss_list
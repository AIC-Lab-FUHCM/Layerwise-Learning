import numpy as np
from pathlib import Path
import torch
import matplotlib.pyplot as plt

from train_test.plot_results import plot_test_results
from sympy.physics.vector.printing import params
from data_processing.mask import sequence_order_position,masked_batch_generation, masked_test_batch_generation

def model_testing(ds_loader, net, model_name, params):
    project_dir = Path(__file__).resolve().parents[1]

    checkpoint_dir = Path(params.save_path)

    if not checkpoint_dir.is_absolute():
        checkpoint_dir = project_dir / checkpoint_dir

    
    checkpoints = {
        "local": (
            checkpoint_dir
            / f"{model_name[0]}_run_88"
            / "best.pth"
        ),
        "backprop": (
            checkpoint_dir
            / f"{model_name[0]}_backprop_7"
            / "best.pth"
        ),
    }

    if params.w != params.predicted_length:
        raise ValueError(
            "Cách ghép window này yêu cầu "
            "w == predicted_length."
        )

    for model_link in checkpoints.values():
        if not model_link.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint: {model_link}"
            )

    test_type = "normal"
    net = net.cuda()

    for method_name, model_link in checkpoints.items():
        checkpoint = torch.load(
            model_link,
            map_location="cuda"
        )

        net.load_state_dict(checkpoint["model"])
        net.eval()

        print(
            f"\nTesting {method_name.upper()}"
            f"\nCheckpoint: {model_link}"
        )

        result = test(
            net=net,
            ds_loader=ds_loader,
            w=params.w,
            masked_value=params.masked_value,
            predicted_length=params.predicted_length,
            batch_size=1,
            test_type=test_type
        )

        if result is None:
            raise ValueError(
                "Kiểm tra return trong test() "
                "và test_normal()."
            )

        plot_test_results(
            result=result,
            save_dir=(
                model_link.parent
                / "figures"
                / f"{test_type}_ts_{ds_loader.ts_num}"
            ),
            method_name=method_name,
            plot_range=None,
            save_test=(method_name == "local"),
            show=False
        )

    plt.show()

        

def test(net,ds_loader,w, masked_value, predicted_length,batch_size, test_type):

    anomalies_labels = ds_loader.dataset['test_label']
    test_loader = ds_loader.val_test_loader_generation(batch_size=batch_size,shuffle=False)
    threshold_list = np.linspace(0., 1, 800)
    if test_type == 'normal':
        return test_normal(anomalies_labels, test_loader, net, w, masked_value, predicted_length, test_type)

def test_normal(anomalies_labels,test_loader,net,w,masked_value,predicted_length,test_type):
    anomalies=None
    target_list = []
    reconstruction_list = []
    with torch.no_grad():
        net = net.cuda()
        net.eval()

        for b_index, (data, label) in enumerate(test_loader):
            if (b_index + 1) % 100 == 0:
                print('batch number: ' + str(b_index))

            mask_pos_list = sequence_order_position(window_size=w, data_dimension=data.shape[2])
            generated_batch, generated_labels = masked_test_batch_generation(data_batch=data,
                                                                             mask_pos_list=mask_pos_list,
                                                                             window_size=w, mask_value=masked_value,
                                                                             predicted_length=predicted_length,
                                                                             test_type=test_type)

            generated_batch = generated_batch.cuda()
            generated_labels = generated_labels.cuda()
            x, rc_output = net(generated_batch)
            if rc_output.shape != generated_labels.shape:
                raise ValueError(
                    "Reconstruction và target không cùng shape."
                )

            feature_count = rc_output.shape[-1]

            target_batch = generated_labels.detach().reshape(
                -1,
                feature_count
            )

            reconstruction_batch = rc_output.detach().reshape(
                -1,
                feature_count
            )

           
            if b_index > 0:
                target_batch = target_batch[-1:]
                reconstruction_batch = reconstruction_batch[-1:]

            target_list.append(target_batch.cpu())

            reconstruction_list.append(
                reconstruction_batch.cpu()
            )

            
            anomaly_scores = torch.linalg.vector_norm(rc_output - generated_labels,ord=2,dim=2)                                                                                 
            anomaly_scores = anomaly_scores.reshape(-1).detach().cpu() 
            if b_index == 0:
                anomalies = anomaly_scores.clone()
            else:
                anomalies = torch.cat( (anomalies, anomaly_scores[-1:])) 
                                                                            
    best_f1 = 0
    best_precision = 0
    best_recall = 0
    best_threshold = 0

    anomalies_labels = np.asarray(anomalies_labels).reshape(-1)
    anomalies = anomalies.numpy().reshape(-1)
    if len(anomalies) != len(anomalies_labels):
        raise ValueError(
            f"Số score ({len(anomalies)}) khác số nhãn "
            f"({len(anomalies_labels)}). "
            "Kiểm tra cách ghép window."
        )

    threshold_list = np.linspace( 0.0, anomalies.max() + 1e-6,800) 

    for delta in threshold_list:
        predicted = (anomalies >= delta).astype(np.int64)
        f1, precision, recall = F1_PA(anomalies_labels, predicted)

        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            best_threshold = delta

    print('Threshold: ' + str(best_threshold) + ' -Precision: ' + str(best_precision) + ' -Recall: ' + str(
        best_recall) + ' -F1: ' + str(best_f1))

    return {
        "target": torch.cat(
            target_list,
            dim=0
        ).numpy(),

        "reconstruction": torch.cat(
            reconstruction_list,
            dim=0
        ).numpy(),

        "score": anomalies,
        "label": anomalies_labels,
        "threshold": best_threshold,
    }

def F1_PA(groundtrue, predicted,delay=None):
    start_anomalies=[]
    end_anomalies=[]
    predicted = np.array(predicted)

    for i in range (len(groundtrue)-1):

        if groundtrue[i+1] ==1 and groundtrue[i]==0:
            start_anomalies.append(i+1)
        if groundtrue[i] ==1 and groundtrue[i+1] ==0.:
            end_anomalies.append(i+1)

    if delay ==None:

        for k in range (len(start_anomalies)):
            for j in range(start_anomalies[k], end_anomalies[k] + 1):
                if predicted[j] == 1.:
                    np.put(predicted, np.arange(start_anomalies[k], end_anomalies[k] + 1, 1), 1.)
                    break
    else:
        for k in range (len(start_anomalies)):
            for j in range(start_anomalies[k], start_anomalies[k]+delay):
                if predicted[j] == 1.:
                    np.put(predicted, np.arange(start_anomalies[k], end_anomalies[k] + 1, 1), 1.)
                    break

    score, precision, recall = F1_score(predicted, groundtrue)
    return score,precision, recall

def F1_score(reconstruced_labels ,anomaly_labels):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    if len(anomaly_labels)- len(reconstruced_labels) ==1:
        anomaly_labels = np.delete(anomaly_labels, -1)

    for index, value in enumerate(anomaly_labels):
        if value == 1.:
            if reconstruced_labels[index] == 1:
                TP += 1
            elif reconstruced_labels[index] == 0.:
                FN += 1
        elif value == 0.:
            if reconstruced_labels[index] == 0:
                TN += 1
            elif reconstruced_labels[index] == 1.:
                FP += 1

    if (TP+FP)==0 or (TP+FN)==0:
        return 0,0,0
    else:
        Precision = TP /(TP +FP)
        Recall = TP /(TP +FN)

        if (Precision +Recall)==0:
            return 0, 0, 0
        else:
            F_score = ( 2* Precision *Recall) / (Precision +Recall)
            return F_score, Precision,Recall


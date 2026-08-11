import numpy as np
from pathlib import Path
import torch
from sympy.physics.vector.printing import params

from data_processing.mask import sequence_order_position,masked_batch_generation, masked_test_batch_generation

def model_testing(ds_loader,
                  net,
                  model_name,
                  params):


    model_link = (Path(params.save_path)/ f"{model_name[0]}_run_90" / "best.pth")
    print(model_link)
    if model_link.exists():
        # generate loader
        #state_dict = torch.load(model_link)
        #net.load_state_dict(state_dict['model'])
        checkpoint = torch.load(
            model_link,
            map_location='cuda'
        )

        net.load_state_dict(
            checkpoint['model']
        )

        net = net.cuda()

        print(
            f"Loaded best.pth | "
            f"Epoch: {checkpoint['epoch'] + 1} | "
            f"Loss: {checkpoint['best_loss']:.6f}"
        )

        batch_size = params.batch_size
        if params.test_type =='normal':
            batch_size = 1

        test(net=net,
             ds_loader=ds_loader,
             w=params.w,
             masked_value=params.masked_value,
             predicted_length=params.predicted_length,
             batch_size=batch_size,
             test_type= params.test_type,
             )

    else:
        raise ValueError('The model is not trained yet ')



def test(net,ds_loader,w, masked_value, predicted_length,batch_size, test_type):

    anomalies_labels = ds_loader.dataset['test_label']


    test_loader = ds_loader.val_test_loader_generation(batch_size=batch_size,shuffle=False)



    threshold_list = np.linspace(0., 1, 800)
    if test_type == 'normal':
        test_normal(anomalies_labels, test_loader, net, w, masked_value, predicted_length, test_type)

    elif test_type =='online':
        test_online(anomalies_labels, test_loader, threshold_list, net, w, masked_value, predicted_length, test_type)



def test_online(anomalies_labels,test_loader,threshold_list,net,w,masked_value,predicted_length,test_type):
    anomalies=None
    reconstructed_element = None
    with torch.no_grad():
        net = net.cuda()
        net.eval()
        for delta in threshold_list:

            for b_index, (data, label) in enumerate(test_loader):

                if (b_index + 1) % 100 == 0:
                    print('batch number: ' + str(b_index))

                # for each batch, generate a masked_batch

                mask_pos_list = sequence_order_position(window_size=w, data_dimension=data.shape[1])

                generated_batch, generated_labels = masked_test_batch_generation(data_batch=data,
                                                                                 mask_pos_list=mask_pos_list,
                                                                                 window_size=w, mask_value=masked_value,
                                                                                 predicted_length=predicted_length,
                                                                                 test_type=test_type,
                                                                                 reconstructed_element=reconstructed_element)




                # feed data to GPUs
                generated_batch = generated_batch.cuda()
                generated_labels = generated_labels.cuda()



                # forward generated-batch
                # lay layer cuoi de test
                rc_output = net(generated_batch)

                anomaly_scores = torch.norm(torch.abs(torch.sub(rc_output, generated_labels)), dim=1)
                anomaly_scores = anomaly_scores.detach().cpu()
                print(anomaly_scores.shape)
                print(anomaly_scores)
                print(anomalies_labels[b_index])
                breakpoint()


                if b_index == 0:
                    t1 = torch.index_select(anomaly_scores, 0, torch.arange(0, 16, 1))
                    t2 = torch.index_select(anomaly_scores, 0, torch.arange(31, generated_batch.shape[0], 16))

                    f_dif = torch.cat((t1, t2), dim=0)
                    anomalies = torch.clone(f_dif)

                else:

                    f_dif = torch.index_select(anomaly_scores, 0, torch.arange(15, generated_batch.shape[0], 16))
                    anomalies = torch.cat((anomalies, f_dif), dim=0)

    best_f1 = 0
    best_precision = 0
    best_recall = 0
    best_threshold = 0

    for delta in threshold_list:
        conditioned = anomalies >= delta
        predicted = conditioned.int()
        f1, precision, recall = F1_PA(anomalies_labels, predicted)
        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            best_threshold = delta

    print('Threshold: ' + str(best_threshold) + ' -Precision: ' + str(best_precision) + ' -Recall: ' + str(
        best_recall) + ' -F1: ' + str(best_f1))




def test_normal(anomalies_labels,test_loader,net,w,masked_value,predicted_length,test_type):
    anomalies=None
    with torch.no_grad():
        net = net.cuda()
        net.eval()

        for b_index, (data, label) in enumerate(test_loader):
            if (b_index + 1) % 100 == 0:
                print('batch number: ' + str(b_index))

            # for each batch, generate a masked_batch

            mask_pos_list = sequence_order_position(window_size=w, data_dimension=data.shape[2])
            generated_batch, generated_labels = masked_test_batch_generation(data_batch=data,
                                                                             mask_pos_list=mask_pos_list,
                                                                             window_size=w, mask_value=masked_value,
                                                                             predicted_length=predicted_length,
                                                                             test_type=test_type)


            # feed data to GPUs
            generated_batch = generated_batch.cuda()
            generated_labels = generated_labels.cuda()

            # forward generated-batch
            x, rc_output = net(generated_batch)

            anomaly_scores = torch.linalg.vector_norm(rc_output - generated_labels,ord=2,dim=2) # L2 norm de tinh do lon cua rc_output va ground truth
                                                                                                 # vector_norm: gop score cua tat ca feature thanh 1 score tai 1 timestep
            anomaly_scores = anomaly_scores.reshape(-1).detach().cpu() # fatten, detach: tao 1 tensor luu anomaly score nhung khong cho backward

            if b_index == 0: #window dau tien
                anomalies = anomaly_scores.clone()
            else: #cac window sau
                anomalies = torch.cat( (anomalies, anomaly_scores[-1:])) # lay score cua timestep cuoi (moi), noi vao list anomaly
                                                                            #truot 1 (tensor[start:stop:step])
                """   if b_index == 0:
                                t1 = torch.index_select(anomaly_scores, 0, torch.arange(0, 16, 1))
                                t2 = torch.index_select(anomaly_scores, 0, torch.arange(31, generated_batch.shape[0], 16))

                                f_dif = torch.cat((t1, t2), dim=0)
                                anomalies = torch.clone(f_dif)

                            else:

                                f_dif = torch.index_select(anomaly_scores, 0, torch.arange(15, generated_batch.shape[0], 16))
                                anomalies = torch.cat((anomalies, f_dif), dim=0)
                        """

    best_f1 = 0
    best_precision = 0
    best_recall = 0
    best_threshold = 0

    anomalies_labels = np.asarray(anomalies_labels).reshape(-1)
    anomalies = anomalies.numpy().reshape(-1)

    threshold_list = np.linspace( 0.0, anomalies.max() + 1e-6,800) # threshold 0-max score


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



def F1_PA(groundtrue, predicted,delay=None):
    start_anomalies=[]
    end_anomalies=[]
    predicted = np.array(predicted)

    #find anomaly range
    for i in range (len(groundtrue)-1):

        if groundtrue[i+1] ==1 and groundtrue[i]==0:
            start_anomalies.append(i+1)
        if groundtrue[i] ==1 and groundtrue[i+1] ==0.:
            end_anomalies.append(i+1)

    # print(start_anomalies)
    # print(end_anomalies)

    # predicted = np.asarray(predicted)
    # print('len predict')
    # print(len(predicted))
    #
    # print('len ground-true')
    # print(len(groundtrue))

    # breakpoint()

    if delay ==None:

        # adjust predicted values
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



    # plt.plot(groundtrue)
    # plt.title('ground true labels')
    # plt.ylabel('labels')
    # plt.xlabel('time point')
    # plt.show()
    #
    # plt.plot(predicted)
    # plt.title('predicted labels')
    # plt.ylabel('labels')
    # plt.xlabel('time point')
    # plt.show()

    # score = F1_score(predicted,groundtrue)
    # score = f1_score(groundtrue, predicted)
    score, precision, recall = F1_score(predicted, groundtrue)
    return score,precision, recall



def F1_score(reconstruced_labels ,anomaly_labels):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    # print(len(reconstruced_labels), len(anomaly_labels))

    if len(anomaly_labels)- len(reconstruced_labels) ==1:
        anomaly_labels = np.delete(anomaly_labels, -1)


    # calculate TP,FP,TN,FN
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


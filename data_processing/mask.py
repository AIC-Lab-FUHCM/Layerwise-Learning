import numpy as np
import torch


def random_position2(window_size, data_dimension, masking_factor):
    num_list = np.arange(0, window_size, 1)
    rand_list = []
    rand_list.append(np.random.choice(num_list, int(window_size * masking_factor), replace=False))
    return rand_list

def random_position(window_size, data_dimension, masking_factor):
    num_list = np.arange(0, window_size, 1)
    rand_list = []

    for j in range(data_dimension):
        rand_list.append(np.random.choice(num_list, int(window_size*masking_factor), replace=False))
    return rand_list

def sequence_order_position(window_size, data_dimension):
    num_list = np.arange(0, window_size, 1)
    masked_pos_list = []
    masked_pos_list.append(num_list)
    return masked_pos_list


def masked_test_batch_generation(data_batch, mask_pos_list, window_size,predicted_length, mask_value,test_type, reconstructed_element=None):

    if test_type=='normal':
        masked_data_batch, reconstruct_label_data_batch = masked_batch_generation_replace(data_batch, mask_pos_list, window_size,predicted_length, mask_value)
        return masked_data_batch, reconstruct_label_data_batch

    elif test_type =='online':

        masked_data_batch = torch.zeros((1, data_batch.shape[1], window_size))
        reconstruct_label_data_batch = torch.zeros((1, data_batch.shape[1], predicted_length))

        for i in range(1):  # for each  sample in data_batch
            sliding_window = torch.clone(data_batch[i])
            if reconstructed_element!=None:
                for r in range (sliding_window.shape[0]):
                    sliding_window[r][window_size-2] = reconstructed_element[r][0]

            for j in range(i * len(mask_pos_list[0]), i * len(mask_pos_list[0]) + len(
                    mask_pos_list[0])):  # determine the range for filling data samples

                masked_data = torch.clone(sliding_window)
                for l in range(masked_data.shape[0]):  # loop according to dimension
                    mask_loc = mask_pos_list[0][j - (i * len(mask_pos_list[0]))]
                    masked_data[l][mask_loc] = mask_value
                    reconstruct_label_data_batch[j][l] = masked_data[l][mask_loc]

                masked_data_batch[j] = masked_data

        return masked_data_batch, reconstruct_label_data_batch


def masked_batch_generation_replace(data_batch, mask_pos_list, window_size, predicted_length, mask_value):
    n_batch_samples = data_batch.size(0)
    # Chỉ giữ các vị trí không làm đoạn mask vượt khỏi window
    valid_positions = [
        pos for pos in mask_pos_list[0]
        if pos + predicted_length <= window_size
    ]

    if len(valid_positions) == 0:
        raise ValueError(
            'No valid mask position: '
            f'window_size={window_size}, '
            f'predicted_length={predicted_length}'
        )

    n_mask_positions = len(valid_positions)
    n_batch_masked_samples = n_batch_samples * n_mask_positions

    # Pre-allocate tensors
    masked_data_batch = data_batch.unsqueeze(1).repeat(1, n_mask_positions, 1, 1)  # (B, N, C, W)

    masked_data_batch = masked_data_batch.reshape(n_batch_masked_samples, window_size, data_batch.size(2))


    reconstruct_label_data_batch = torch.zeros((n_batch_masked_samples, predicted_length, data_batch.size(2)),
                                               dtype=data_batch.dtype, device=data_batch.device)

    # Convert mask positions to tensor
    mask_positions = torch.tensor(valid_positions, dtype=torch.long, device=data_batch.device)

    for i in range(n_batch_samples):
        for j, pos in enumerate(mask_positions):
            idx = i * n_mask_positions + j
            reconstruct_label_data_batch[idx] = masked_data_batch[idx, pos:pos + predicted_length, :]
            masked_data_batch[idx, pos:pos + predicted_length, :] = mask_value

    return masked_data_batch, reconstruct_label_data_batch



def masked_batch_generation(data_batch, mask_pos_list, window_size,predicted_length, mask_value):

    n_batch_samples = len(data_batch)
    n_batch_masked_samples = len(mask_pos_list[0]) * n_batch_samples

    # generate data_batch
    masked_data_batch = torch.zeros((n_batch_masked_samples, data_batch.shape[1], window_size))
    reconstruct_label_data_batch = torch.zeros((n_batch_masked_samples, data_batch.shape[1], predicted_length))


    for i in range(n_batch_samples):  # for each  sample in data_batch
        sliding_window = torch.clone(data_batch[i])
        for j in range(i * len(mask_pos_list[0]), i * len(mask_pos_list[0]) + len(mask_pos_list[0])):  # determine the range for filling data samples

            masked_data = torch.clone(sliding_window)
            # reconstruct_label_data_batch[j] = torch.clone(sliding_window)
            for l in range(masked_data.shape[0]):  # loop according to dimension
                mask_loc = mask_pos_list[0][j - (i * len(mask_pos_list[0]))]
                reconstruct_label_data_batch[j][l] = masked_data[l][mask_loc]
                masked_data[l][mask_loc] = mask_value


            masked_data_batch[j] = masked_data

    return masked_data_batch, reconstruct_label_data_batch

    pass


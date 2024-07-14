import sequence_handling_utils as seq_utils
import torch
import datasets as datasets
import os, sys

N = 10
B = 6
C = 3
H = 5
W = 4

# raft.generate_template
def generate_template_seq_success_test():
    seq = torch.randn(N, H, W)
    expected_result = torch.zeros(seq.shape[1], seq.shape[2])
    for n in range(0, seq.shape[0]):
        for h in range(0, seq.shape[1]):
            for w in range(0, seq.shape[2]):
                expected_result[h,w] += seq[n,h,w]
    expected_result /= seq.shape[0]
    if(expected_result.all() == seq_utils.generate_template(seq, "avg").all()):
        print("seq_utils.generate_template with sequence as input Test passed.")
        return 1
    else:
        print("seq_utils.generate_template with sequence as input Test failed.")
        return 0

def generate_template_batch_success_test():
    seq = torch.randn(B, N, H, W)
    expected_result = torch.zeros(seq.shape[0], seq.shape[2], seq.shape[3])
    for b in range(0, seq.shape[0]):
        for n in range(0, seq.shape[1]):
            for h in range(0, seq.shape[2]):
                for w in range(0, seq.shape[3]):
                    expected_result[b,h,w] += seq[b,n,h,w]
    expected_result /= seq.shape[1]
    if(expected_result.all() == seq_utils.generate_template(seq, "avg").all()):
        print("seq_utils.generate_template with batch as input Test passed.")
        return 1
    else:
        print("seq_utils.generate_template with batch as input Test failed.")
        return 0

def warp_batch_shape_success_test():
    seq_batch = torch.randn(B, N, H, W) #[B, C, H, W] Batch of images sequences im1
    flo_batch = torch.randn(B, N, 2, H, W) # flow_predictions
    
    warped_batch = seq_utils.warp_batch(seq_batch, flo_batch)
    if (warped_batch.shape ==  torch.Size([B, N, H, W])):
        print("seq_utils.warp_batch test for output tensor shape passed.")
        return 1
    else:
        print("seq_utils.warp_batch test for output tensor shape failed. Supposed to be", [B, N, H, W], ", instead it is", 
                                                                               warped_batch.shape.value)
        return 0
    
def warp_batch_null_flow_success_test():
    seq_batch = torch.randn(B, N, H, W) #[B, C, H, W] Batch of images sequences im1
    flo_batch = torch.zeros(B, N, 2, H, W) # flow_predictions
    
    warped_batch = seq_utils.warp_batch(seq_batch, flo_batch)
    if (warped_batch.all() == seq_batch.all()):
        print("seq_utils.warp_batch test for null flow passed.")
        return 1
    else:
        print("seq_utils.warp_batch test for for null flow failed. Expected the warped seq to be the same as the orgininal seq.")
        return 0
def check_template_repeats_properly_ACDC_dataset():
    train_dataset = datasets.ACDCDataset("/home/guests/manal_hamdi/manal/RAFT/datasets/ACDC_processed/training", "training")
    seq, template = train_dataset[0]
    temp_ref = template[0, :, :]
    for n in range(0, template.shape[0]):
        if(template[n, :, :].all() != temp_ref.all()):
            print("template does not repeat properly in ACDC Dataset get item. check_template_repeats_properly_ACDC_dataset failed")
            return 0
    print("check_template_repeats_properly_ACDC_dataset passed.") 
    return 1

        
            
total = 5
passed = 0
passed += generate_template_seq_success_test()
passed += generate_template_batch_success_test()
passed += warp_batch_shape_success_test()
passed += check_template_repeats_properly_ACDC_dataset()
passed += warp_batch_null_flow_success_test()


print("Passed", passed, "/", total, ".")
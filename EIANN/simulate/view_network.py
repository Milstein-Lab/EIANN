from EIANN import utils as ut

network_pkl_name = '20231129_EIANN_2_hidden_mnist_van_bp_relu_SGD_config_G_complete_optimized_66049_257.pkl'
path = f'/scratch/yc1376/data/eiann/{network_pkl_name}'
net = ut.load_network(path)

name = path.split('/')[-1].split('.')[0]
print(f'Network Name: {name}')
print(f'Final Val Accuracy: {net.val_accuracy_history[-1]}')
print(f'Final Val Loss: {net.val_loss_history[-1]}')
print(f'Using Device: {net.device}')
print(f'Network Run Time: {net.run_time} sec')

# salloc --mem=4G --time=00:15:00 --cpus-per-task=1



# ----- Tracking networks ----- 

# van_bp (CPU)
# - Final Val Accuracy: 96.52999877929688
# - Final Val Loss: 0.008667627349495888
# - Using Device: cpu
# - Network Run Time: 116.58439445495605 sec
# - 5 Networks Runtime: 580 sec

# van_bp (GPU)
# TODO: Running into pkl opening error but pkl did successfully load
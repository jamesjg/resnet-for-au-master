import os
import numpy as np

data_root = '/media/ljy/新加卷/FEAFA+/Disfa'
#You nead downloading DISFA including 'ActionUnit_Labels'
label_path = '/media/ljy/新加卷/Disfa_original/ActionUnit_Labels'
list_path_prefix = './list/'

part1 = ['SN002','SN010','SN001','SN026','SN027','SN032','SN030','SN009']
part2 = ['SN013','SN018','SN011','SN028','SN012','SN006','SN031','SN021','SN024']
part3 = ['SN003','SN029','SN023','SN025','SN008','SN005','SN007','SN017','SN004']

test_img_prefix_part1 = ['gong2_2_', 'li2_10_', 'li2_1_','cao2_26_','gong2_27_','','cao2_30_','']
test_img_prefix_part2 = ['li2_13_', 'gong2_18_', 'li2_11_', 'cao2_28_', 'gong2_12_', 'li2_6_', '','gong2_21_', 'cao2_24_']
test_img_prefix_part3 = ['cao2_3_', 'cao2_29_', 'gong2_23_', 'gong2_25_', 'gong2_8_', '', 'li2_7_', 'cao2_17_', 'li2_4_']
# fold1:  train : part1+part2 test: part3
# fold2:  train : part1+part3 test: part2
# fold3:  train : part2+part3 test: part1

test_part = part1 + part2 + part3
test_img_prefix = test_img_prefix_part1 + test_img_prefix_part2 + test_img_prefix_part3

#au_idx = [1, 2, 4, 6, 9, 12, 25, 26] # [1,2,4,5,6,9,12,15,17,20,25,26]
au_idx = [1,2,4,5,6,9,12,15,17,20,25,26]


with open(list_path_prefix + 'DISFA_test_img_path.txt','w') as f:
    u = 0

frame_list = []
numpy_list = []
for folder_idx, fr in enumerate(test_part):
    if fr[-2] == str(0):
        id = fr[-1]
    else:
        id = fr[-2:]

    fr_path = os.path.join(label_path,fr)
    au1_path = os.path.join(fr_path,fr+'_au1.txt')
    with open(au1_path, 'r') as label:
        total_frame = len(label.readlines())  #这个视频中的所有帧
    au_label_array = np.zeros((total_frame,12),dtype=np.int)
    for ai, au in enumerate(au_idx):
        AULabel_path = os.path.join(fr_path,fr+'_au'+str(au) +'.txt')
        if not os.path.isfile(AULabel_path):
            print(AULabel_path + 'do not exist')
            continue
        print("--Checking AU:" + str(au) + " ...")
        with open(AULabel_path, 'r') as label:
            for t, lines in enumerate(label.readlines()):  #对该视频中的每一个帧
                frameIdx, AUIntensity = lines.split(',')
                frameIdx, AUIntensity = int(frameIdx), int(AUIntensity)
                au_label_array[t,ai] = AUIntensity
    for i in range(total_frame):
        if int(id) < 28 :
            frame_img_name = 'Disfa_train'+'/' + id + '.output' +'/' + test_img_prefix[folder_idx] + str(i).zfill(8) + '_crop.jpg'  
        else:
            frame_img_name = 'Disfa_test'+'/' + id + '.output' +'/' + test_img_prefix[folder_idx] + str(i).zfill(8) + '_crop.jpg'          
        frame_list.append(frame_img_name)
        with open(list_path_prefix + 'DISFA_test_img_path.txt', 'a+') as f:
            f.write(frame_img_name+'\n')
    numpy_list.append(au_label_array)

numpy_list = np.concatenate(numpy_list,axis=0)
# part1 test for fold3
np.savetxt(list_path_prefix + 'DISFA_test_label.txt', numpy_list,fmt='%d', delimiter=' ')


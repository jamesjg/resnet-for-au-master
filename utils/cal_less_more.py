import numpy as np
def pred_less_and_more(pred, label):
    pred_less = pred < label
    pred_less[label==0] = 0
    pred_less[label==1] = 0
    pred_more = pred > label 
    pred_more[label==0] = 0
    pred_more[label==1] = 0
    print("非0,1标签中,预测小了的个数:", np.sum(pred_less, axis=0))
    print("非0,1标签中,预测大了的个数:", np.sum(pred_more, axis=0))
    print("非0,1标签中,预测小了的个数:", np.sum(pred_less))
    print("非0,1标签中,预测大了的个数:", np.sum(pred_more))
    pred_less = pred < label 
    pred_less[label<0.1] = 0
    pred_less[label>0.3] = 0
    pred_more = pred > label 
    pred_more[label<0.1] = 0
    pred_more[label>0.3] = 0
    print("标签0.1-0.3中,预测小了的个数:", np.sum(pred_less, axis=0))
    print("标签0.1-0.3中,预测大了的个数:", np.sum(pred_more, axis=0))
    print("标签0.1-0.3中,预测小了的个数:", np.sum(pred_less))
    print("标签0.1-0.3中,预测大了的个数:", np.sum(pred_more))
    pred_less = pred < label 
    pred_less[label<=0.3] = 0
    pred_less[label>0.5] = 0
    pred_more = pred > label
    pred_more[label<=0.3] = 0
    pred_more[label>0.5] = 0
    print("标签0.3-0.5中,预测小了的个数:", np.sum(pred_less, axis=0))
    print("标签0.3-0.5中,预测大了的个数:", np.sum(pred_more, axis=0))
    print("标签0.3-0.5中,预测小了的个数:", np.sum(pred_less))
    print("标签0.3-0.5中,预测大了的个数:", np.sum(pred_more))
    pred_less = pred < label
    pred_less[label<=0.5] = 0
    pred_less[label>0.7] = 0
    pred_more = pred > label 
    pred_more[label<=0.5] = 0
    pred_more[label>0.7] = 0
    print("标签0.5-0.7中,预测小了的个数:", np.sum(pred_less, axis=0))
    print("标签0.5-0.7中,预测大了的个数:", np.sum(pred_more, axis=0))
    print("标签0.5-0.7中,预测小了的个数:", np.sum(pred_less))
    print("标签0.5-0.7中,预测大了的个数:", np.sum(pred_more))
    pred_less = pred < label 
    pred_less[label<=0.7] = 0
    pred_less[label>0.9] = 0
    pred_more = pred > label 
    pred_more[label<=0.7] = 0
    pred_more[label>0.9] = 0
    print("标签0.7-0.9中,预测小了的个数:", np.sum(pred_less, axis=0))
    print("标签0.7-0.9中,预测大了的个数:", np.sum(pred_more, axis=0))
    pred_more[label>0.9] = 0
    print("标签0.7-0.9中,预测小了的个数:", np.sum(pred_less))
    print("标签0.7-0.9中,预测大了的个数:", np.sum(pred_more))

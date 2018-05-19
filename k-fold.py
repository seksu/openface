import os
import shutil
import sys
from subprocess import Popen, PIPE
import numpy as np
loss = 0
def infer(save_path):
    proc = Popen(
        'python2 demos/classifier.py infer demos/509_new_features/classifier.pkl ' + save_path,
        shell=True,
        stdout=PIPE,
        stderr=PIPE
    )
    proc.wait()
    res = proc.communicate()
    if proc.returncode:
        print('res[1] >>>>',res[1])
        global loss
        loss += 1
    result = res[0]
    return result

def train(model):
    proc = Popen(
        'python2 demos/classifier.py train demos/509_new_features --classifier ' + model,
        shell=True,
        stdout=PIPE,
        stderr=PIPE
    )
    proc.wait()
    res = proc.communicate()
    if proc.returncode:
        print('res[1] >>>>',res[1])
    result = res[0]
    return result

def represent():
    proc = Popen(
        './batch-represent/main.lua -outDir demos/509_new_features -data demos/509_new',
        shell=True,
        stdout=PIPE,
        stderr=PIPE
    )
    proc.wait()
    res = proc.communicate()
    if proc.returncode:
        print('res[1] >>>>',res[1])
    result = res[0]
    return result

def check(temp):
    lines = temp.split('\n')
    lineEnd = lines[len(lines)-2]
    words = lineEnd.split(' ')
    if(words[0] == "Predict"):
        return words[1]
    else:
        return None

path_origin = ['/home/pansek/openface/demos/509/pan',
               '/home/pansek/openface/demos/509/ball',
               # '/home/pansek/openface/demos/509/aeim',
               # '/home/pansek/openface/demos/509/bee',
               # '/home/pansek/openface/demos/509/donk',
               # '/home/pansek/openface/demos/509/jane',
               '/home/pansek/openface/demos/509/unknown']

path_align = ['/home/pansek/openface/demos/509_align/pan',
              '/home/pansek/openface/demos/509_align/ball',
              # '/home/pansek/openface/demos/509_align/aeim',
              # '/home/pansek/openface/demos/509_align/bee',
              # '/home/pansek/openface/demos/509_align/donk',
              # '/home/pansek/openface/demos/509_align/jane',
              '/home/pansek/openface/demos/509_align/unknown']

path_new = ['/home/pansek/openface/demos/509_new/pan',
            '/home/pansek/openface/demos/509_new/ball',
            # '/home/pansek/openface/demos/509_new/aeim',
            # '/home/pansek/openface/demos/509_new/bee',
            # '/home/pansek/openface/demos/509_new/donk',
            # '/home/pansek/openface/demos/509_new/jane',
            '/home/pansek/openface/demos/509_new/unknown']

company = '/home/pansek/openface/demos/509_new'

segment = 30
all_seg = 90.0
model = ['LinearSvm',
         'GridSearchSvm',
         'GMM',
         'RadialSvm',
         'DecisionTree',
         'GaussianNB',
         'DBN']

# nickname = ['pan','ball','aeim','bee','donk','jane','unknown']
conclusion = []
lost_list = []
correct_list = []
incorrect_list = []


for mode in model:
    print('model is {}'.format(mode))
    temp_score = []
    first = 0
    last = 30
    total = 300
    loss = 0
    correct_all = 0
    incorrect_all = 0
    for k in range(1,11):
        if os.path.exists(company) and os.path.isdir(company):
            shutil.rmtree(company)
            os.mkdir(company)
        else:
            os.mkdir(company)
        print("make company : {}\n".format(k))
        for new,now in zip(path_new,path_align):
            if os.path.exists(new) and os.path.isdir(new):
                shutil.rmtree(new)
                os.mkdir(new)
            else:
                os.mkdir(new)
            list = os.listdir(now)[0:first]
            list.extend(os.listdir(now)[last:total])
            for pic in list:
                shutil.copy(now+'/'+pic, new)
            print("make : {} done\n".format(new))

        rep = represent()
        print(rep)
        train_result = train(mode)
        print(train_result)
        correct = 0
        for test,align in zip(path_origin,path_align):
            face_count = 0
            for pic in os.listdir(align)[first:last]:
                img = test+'/'+pic.split('.')[0]+'.jpg'
                predict = infer(img)
                ans = check(predict)
                if ans == test.split('/')[6]:
                    correct += 1
                    correct_all += 1
                    print('***** CORRECT {} ****** ALL IS {}'.format(correct,correct_all))
                else:
                    incorrect_all += 1
                    print('***** IN-CORRECT {} ******'.format(incorrect_all))
                face_count += 1
                print('>>> round : {} <<<{} : {} '.format(k,img,face_count))
        print("{} / {} ".format(correct,all_seg))
        score = correct/all_seg*100
        print("round {} have score : {} loss : {}\n".format(k,score,loss))
        temp_score.append(score)
        first+=segment
        last+=segment
    lost_list.append(loss)
    score_re = np.mean(temp_score)
    conclusion.append(score_re)
    correct_list.append(correct_all)
    incorrect_list.append(incorrect_all)

for mod,con,lost,correct,incorrect in zip(model,conclusion,lost_list,correct_list,incorrect_list):
    with open("Output_kfold.txt", "a") as text_file:
        text_file.write("Accuracies of {} is {} and loss face : {} (>>correct : {} incorrect: {}<<)\n".format(mod,con,lost,correct,incorrect))

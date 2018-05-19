import os
import subprocess
import shutil
import numpy as np

# def facerecognition():
#     subprocess.check_call(['python', '/home/pansek/openface/demos/classifier.py', 'infer', 'classifier.pkl', '/home/pansek/openface/demos/pan.jpg'])
#
# company = '/home/pansek/openface/demos/509_new'
# if os.path.exists(company) and os.path.isdir(company):
#     print os.path.exists(company),os.path.isdir(company)

person_name = [ '/home/pansek/openface/demos/509_align/pan',
'/home/pansek/openface/demos/509_align/ball',
'/home/pansek/openface/demos/509_align/aeim',
'/home/pansek/openface/demos/509_align/bee',
'/home/pansek/openface/demos/509_align/donk',
'/home/pansek/openface/demos/509_align/jane',
'/home/pansek/openface/demos/509_align/unknown']


for person in person_name:
    sname = person.split('/')[6]
    print(sname + '=>',len([name for name in os.listdir(person) if os.path.isfile(os.path.join(person, name))]))

# TotalAmount = 500
# ei = 'hijack'
# with open("Output.txt", "a") as text_file:
#     text_file.write("Purchase Amount: {} {}\n".format(ei, TotalAmount))

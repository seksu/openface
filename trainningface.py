import os
import glob

dataset = glob.glob("test_image/")

temp = os.system("./batch-represent/main.lua -outDir demos/lab509_features -data demos/lab509_aligned")

print("temp = " + str(temp))

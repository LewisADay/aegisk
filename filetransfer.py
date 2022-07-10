
import os
import subprocess

wanted_string = "Forrester"
dir = "./results"
directory = os.listdir(dir)

output_dir = "./results-copy/"

for fname in directory:
    if wanted_string in fname:
        os.system(f"cp {dir + os.sep + fname} {output_dir}")


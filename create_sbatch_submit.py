import subprocess
import os
import copy

out_dir = "submission_Group1"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

with open('sbatch_template.sh', 'r') as file1:
    Lines = file1.readlines()

command = []
Header = []
flag = 0

for line in Lines:
    if not flag:
        Header.append(line)
    if flag:
        command.append(line.replace("--","").replace(" \\\n",""))
    if "# RUN PROGRAM" in line:
        flag=1

command_dict = {}
for cm in command:
    pos = cm.find(" ")
    command_dict[cm[:pos]] = cm[pos+1:]

Tasks = [
    {},
    {"precision":"16"},
    {"optimizer":"Adam","optimizer_hp":"1e-3 1e-5"},
    {"scheduler":"Cosine","scheduler_hp":"50 1e-4"}
]
for i,task in enumerate(Tasks):
    dd = copy.deepcopy(command_dict)
    dd.update(task)
    # add head and tail
    new_command = []
    for key,val in dd.items():
        if "python" in key:
            new_command.append(key+" "+val+" \\\n")
            continue
        new_command.append("--"+key+" "+val+" \\\n")

    with open(os.path.join(out_dir,"task-{}.sh".format(i)),"w") as ofile:
        ofile.writelines(Header + new_command)

for shfile in os.listdir(out_dir):
    subprocess.run(["sbatch",os.path.join(out_dir,shfile)])

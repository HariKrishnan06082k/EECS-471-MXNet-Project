#!/bin/bash

if [ "$#" -ne 0 ]; then
        echo "Usage: /scratch/eecs471f23_class_root/eecs471f23_class/$USER/final_project/submit_code.sh"
        exit 1
fi

chmod 700 new-forward.cuh
cp -f new-forward.cuh /scratch/eecs471f23_class_root/eecs471f23_class/all_sub/$USER/final_project/$USER.cuh
setfacl -m u:"joydong":rwx /scratch/eecs471f23_class_root/eecs471f23_class/all_sub/$USER/final_project/$USER.cuh
setfacl -m u:"mgmii":rwx /scratch/eecs471f23_class_root/eecs471f23_class/all_sub/$USER/final_project/$USER.cuh
setfacl -m u:"vtenishe":rwx /scratch/eecs471f23_class_root/eecs471f23_class/all_sub/$USER/final_project/$USER.cuh

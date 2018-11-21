# Classification of Diabetic Retinopathy

Project work for the course ELEC-E8739 - AI in health technologies

Authors:

Anton Mattsson
Janina Niiranen
Johanna Vikkula

### Notes:

#### Working on the server:

Connect using ```ssh username@taito-gpu.csc.fi ```  

Use the ```$WRKDIR``` folder. The git repository is located there.

If you run **heavy jobs**, define the filename in ```run.sh``` (now generator_test.py defined there) and run ```sbatch run.sh``` (this activates also Python 3 and tensorflow). Outputs are saved to ```output.txt``` file and errors to ```errors.t``` file. To investigate output/error file, run ```vim output.txt``` or ```vim errors.t```. Close the opened file by clicking ```esc```, then ```:q``` and finally ```enter```.

If you run regular jobs, run first these three commands (to activate Python 3 and tensorflow):
```module purge```  
```module load python-env/3.5.3 cuda/9.0 cudnn/7.0-cuda9```  
```export PYTHONPATH=$USERAPPL/tensorflow.1.11.0-py35/lib/python3.5/site-packages```  

UPDATE: These commands are now in server_setup.sh script in the working directory, so you can just run  
```./server_setup.sh```

and then run ```python filename.py```.

**Git push is not working atm!** So to change something, commit on your own machine and then pull to the server.

#### Setting up a small dataset for own testing:

There is a 1000 image subset (1.1 GB) in the folder data/train_samples

Run the following commands in the project main folder:

1. Copy the images and labels from the server to your own data folder:

```scp username@taito.csc.fi:/wrk/student083/diabetic_retinopathy/data/train_samples/*  data/train/```  
```scp username@taito.csc.fi:/wrk/student083/diabetic_retinopathy/data/trainLabels.csv  data/```

2. Create a list of the images contained: ``` ls data/train > train_list.txt```



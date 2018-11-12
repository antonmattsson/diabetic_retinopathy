# Classification of Diabetic Retinopathy

Project work for the course ELEC-E8739 - AI in health technologies

Authors:

Anton Mattsson
Janina Niiranen
Johanna Vikkula

### Notes:

#### Working on the server:

Connect using ```ssh username@taito.csc.fi ```  

Use the ```$WRKDIR``` folder. The git repository is located there.

To activate Python 3, you need to run ```source ~/.bashrc_profile``` at the beginning of every session.

**Git push is not working atm!** So to change something, commit on your own machine and then pull to the server.

#### Setting up a small dataset for own testing:

There is a 1000 image subset (1.1 GB) in the folder data/train_samples

Run the following commands in the project main folder:

1. Copy the images and labels from the server to your own data folder:

```scp username@taito.csc.fi:/wrk/student083/diabetic_retinopathy/data/train_samples/*  data/train/```  
```scp username@taito.csc.fi:/wrk/student083/diabetic_retinopathy/data/trainLabels.csv  data/```

2. Create a list of the images contained: ``` ls data/train > train_list.txt```



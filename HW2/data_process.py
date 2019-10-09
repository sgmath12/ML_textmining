import csv
import os

def write_csv_file(target_path,data_path,label_idx,data_split):    
    for filename in os.listdir(data_path):
            with open(os.path.join(data_path,filename)) as f:
                t = f.read()
                with open(target_path+"/"+data_split+"_log.csv","a") as out_file:
                    writer = csv.writer(out_file)
                    writer.writerow((t,label_idx))

                
def make_csv_file_from_rawtext():
    '''

    Return:
        Make CSV file, "sentence , label"
        No return
    '''
    data_split = ["train","test"]
    for k in data_split:
        new_path = "./data/"+k
        with open("./data/"+k+"_log.csv","w") as out_file:
            writer = csv.writer(out_file)
            writer.writerow(("text","label"))
        for label_idx,foldername in enumerate(os.listdir(new_path)):
            write_csv_file("./data",os.path.join(new_path,foldername),label_idx,k)
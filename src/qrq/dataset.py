import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5Config, T5Model
from transformers import T5Tokenizer,T5ForConditionalGeneration
import pandas as pd
import json
import os


class IntentDataset(Dataset):
    """TaskMaster 2 dataset to intent"""
    def __init__(self, dataset_folder, sample=False):
        """
        Args:
            json_data_file (string):  Path to the dataset file. Should contain a "data"
                                    and "ontology" folder
        """
        # Sample parameter makes dataloader faster by just loading first 10 lines/data points
        if sample:
            fname = 'Q2ClassDataset1.txt'      
        else:
            fname = 'Q2ClassDataset.txt'

        if not os.path.exists(dataset_folder+"/" + fname):
            print("Creating file", fname)
            annotations = set()

            for filename in os.listdir(dataset_folder+"/ontology/"):
                if not ".json" in filename:
                    continue
                f = open(dataset_folder+"/ontology/"+filename,"r")
                ontology = f.read()
                f.close()
                ontology = json.loads(ontology)

                for key in ontology.keys():
                    for annotation_set in ontology[key]:
                        annotations.add(annotation_set['prefix'])
                    for annotation in annotation_set["annotations"]:
                        annotation = annotation.split(".")
                        for annotation_segment in annotation:
                            if annotation_segment == " " or annotation_segment.find("_detail") > -1:
                                continue
                            annotations.add(annotation_segment)

            annotations = list(annotations)
            annotations.sort()

            for filename in os.listdir(dataset_folder+"/data/"):
                if not ".json" in filename:
                    continue
            
                f = open(dataset_folder+"/data/"+filename,"r")
                data = f.read()
                f.close()
                data = json.loads(data)

                for conversation in data:
                    for utterance in conversation['utterances']:
                        query = utterance['text']
                        query_annotations = [0] * len(annotations)
                        if not 'segments' in utterance.keys():
                            continue
                        for utterance_segment in utterance['segments']:
                            for annotation in utterance_segment['annotations']:
                                annotation = annotation["name"].split(".")
                                for annotation_segment in annotation:
                                    if annotation_segment == " " or annotation_segment.find("_detail") > -1:
                                        continue
                                    if annotation_segment in annotations:
                                        id = annotations.index(annotation_segment)
                                        query_annotations[id] = 1
                        output_string = ""
                        for i in query_annotations:
                            output_string += str(i)
                        f = open(dataset_folder+"/Q2ClassDataset.txt","a")
                        f.write(query+","+output_string+"\n")
                        f.close() 
        print("Reading from file...")
        f = open(dataset_folder+"/" + fname, "r")
        self.data = f.readlines()
        f.close()
        print("Tokenizing data...")
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')

    def preproc_(self, line):
        pair = line.split(",")
        temp = pair[-1]
        query = "".join(pair[:-1])
        annotations = []
        for num in temp:
            if num >= "0" and num <= "1":
                annotations.append(int(num))
        input_ids = self.tokenizer(query,return_tensors="pt").input_ids
        query = self.tokenizer.encode(query,return_tensors="pt",max_length=512,truncation=True)
        return query, annotations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        return self.preproc_(line)
    
    
class QRQ2Intent(Dataset):
    """TaskMaster 2 dataset to intent"""
    def __init__(self, dataset_folder, sample=False):
        """
        Args:
            json_data_file (string):  Path to the dataset file. Should contain a "data"
                                    and "ontology" folder
        """
        # Sample parameter makes dataloader faster by just loading first 10 lines/data points
        if sample:
            fname = 'QRQ2Intent1.txt'      
        else:
            fname = 'QRQ2Intent.txt'
        print(dataset_folder+"/"+fname)
        if not os.path.exists(dataset_folder+"/" + fname):
            annotations = set()

            for filename in os.listdir(dataset_folder+"/ontology/"):
                if not ".json" in filename:
                    continue
                f = open(dataset_folder+"/ontology/"+filename,"r")
                ontology = f.read()
                f.close()
                ontology = json.loads(ontology)

                for key in ontology.keys():
                    for annotation_set in ontology[key]:
                        annotations.add(annotation_set['prefix'])
                    for annotation in annotation_set["annotations"]:
                        annotation = annotation.split(".")
                        for annotation_segment in annotation:
                            if annotation_segment == " " or annotation_segment.find("_detail") > -1:
                                continue
                            annotations.add(annotation_segment)

            annotations = list(annotations)
            annotations.sort()

            for filename in os.listdir(dataset_folder+"/data/"):
                if not ".json" in filename:
                    continue
            
                f = open(dataset_folder+"/data/"+filename,"r")
                data = f.read()
                f.close()
                data = json.loads(data)

                total=0
                for conversation in data:
                  i = 0
                  while i < len(conversation['utterances']):
                    text = []
                    query_annotations = [0]*len(annotations)
                    speaker = conversation['utterances'][i]['speaker']
                    if speaker == 'ASSISTANT':
                      i+=1
                      continue
                    j=i
                    while j < len(conversation['utterances']) and conversation['utterances'][j]['speaker'] == 'USER':
                      text.append(conversation['utterances'][i]['text'])
                      if 'segments' in conversation['utterances'][j]:
                        utterance = conversation['utterances'][j]
                        for utterance_segment in utterance['segments']:
                            for annotation in utterance_segment['annotations']:
                                annotation = annotation["name"].split(".")
                                for annotation_segment in annotation:
                                    if annotation_segment == " " or annotation_segment.find("_detail") > -1:
                                        continue
                                    if annotation_segment in annotations:
                                        id = annotations.index(annotation_segment)
                                        query_annotations[id] = 1
                      j+=1
                    while j < len(conversation['utterances']) and conversation['utterances'][j]['speaker'] == 'ASSISTANT':
                      text.append(conversation['utterances'][j]['text'])
                      j+=1
                    enduser = 0
                    if j < len(conversation['utterances']) and conversation['utterances'][j]['speaker'] == 'USER':
                      enduser+=1
                      text.append(conversation['utterances'][j]['text'])
                      if 'segments' in conversation['utterances'][j]:
                        utterance = conversation['utterances'][j]
                        for utterance_segment in utterance['segments']:
                            for annotation in utterance_segment['annotations']:
                                annotation = annotation["name"].split(".")
                                for annotation_segment in annotation:
                                    if annotation_segment == " " or annotation_segment.find("_detail") > -1:
                                        continue
                                    if annotation_segment in annotations:
                                        id = annotations.index(annotation_segment)
                                        query_annotations[id] = 1
                      j+=1
                    if enduser>0:
                      query = " ".join(text)
                      output_string = ""
                      for k in query_annotations:
                          output_string += str(k)
                      if total < 10:
                        f = open(dataset_folder+"/QRQ2Intent1.txt","a")
                        f.write(query+","+output_string+"\n")
                        f.close()
                      total+=1
                      f = open(dataset_folder+"/QRQ2Intent.txt","a")
                      f.write(query+","+output_string+"\n")
                      f.close() 
                    i+=1

        f = open(dataset_folder+"/" + fname, "r")
        self.data = f.readlines()
        f.close()
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')

    def preproc_(self, data_pt):
        data_pt = data_pt.split(",")
        query = "".join(data_pt[:-1])
        input_ids = self.tokenizer(query,return_tensors="pt").input_ids
        query = self.tokenizer.encode(query,return_tensors="pt",max_length=512,truncation=True)
        temp = data_pt[-1]
        annotations = []
        for num in temp:
            if num >= "0" and num <= "1":
                annotations.append(int(num))
        return query, annotations
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query, labels = self.preproc_(self.data[idx])
        return query, labels
    
    
class QQ2Intent(Dataset):
    """TaskMaster 2 dataset to intent"""
    def __init__(self, dataset_folder, sample=False):
        """
        Args:
            json_data_file (string):  Path to the dataset file. Should contain a "data"
                                    and "ontology" folder
        """
        # Sample parameter makes dataloader faster by just loading first 10 lines/data points
        if sample:
            fname = 'QQ2Intent1.txt'      
        else:
            fname = 'QQ2Intent.txt'
        print(dataset_folder+"/"+fname)
        if not os.path.exists(dataset_folder+"/" + fname):
            annotations = set()

            for filename in os.listdir(dataset_folder+"/ontology/"):
                if not ".json" in filename:
                    continue
                f = open(dataset_folder+"/ontology/"+filename,"r")
                ontology = f.read()
                f.close()
                ontology = json.loads(ontology)

                for key in ontology.keys():
                    for annotation_set in ontology[key]:
                        annotations.add(annotation_set['prefix'])
                    for annotation in annotation_set["annotations"]:
                        annotation = annotation.split(".")
                        for annotation_segment in annotation:
                            if annotation_segment == " " or annotation_segment.find("_detail") > -1:
                                continue
                            annotations.add(annotation_segment)

            annotations = list(annotations)
            annotations.sort()
            total = 0
            for filename in os.listdir(dataset_folder+"/data/"):
                if not ".json" in filename:
                    continue
            
                f = open(dataset_folder+"/data/"+filename,"r")
                data = f.read()
                f.close()
                data = json.loads(data)

                for conversation in data:
                  i = 0
                  while i < len(conversation['utterances']):
                    text = []
                    query_annotations = [0]*len(annotations)
                    speaker = conversation['utterances'][i]['speaker']
                    if speaker == 'ASSISTANT':
                      i+=1
                      continue
                    j=i
                    while j < len(conversation['utterances']) and conversation['utterances'][j]['speaker'] == 'USER':
                      text.append(conversation['utterances'][i]['text'])
                      if 'segments' in conversation['utterances'][j]:
                        utterance = conversation['utterances'][j]
                        for utterance_segment in utterance['segments']:
                            for annotation in utterance_segment['annotations']:
                                annotation = annotation["name"].split(".")
                                for annotation_segment in annotation:
                                    if annotation_segment == " " or annotation_segment.find("_detail") > -1:
                                        continue
                                    if annotation_segment in annotations:
                                        id = annotations.index(annotation_segment)
                                        query_annotations[id] = 1
                      j+=1
                    while j < len(conversation['utterances']) and conversation['utterances'][j]['speaker'] == 'ASSISTANT':
                      j+=1
                    enduser = 0
                    if j < len(conversation['utterances']) and conversation['utterances'][j]['speaker'] == 'USER':
                      enduser+=1
                      text.append(conversation['utterances'][j]['text'])
                      if 'segments' in conversation['utterances'][j]:
                        utterance = conversation['utterances'][j]
                        for utterance_segment in utterance['segments']:
                            for annotation in utterance_segment['annotations']:
                                annotation = annotation["name"].split(".")
                                for annotation_segment in annotation:
                                    if annotation_segment == " " or annotation_segment.find("_detail") > -1:
                                        continue
                                    if annotation_segment in annotations:
                                        id = annotations.index(annotation_segment)
                                        query_annotations[id] = 1
                      j+=1
                    if enduser>0:
                      query = " ".join(text)
                      output_string = ""
                      for k in query_annotations:
                          output_string += str(k)
                      if total < 10:
                        f = open(dataset_folder+"/QQ2Intent1.txt","a")
                        f.write(query+","+output_string+"\n")
                        f.close()
                      total+=1
                      f = open(dataset_folder+"/QQ2Intent.txt","a")
                      f.write(query+","+output_string+"\n")
                      f.close() 
                    i+=1

        f = open(dataset_folder+"/" + fname, "r")
        data = f.readlines()
        f.close()
        self.data = []
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')

    def _preproc(self, data_pt):
        data_pt = data_pt.split(",")
        query = "".join(data_pt[:-1])
        input_ids = self.tokenizer(query,return_tensors="pt").input_ids
        query = self.tokenizer.encode(query,return_tensors="pt",max_length=512,truncation=True)
        temp = data_pt[-1]
        annotations = []
        for num in temp:
            if num >= "0" and num <= "1":
                annotations.append(int(num))
        return query, annotations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query, labels = self.data[idx]
        return query, labels    



def padding_collate_fn(batch):
    batch_size = len(batch)
    # Get the max length of an input sequence (each item is an input seq and a label)
    max_length = max([len(item[0][0]) for item in batch])
    # Its weird but the first item is a tuple not a tensor. 
    # print('data',[item[0][0].shape for item in batch])
    batch_seq = torch.zeros((batch_size, max_length), dtype=torch.float)
    batch_label = []

    for i, sequence in enumerate(batch):
        batch_seq[i, :sequence[0].shape[1]] = sequence[0].squeeze()
        batch_label += [sequence[1]]
    return batch_seq.long(), torch.tensor(batch_label)

def padding_collate_fn_enc_dec(batch):
    batch_size = len(batch)
    # Get the max length of an input sequence (each item is an input seq and a label)
    max_length1 = max([len(item[0][0]) for item in batch])
    max_length2 = max([len(item[1][0]) for item in batch])
    # Its weird but the first item is a tuple not a tensor. 
    # print('data',[item[0][0].shape for item in batch])
    inp_seq = torch.zeros((batch_size, max_length1), dtype=torch.float)
    out_seq = torch.zeros((batch_size, max_length2), dtype=torch.float)

    for i, sequence in enumerate(batch):
        inp_seq[i, :sequence[0].shape[1]] = sequence[0].squeeze()
        out_seq[i, :sequence[1].shape[1]] = sequence[1].squeeze()
    return inp_seq.long(), out_seq.long()



class QR_dataset(Dataset):

    def __init__(self, dataset_folder):
        self.lst_query=[]
        self.lst_response=[]
        dataset = pd.DataFrame(columns=['index','speaker', 'text', "intent"])
        pth= dataset_folder+"/data/" #"/content/drive/MyDrive/data"
        for filename in os.listdir(pth):
            if not ".json" in filename:
                continue

            with open(dataset_folder + "/data/"+filename) as json_file:
                data = json.load(json_file)

                for i in data:
                    utter=i[ "utterances"]
                    j=0
                    while j < len(utter):
                        speaker = list(utter[j].values())[1]
                        if speaker=="ASSISTANT":
                          j+=1
                          continue
                        text = list(utter[j].values())[2]
                        query = ""
                        response = ""
                        if speaker == "USER":
                          query+=text
                        new_query = False
                        j+=1
                        while j < len(utter) and not new_query:
                          speaker = list(utter[j].values())[1]
                          text = list(utter[j].values())[2]
                          if response == "" and speaker == "USER":
                            query+=text
                          elif speaker == "USER":
                            new_query = True
                            continue
                          else:
                            response += text
                          j+=1
                        self.lst_query.append(query)
                        self.lst_response.append(response)

        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')

    def preproc_(self, text):
        input_ids = self.tokenizer(text,return_tensors="pt").input_ids
        text = self.tokenizer.encode(text,return_tensors="pt", max_length=512, truncation=True)
        return text

    def __len__(self):
        return len(self.lst_query)

    def __getitem__(self, idx):
        response = self.lst_response[idx]
        query = self.lst_query[idx]
        return self.preproc_(query) , self.preproc_(response)
    
class RQ_dataset(Dataset):

    def __init__(self, dataset_folder):
        self.lst_query=[]
        self.lst_response=[]
        dataset = pd.DataFrame(columns=['index','speaker', 'text', "intent"])
        pth= dataset_folder+"/data/" #"/content/drive/MyDrive/data"
        for filename in os.listdir(pth):
            if not ".json" in filename:
                continue

            with open(dataset_folder + "/data/"+filename) as json_file:
                data = json.load(json_file)
                for i in data:
                    utter=i[ "utterances"]
                    j=0
                    while j < len(utter):
                        speaker = list(utter[j].values())[1]
                        if speaker=="USER":
                          j+=1
                          continue
                        text = list(utter[j].values())[2]
                        query = ""
                        response = ""
                        if speaker == "ASSISTANT":
                          query+=text
                        new_query = False
                        j+=1
                        while j < len(utter) and not new_query:
                          speaker = list(utter[j].values())[1]
                          text = list(utter[j].values())[2]
                          if response == "" and speaker == "ASSISTANT":
                            query+=text
                          elif speaker == "ASSISTANT":
                            new_query = True
                            continue
                          else:
                            response += text
                          j+=1
                        self.lst_query.append(query)
                        self.lst_response.append(response)

        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')

    def preproc_(self, text):
        input_ids = self.tokenizer(text,return_tensors="pt").input_ids
        text = self.tokenizer.encode(text,return_tensors="pt", max_length=512, truncation=True)
        return text

    def __len__(self):
        return len(self.lst_query)

    def __getitem__(self, idx):
        response = self.lst_response[idx]
        query = self.lst_query[idx]
        return self.preproc_(query) , self.preproc_(response)

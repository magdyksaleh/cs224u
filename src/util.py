import json
import collections

with open('../data/drop_dataset/drop_dataset_dev.json') as json_file:  
    data = json.load(json_file)
    res = []
    for k in data:
      for elem in data[k]['qa_pairs']:
        question = elem['question']
        answer_type = "" 
        for key in elem['answer']:
          if (key == 'number') and (len(elem['answer'][key]) != 0):
              answer_type = "number"
              break

          if (key == 'spans') and (len(elem['answer'][key]) != 0):
            answer_type = "spans"
            break
          answer_type = 'date'  
        res.append(answer_type)

print(collections.Counter(res))
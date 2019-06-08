import json
import collections
from typing import Dict

import numpy
import torch
from overrides import overrides

from allennlp.data.fields.field import Field


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

with open('../data/drop_dataset/drop_dataset_dev.json') as json_file:  
    data = json.load(json_file)
    res = []
    for k in data:
      for elem in data[k]['qa_pairs']:
        question = elem['question']
        answer_type = "" 
        for key in elem['answer']:
          if (key == 'number') and (len(elem['answer'][key]) != 0):
              if int(float(elem['answer'][key])) > 9:
                answer_type = "arithmetic"
              else:
                answer_type = "count"
              break

          if (key == 'spans') and (len(elem['answer'][key]) != 0):
            answer_type = "spans"
            break
          answer_type = 'date'  
        res.append(answer_type)

print(collections.Counter(res))




class ArrayField(Field[numpy.ndarray]):
    """
    A class representing an array, which could have arbitrary dimensions.
    A batch of these arrays are padded to the max dimension length in the batch
    for each dimension. Max_pad allows max padding to be limited.
    """
    def __init__(self,
                 array: numpy.ndarray,
                 padding_value: int = 0,
                 max_pad: int = 5,
                 dtype: numpy.dtype = numpy.float32) -> None:
        self.array = array
        self.padding_value = padding_value
        self.max_pad = max_pad
        self.dtype = dtype

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {"dimension_" + str(i): shape
                for i, shape in enumerate(self.array.shape)}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        max_shape = [max(padding_lengths["dimension_{}".format(i)], self.max_pad)
                     for i in range(len(padding_lengths))]
        # Convert explicitly to an ndarray just in case it's an scalar
        # (it'd end up not being an ndarray otherwise).
        # Also, the explicit dtype declaration for `asarray` is necessary for scalars.
        return_array = numpy.asarray(numpy.ones(max_shape, dtype=self.dtype) * self.padding_value,
                                     dtype=self.dtype)

        # If the tensor has a different shape from the largest tensor, pad dimensions with zeros to
        # form the right shaped list of slices for insertion into the final tensor.
        slicing_shape = list(self.array.shape)
        if len(self.array.shape) < len(max_shape):
            slicing_shape = slicing_shape + [0 for _ in range(len(max_shape) - len(self.array.shape))]
        
        slices = tuple([slice(0, x) for x in slicing_shape])
        return_array[slices] = self.array
        return_array = return_array[:self.max_pad]
        tensor = torch.from_numpy(return_array)
        return tensor

    @overrides
    def empty_field(self):  # pylint: disable=no-self-use
        # Pass the padding_value, so that any outer field, e.g., `ListField[ArrayField]` uses the
        # same padding_value in the padded ArrayFields
        return ArrayField(numpy.array([], dtype=self.dtype),
                          padding_value=self.padding_value,
                          dtype=self.dtype)

    def __str__(self) -> str:
        return f"ArrayField with shape: {self.array.shape} and dtype: {self.dtype}."
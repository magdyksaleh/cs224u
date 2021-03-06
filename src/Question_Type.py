#!/usr/bin/env python
# coding: utf-8

# ### Predict question type 

# In[1]:


from typing import Iterator, List, Dict
import torch
from util import ArrayField
import torch.optim as optim
import json
import string
import numpy as np
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField, LabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.dataset_readers.reading_comprehension.drop import DropReader
from allennlp.data.tokenizers import Token, WordTokenizer

from allennlp.data.dataset_readers.reading_comprehension.util import (IGNORED_TOKENS,
                                                                      STRIPPED_CHARACTERS,
                                                                      make_reading_comprehension_instance,
                                                                      split_tokens_by_hyphen)

from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper 
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper, Seq2VecEncoder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor

from allennlp.common import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.predictors import Predictor
from overrides import overrides

torch.manual_seed(1)


# In[2]:


WORD_NUMBER_MAP = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                   "five": 5, "six": 6, "seven": 7, "eight": 8,
                   "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
                   "thirteen": 13, "fourteen": 14, "fifteen": 15,
                   "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}


# In[3]:


class DropTypeDatasetReader(DatasetReader):
    """
    DatasetReader
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._tokenizer = WordTokenizer()
        
    def text_to_instance(self, passage_numbers: List[int], tokens_q: List[Token], tag: List[str] = None) -> Instance:
        question_field = TextField(tokens_q, self.token_indexers) 
        number_field = ArrayField(np.array(passage_numbers))
        fields = {"question": question_field, "numbers_in_passage": number_field}

        if tag:
            label_field = LabelField(tag)
            fields["label"] = label_field
        return Instance(fields)
    
    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as json_file:  
            data = json.load(json_file)
            for k in data:
                passage = data[k]['passage']
                passage_tokens =  split_tokens_by_hyphen(self._tokenizer.tokenize(passage))
                passage_numbers = self.get_numbers_in_passage(passage_tokens)
                for elem in data[k]['qa_pairs']:
                    question = elem['question']
                    answer_type = "" 
                    for key in elem['answer']:
                        if (key == 'number') and (len(elem['answer'][key]) != 0):
                            if float(elem['answer']['number']) > 9:
                                answer_type = "arithmetic"
                            else: 
                                answer_type = "count"
                            break

                        if (key == 'spans') and (len(elem['answer'][key]) != 0):
                            answer_type = "spans"
                            break

                        answer_type = 'date'
                    yield self.text_to_instance(passage_numbers, [Token(word) for word in question], answer_type)

    
    def get_numbers_in_passage(self, passage_tokens):
        """
        Returns list of numbers in the passage
        """
        numbers_in_passage = []
        number_indices = []
        for token_index, token in enumerate(passage_tokens):
            number = self.convert_word_to_number(token.text)
            if number is not None:
                numbers_in_passage.append(number)
                number_indices.append(token_index)

        return numbers_in_passage
                    
    def convert_word_to_number(self, word: str, try_to_include_more_numbers=False):
        """
        Currently we only support limited types of conversion.
        """
        if try_to_include_more_numbers:
            # strip all punctuations from the sides of the word, except for the negative sign
            punctruations = string.punctuation.replace('-', '')
            word = word.strip(punctruations)
            # some words may contain the comma as deliminator
            word = word.replace(",", "")
            # word2num will convert hundred, thousand ... to number, but we skip it.
            if word in ["hundred", "thousand", "million", "billion", "trillion"]:
                return None
            try:
                number = word_to_num(word)
            except ValueError:
                try:
                    number = int(word)
                except ValueError:
                    try:
                        number = float(word)
                    except ValueError:
                        number = None
            return number
        else:
            no_comma_word = word.replace(",", "")
            if no_comma_word in WORD_NUMBER_MAP:
                number = WORD_NUMBER_MAP[no_comma_word]
            else:
                try:
                    number = int(no_comma_word)
                except ValueError:
                    number = None
            return number


# In[4]:


reader = DropTypeDatasetReader()


# In[5]:


data_train = reader.read('../data/drop_dataset/drop_dataset_train.json')
data_dev = reader.read('../data/drop_dataset/drop_dataset_dev.json')


# In[6]:


vocab = Vocabulary.from_instances(data_train + data_dev)


# In[7]:


# Model in AllenNLP represents a model that is trained.
class LstmClassifier(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 vocab: Vocabulary,
                 num_pad_size=10) -> None:
        super().__init__(vocab)
        # We need the embeddings to convert word IDs to their vector representations
        self.word_embeddings = word_embeddings

        # Seq2VecEncoder is a neural network abstraction that takes a sequence of something
        # (usually a sequence of embedded word vectors), processes it, and returns it as a single
        # vector. Oftentimes, this is an RNN-based architecture (e.g., LSTM or GRU), but
        # AllenNLP also supports CNNs and other simple architectures (for example,
        # just averaging over the input vectors).
        self.encoder = encoder
        self.num_pad_size = num_pad_size
        # After converting a sequence of vectors to a single vector, we feed it into
        # a fully-connected linear layer to reduce the dimension to the total number of labels.
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim()+self.num_pad_size,
                                          out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()

        # We use the cross-entropy loss because this is a classification task.
        # Note that PyTorch's CrossEntropyLoss combines softmax and log likelihood loss,
        # which makes it unnecessary to add a separate softmax layer.
        self.loss_function = torch.nn.CrossEntropyLoss()

    # Instances are fed to forward after batching.
    # Fields are passed through arguments with the same name.
    def forward(self,
                question: Dict[str, torch.Tensor],
                numbers_in_passage: List[int],
                label: torch.Tensor = None) -> torch.Tensor:
        # In deep NLP, when sequences of tensors in different lengths are batched together,
        # shorter sequences get padded with zeros to make them of equal length.
        # Masking is the process to ignore extra zeros added by padding
        mask = get_text_field_mask(question)

        # Forward pass
        embeddings = self.word_embeddings(question)
        encoder_out = self.encoder(embeddings, mask)
#         print(encoder_out.shape)
#         print(numbers_in_passage.shape)
        inputs = torch.cat((encoder_out, numbers_in_passage), 1)
        logits = self.hidden2tag(inputs)

        # In AllenNLP, the output of forward() is a dictionary.
        # Your output dictionary must contain a "loss" key for your model to be trained.
        output = {"logits": logits}
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = self.loss_function(logits, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


# In[8]:


EMBEDDING_DIM = 6
HIDDEN_DIM = 6
token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})


# In[9]:


lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
model = LstmClassifier(word_embeddings, lstm, vocab, num_pad_size=5)


# In[10]:


optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
iterator = BucketIterator(batch_size=64, sorting_keys=[("question", "num_tokens")])
iterator.index_with(vocab)
trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=data_train,
                  validation_dataset=data_dev,
                  patience=10,
                  num_epochs=40,
                  cuda_device=-1)
trainer.train()


# In[ ]:


# You need to name your predictor and register so that `allennlp` command can recognize it
# Note that you need to use "@Predictor.register", not "@Model.register"!
@Predictor.register("sentence_classifier_predictor2")
class SentenceClassifierPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language='en_core_web_sm', pos_tags=True)

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"question" : question})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["question"]
        tokens = self._tokenizer.split_words(question)
        return self._dataset_reader.text_to_instance([str(t) for t in tokens])


# In[17]:


tokens = ['This', 'is', 'the', 'best', 'movie', 'ever', '!']
predictor = SentenceClassifierPredictor(model, dataset_reader=reader)
logits = predictor.predict(tokens)['logits']
label_id = np.argmax(logits)

print(model.vocab.get_token_from_index(label_id, 'labels'))


# In[ ]:





# In[ ]:





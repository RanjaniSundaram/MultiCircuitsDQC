'''
Manage all the inputs and the outputs. Make running experiments and generating plots easy.
'''
from dataclasses import dataclass  # for python 3.7+
from typing import List
import json


@dataclass
class Default:
    '''some default configuration. start with a toy configuration.
    '''
    NODE     = 10    # number of nodes
    DEGREE   = 2     # avg degree for a node, used for creating a random graph. Let max degree be 2 times avg.
    CAPACITY = 2     # avg capacity for an edge. Let max capacity be 2 times avg.
    SD_PAIR  = 3     # number of source-destination pairs
    BSM_RATE = 0.9   # BSM success rate
    METHODS  = ['our', 'sigcomm']


@dataclass
class Input:
    '''Encapusulate the input of the routing algorithm
    '''
    methods: List[str]
    experiment_num: int = 0  # experiment num
    node: int       = Default.NODE
    degree: int     = Default.DEGREE
    capacity: float = Default.CAPACITY
    sd_pair: int    = Default.SD_PAIR
    bsm_rate: float = Default.BSM_RATE

    def to_json_str(self):
        '''return json formated string
        Return:
            str
        '''
        inputdict = {
            'experiment_num': self.experiment_num,
            'node': self.node,
            'degree': self.degree,
            'capacity': self.capacity,
            'sd_pair': self.sd_pair,
            'bsm_rate': self.bsm_rate,
            'methods': self.methods
        }
        return json.dumps(inputdict)

    @classmethod
    def from_json_str(cls, json_str):
        '''create an Input object from json_str
        Args:
            json_str -- str
        Return:
            Input
        '''
        pass # TODO


@dataclass
class Output:
    '''Encapsulate the output of the routing algorithm (the evaluation metric)
    '''
    method: str
    throughput: float

    def to_json_str(self):
        '''return json formatec string
        Return:
            str
        '''
        outputdic = {
            'method': self.method,
            'throughput': self.throughput
        }
        return json.dumps(outputdic)

    @classmethod
    def from_json_str(cls, json_str):
        '''return Output object from json_str
        Args:
            json_str -- str
        Return:
            Output
        '''
        pass # TODO


if __name__ == '__main__':
    inpt = Input(Default.METHODS)
    print(inpt.to_json_str())

    output = Output('our', 10)
    print(output.to_json_str())

    # example experiment results: https://github.com/caitaozhan/deeplearning-localization/blob/master/result/11.14/log


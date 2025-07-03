import pandas as pd 
import torch
import os 
from torch.utils.data import DataLoader





def main(): 

    # clear the GPU
    torch.cuda.empty_cache()
    
    # TODO get defualts

    # TODO generate seed

    # TODO make directory results in some file 

    # TODO get data

    # TODO use models for data

    # TODO apply loss/optimizer 

    # TODO make predictions 
    
if __name__ == '__main__':
    begin = time()  
    run = main()  
    end = time()  
    print(f"Total running time: {round(end - begin, 2)}")  
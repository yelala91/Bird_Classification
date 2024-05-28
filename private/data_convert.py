from data_read import txt_convert
import os

if __name__ == '__main__':
    CUB_path = '.' + os.sep + 'dataset' + os.sep + 'CUB_200_2011'
    txt_convert(CUB_path)
    print(f'Have changed the \'\\\' or \'/\' to {os.sep}')
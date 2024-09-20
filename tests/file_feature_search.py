from endf_parserpy import EndfParserCpp, EndfDict, update_directory, EndfParser
import os
from copy import deepcopy

parser = EndfParserCpp()

endf_path = "/home/gschnabel/bigdata/nuclibs/jeff33"
files = os.listdir(endf_path)
filepaths = [os.path.join(endf_path, f) for f in files]

for f, p in zip(files, filepaths): 
    # print(f"testing {f}") 
    try:
        endf_dict = parser.parsefile(p, include=(4,))
        for mt, cont in endf_dict[4].items():
            if mt != 2:
                continue
            if cont['LTT'] == 1 and cont['LI'] == 0:
                print(f'{f} - {mt}')
    except Exception:
        print("failed")


parser = EndfParser()
file = '1-H-2g.endf'
filepath = os.path.join(endf_path, file)
endf_dict = parser.parsefile(filepath)

endf_dict[4][2]


new_dict = EndfDict()
new_dict['1/451'] = deepcopy(endf_dict[1][451])
new_dict['3/2'] = deepcopy(endf_dict[3][2])
new_dict['4/2'] = deepcopy(endf_dict[4][2])
update_directory(new_dict, parser)

parser.writefile(os.path.join('data', f'jeff33_{file}_mf4_mt2.endf'), new_dict)

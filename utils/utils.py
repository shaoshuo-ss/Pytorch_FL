# -*- coding: UTF-8 -*-
def printf(content, path):
    with open(path, 'a+') as f:
        print(content, file=f)

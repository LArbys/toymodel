class config_loader(object):
    def __init__(self,file_):
        data = None
        with open(file_,'r') as f:
            data = f.read()
        data = data.split("\n")
        for line in data:
            SS = "self." + line
            print SS
            exec(SS)
        

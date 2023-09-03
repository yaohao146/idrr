from tqdm import tqdm
labtoplist = []
labseclist = []
labconnlist = []
with open("./corpus/pdtb3/train.txt", 'r', encoding='UTF-8') as f:
    for line in tqdm(f):
        lin = line.strip()
        if not lin:
            continue

        labels1, labels2, arg1, arg2 = [_.strip() for _ in lin.split('|||')]
        labels1, labels2 = eval(labels1), eval(labels2)
        if labels1[1] not in labseclist:
            labseclist.append(labels1[1])
            with open("./corpus/pdtb3/data/class_n.txt", 'a', encoding='UTF-8') as f1:
                f1.write(labels1[1]+"\n")
                f1.close()

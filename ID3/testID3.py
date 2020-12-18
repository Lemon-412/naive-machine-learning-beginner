from ID3 import ID3

if __name__ == '__main__':
    x, y = [], []
    label = ["age", "prescript", "astigmatic", "tear_rate"]
    with open("lenses.txt", "r") as FILE:
        for line in FILE.readlines():
            x.append(line.strip().split("\t"))
            y.append(x[-1].pop())
    id3 = ID3(x, y, label)
    id3.generate_tree()
    print(id3)
    # print(id3.inference())

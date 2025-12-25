with open('features_for_model_1.csv', encoding='utf-8') as f:
    header = f.readline()
    ncol = len(header.strip().split(','))
    for i, line in enumerate(f, 2):
        n = len(line.strip().split(','))
        if n != ncol:
            print(f'Line {i}: {n} columns <-- ERROR')

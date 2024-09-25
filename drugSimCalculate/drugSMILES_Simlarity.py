import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity


def drugSMILES_Similarity(fps1, fps2, resultFormat):
    if resultFormat == 'listFormat':
        # 计算drug SMILE similarity score->返回idx1,idx2,score格式
        simlarity_list = []
        for i in range(len(fps1)):
            for j in range(len(fps2)):
                if fps1[i] is None or fps2[j] is None:
                    simlarity_list.append([i, j, 0.0])  # 如果有任何一个分子的指纹为 None，则相似度为 0
                else:
                    simlarity_list.append([i, j, round(TanimotoSimilarity(fps1[i], fps2[j]), 6)])
        return pd.DataFrame(simlarity_list, columns=['idx1', 'idx2', 'score'])
    elif resultFormat == 'matrixFormat':
        # 计算drug SMILE similarity score->返回list1 * list2 格式
        similarity_matrix = pd.DataFrame(index=range(10), columns=range(13))
        for i in range(len(fps1)):
            for j in range(len(fps2)):
                if fps1[i] is None or fps2[j] is None:
                    similarity_matrix.iloc[i, j] = 0.0  # 如果有任何一个分子的指纹为 None，则相似度为 0
                else:
                    similarity_matrix.iloc[i, j] = round(TanimotoSimilarity(fps1[i], fps2[j]), 6)
        return similarity_matrix
    else:
        pass

# 读取第1个文件中drug的SMILE list
df1 = pd.read_csv('./data/sample01.tsv', sep='\t')
smile_list1 = df1['SMILES']
# 读取第2个文件中drug的SMILE list
df2 = pd.read_csv('./data/sample02.tsv', sep='\t')
smile_list2 = df2['SMILES']
# 计算drug SMILE similarity score
# 创建空的列表来存储无法处理的 SMILES
failed_smiles = []
# step1转化为分子指纹(fingerprints)
fps1 = []
fps2 = []
for smile in smile_list1:
    mol = Chem.MolFromSmiles(smile)
    if mol is not None:
        fps1.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    else:
        failed_smiles.append(smile)
        fps1.append(None)
for smile in smile_list2:
    mol = Chem.MolFromSmiles(smile)
    if mol is not None:
        fps2.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    else:
        failed_smiles.append(smile)
        fps2.append(None)

print('failed smiles:', failed_smiles)
# 结果返回的格式 list or matrix
resultFormat = 'listFormat'
# resultFormat = 'matrixFormat'
result = drugSMILES_Similarity(fps1, fps2, resultFormat)
result.to_csv(f'./data/drugSim_with_{resultFormat}.tsv', sep='\t', index=False)
print('Finish...')

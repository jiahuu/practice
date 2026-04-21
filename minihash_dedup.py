from datasketch import MinHash, MinHashLSH
from datasets import load_dataset
from tqdm import tqdm

def get_ngrams(text, n = 3):
    return set(text[i:i+n] for i in range(len(text)-n+1))

def make_minhash(text, num_permute=128) :
    m = MinHash(num_permute)

    for ngram in get_ngrams(text):
        # 逐个把元素加入minhash中，计算指纹，但是x必须要是bytes类型
        # encode('utf8) 把字符串转为 bytes
        m.update(ngram.encode("utf8"))
    return m

# texts = [
#     "写一个快速排序算法，要求时间复杂度 O(nlogn)",
#     "实现快速排序，时间复杂度需要达到 O(nlogn)",  # 近似重复
#     "解释什么是二叉树",
#     "用 Python 实现二叉树的中序遍历",
#     "写一个快速排序，要求 O(nlogn) 时间复杂度",  # 近似重复
#     "Python 中如何读取 CSV 文件",
# ]

lsh = MinHashLSH(threshold=0.8, num_perm=128)

# minhashes = {}

# for i, text in enumerate(texts) :
#     m = make_minhash(text)
#     minhashes[i] = m
#     lsh.insert(f"doc_{i}", m)

# print("===重复检测结果===")
# for i, text in enumerate(texts):
#     # 找到相似的文本
#     result = lsh.query(minhashes[i])
#     dupes = [ int(r.split("_")[1])  for r in result if int(r.split("_")[1]) != i]
#     if dupes :
#         print(f"文本 {i} : {text[:30]}...")
#         print(f"与文本{dupes}近似重复")

datasets = load_dataset("yahma/alpaca-cleaned", split="train")
to_remove = set()

for i, item in enumerate(tqdm(datasets)):
    text = item["instruction"] + " " + item["output"]
    m = MinHash(num_perm=128)
    for ngram in get_ngrams(text):
        m.update(ngram.encode("utf8"))
    
    dupes = lsh.query(m)
    if dupes :
        to_remove.add(i)
    else:
        lsh.insert(f"doc_{i}", m)

print(f"初始数据量：{len(datasets)}")
print(f"检测到重复：{len(to_remove)}")
print(f"去重后：{len(datasets) - len(to_remove)}")
clean_dataset = datasets.select([ i for i in range(len(datasets)) if i not in to_remove])
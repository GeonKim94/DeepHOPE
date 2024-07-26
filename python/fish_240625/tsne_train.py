import pickle

target_dir = "190112_human_3_class_no_sampler"
base = "/data2/DW/181121_Lympho/lympho_classification/outs/" + target_dir

with open(base + "/features.pkl", "wb") as f:
    features = pickle.load(f)
with open(base + "/paths.pkl", "wb") as f:
    paths = pickle.load(f)
with open(base + "/preds.pkl", "wb") as f:
    preds = pickle.load(f)

print(pos[0])
paths = pos + neg

from scipy.io import loadmat
imgs = [loadmat(p)["data"].flatten() for p in paths]



from sklearn.manifold import TSNE
from multiprocessing import Pool
import pickle
def get_tsne(perp):      
    global imgs
    tsne = TSNE(n_components=2, random_state=0, perplexity=perp,
                init="pca", n_iter=1000)

    result = tsne.fit_transform(imgs)
    with open(base + "result_iter1000_per%d.pkl"%(perp), "wb") as f:
        pickle.dump(result, f)
    return "perp %d Done"%(perp)

perps = [15, 30, 60, 90]
pool = Pool(processes=len(perps))
print(pool.map(get_tsne, perps))


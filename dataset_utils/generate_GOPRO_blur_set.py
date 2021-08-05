import sys
sys.path.append("/".join(sys.path[0].split("/")[:-1]))

import pickle
import coco_utils
import custom_datasets
import random
import time
import pickle

random.seed(1332)

#ds = custom_datasets.GOPRO()
#ds = custom_datasets.RealBlur()
#ds = custom_datasets.REDS()
expandBoxes = True
auxBlur = False
ds = custom_datasets.GOPROSynth(sharpImages=False, blurredImages=True, expandBoxes=False, auxBlur = auxBlur)

baseDirectory = "/media/mosayed/Cache/datasets/GOPROSynth/estimatorDS"
baseDirectory = "/media/mosayed/Cache/datasets/GOPROSynth/normalRat2"
baseDirectory = "/media/mosayed/Cache/datasets/GOPROSynth/normalRat3"


start_time = time.perf_counter()
for i in range(len(ds)):
    if i > 7000:
        break

    image, targets, blur_dict = ds[i]
    
    if auxBlur:
        indexPrint = i + 21895
        blurDir = "auxBlur"
    else:
        indexPrint = i
        blurDir = "blur"

    if not auxBlur:
        with open(os.path.join(baseDirectory, "sharp/{:05d}.dat".format(indexPrint)), 'wb') as f:
            pickle.dump(targets, f)
            f.close()


    if expandBoxes:
        ds.imageInfos[i].expandBoxes()

    with open(os.path.join(baseDirectory,  blurDir + "/{:05d}.dat".format(indexPrint)), 'wb') as f:
        pickle.dump(ds.imageInfos[i].targets, f)
        f.close()

    
    image.save(os.path.join(baseDirectory, blurDir + "/{:05d}.png".format(indexPrint)))

    if not auxBlur:
        ds.imageInfos[i].cleanImage.save(os.path.join(baseDirectory, "sharp/{:05d}.png".format(indexPrint)))

    
    if i % 100 == 0:
        print(i, "Time for 100: ", str(round((time.perf_counter() - start_time)/60, 2)) + "mins", "Time per image: ", str(round((time.perf_counter() - start_time)/100, 2)) + "s")

        start_time = time.perf_counter()
    

print("here")


from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import imutils
from tqdm import tqdm

def main():
    path_fg_net = "../../dataset/wiki_crop/new_database/"

    images = [ f for f in listdir(path_fg_net) if isfile(join(path_fg_net,f)) ]

    for img in tqdm(images):
        if img[0] != "F" and img[0] != "R" and img[0] != "B":
            image = cv2.imread(path_fg_net+img)
            #F for flip
            cv2.imwrite(path_fg_net+"F"+img,cv2.flip(image,1))
            #R for rotate
            for angle in np.arange(-30, 30, 15):
                rotated = imutils.rotate_bound(image, angle)
                cv2.imwrite(path_fg_net+"R"+str(angle)+img,rotated)
            #B for blurred
            blurred = cv2.blur(image,(5,5))
            cv2.imwrite(path_fg_net+"B"+str(angle)+img,blurred)
    
if __name__ == '__main__':
    main()
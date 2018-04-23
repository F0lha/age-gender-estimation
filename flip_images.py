from os import listdir
from os.path import isfile, join
import cv2

def main():
    path_fg_net = "../../dataset/FGNET/images/"

    images = [ f for f in listdir(path_fg_net) if isfile(join(path_fg_net,f)) ]

    for img in images:
        if img[0] != "F":
            cv2.imwrite(path_fg_net+"F"+img,cv2.flip(cv2.imread(path_fg_net+img),1))
    
    
if __name__ == '__main__':
    main()
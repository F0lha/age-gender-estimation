import os
import argparse
import scipy.io
import numpy as np
import cv2
import face_recognition
from utils import get_meta
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="This script seperates the dataset into age folders",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="path to dataset folder")
    parser.add_argument("--db", "-m", type=str, required=True,
                        help="name of the mat file")
    parser.add_argument('--folders', dest='fold', action='store_true')
    parser.set_defaults(folders=False)
    args = parser.parse_args()
    return args

def main():
    
    args = get_args()
    
    parent_folder = args.input + "/"
    db = args.db
    
    newdb_path = parent_folder + "new_database/"
    
    mat_path = parent_folder + "{}.mat".format(db)
    
    #create folder for new dataset
    if not os.path.exists(newdb_path):
        os.makedirs(newdb_path)
        
    #create a folder for every age
    if args.folders:
        for i in range(101):
            new_folder_path = newdb_path + ('%03d' % i)
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path) 
            
    
    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(mat_path, db)
    
    
    for i in tqdm(range(len(full_path))):
        if face_score[i] < 1.0:
            continue

        if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
            continue

        if ~(0 <= age[i] <= 100):
            continue
            
        if np.isnan(gender[i]):
            continue
            
        
        image = face_recognition.load_image_file(parent_folder + full_path[i][0])
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) == 0:
            continue
        face_locations = face_locations[0]
        img = image[face_locations[0]:face_locations[2],face_locations[3]:face_locations[1],:]
        
        resized_image = cv2.resize(img, (224, 224)) 
        
        name_of_file = full_path[i][0].split('/')[-1].split('.')[0]
        
        new_name = name_of_file + "A" + str(age[i]) + ".jpg"
        
        
        if args.folders:
            cv2.imwrite(newdb_path + ('%03d' % age[i]) + "/" + new_name, resized_image)
        else:
            cv2.imwrite(newdb_path + new_name, resized_image)

if __name__ == '__main__':
    main()
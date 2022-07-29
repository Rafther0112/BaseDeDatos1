#%%
import numpy as np
import os, cv2
import csv
from skimage import img_as_float

path_test_non_demented = os.path.join("test", "NonDemented")
path_test_moderate_demented =  os.path.join("test" , "ModerateDemented")
path_test_very_mild_demented= os.path.join("test" , "VeryMildDemented")

path_train_non_demented = os.path.join("train", "NonDemented")
path_train_moderate_demented =  os.path.join( "train", "ModerateDemented")
path_train_very_mild_demented= os.path.join("train", "VeryMildDemented")

path_validation_non_demented = os.path.join( "validation", "NonDemented")
path_validation_very_mild_demented= os.path.join("validation", "VeryMildDemented")
path_validation_moderate_demented =  os.path.join("validation", "ModerateDemented")

#%%
nombres_imagenes_testeo_moderado = sorted(os.listdir(path_test_moderate_demented))
nombres_imagenes_testeo_no_demencia = sorted(os.listdir(path_test_non_demented))
nombres_imagenes_testeo_very_mild = sorted(os.listdir(path_test_very_mild_demented))

nombres_imagenes_entrenamiento_moderado = sorted(os.listdir(path_train_moderate_demented))
nombres_imagenes_entrenamiento_no_demencia = sorted(os.listdir(path_train_non_demented))
nombres_imagenes_entrenamiento_very_mild = sorted(os.listdir(path_train_very_mild_demented))

nombres_imagenes_validacion_moderado = sorted(os.listdir(path_validation_moderate_demented))
nombres_imagenes_validacion_no_demencia = sorted(os.listdir(path_validation_non_demented))
nombres_imagenes_validacion_very_mild = sorted(os.listdir(path_validation_very_mild_demented))

#%%
fields = ['Name', 'Label'] 

# name of csv file 
filename_train = "train_data.csv"
filename_validation = "valid_data.csv"
filename_test = "test_data.csv"
    
# writing to csv file 
with open(filename_train, 'w') as csvfile: 

    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
        
    csvwriter.writerow(fields) 
    
    for i in nombres_imagenes_entrenamiento_no_demencia:
        if i != ".DS_Store":
            row = [i,0]
            # writing the data rows 
            csvwriter.writerow(row)
    for a in nombres_imagenes_entrenamiento_very_mild:
        if a != ".DS_Store":
            row = [a,1]
            csvwriter.writerow(row)
    for b in nombres_imagenes_entrenamiento_moderado:
        if b != ".DS_Store":
            row = [b,2]
            csvwriter.writerow(row)

#%%
with open(filename_validation, 'w') as csvfile: 

    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
        
    csvwriter.writerow(fields) 
    
    for i in nombres_imagenes_validacion_no_demencia:
        if i != ".DS_Store":
            row = [i,0]
            # writing the data rows 
            csvwriter.writerow(row)
    for a in nombres_imagenes_validacion_very_mild:
        if a != ".DS_Store":
            row = [a,1]
            csvwriter.writerow(row)
    for b in nombres_imagenes_validacion_moderado:
         if b != ".DS_Store":
            row = [b,2]
            csvwriter.writerow(row)
# %%
with open(filename_test, 'w') as csvfile: 

    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
        
    csvwriter.writerow(fields) 
    
    for i in nombres_imagenes_testeo_no_demencia:
        if i != ".DS_Store":
            row = [i,0]
            # writing the data rows 
            csvwriter.writerow(row)
    for a in nombres_imagenes_testeo_very_mild:
        if a != ".DS_Store":
            row = [a,1]
            csvwriter.writerow(row)
    for b in nombres_imagenes_testeo_moderado:
        if b != ".DS_Store":
            row = [b,2]
            csvwriter.writerow(row)
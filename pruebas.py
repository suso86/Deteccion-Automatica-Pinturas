#Importamos todas las librerias que necesitamos
#%matplotlib inline
import os
import sys
import random
import math
import re
import time
import json
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import daño
import miVisualize
from sys import argv

ROOT_DIR = os.path.abspath("../")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_WEIGHTS_PATH = ROOT_DIR +"/daño_mask_rcnn_coco.h5"

#-----------------------------------------------------------------------------------------------------------------
# Cargamos nuestro conjunto de datos en variables
config = daño.CustomConfig()
DAÑO_DIR = "/content/Deteccion-Automatica-Pinturas"
#val
dataset = daño.CustomDataset()
dataset.load_custom(DAÑO_DIR, "val")
dataset.prepare()
print("Images value: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

#train
dataset_train = daño.CustomDataset()
dataset_train.load_custom(DAÑO_DIR, "train")
dataset_train.prepare()
print("Images train: {}\nClasses: {}".format(len(dataset_train.image_ids), dataset_train.class_names))

#-----------------------------------------------------------------------------------------------------------------
# cambios para la inferencia.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    # Ejecuta la detección en una imagen a la vez
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

#-----------------------------------------------------------------------------------------------------------------
# Cambiamos el dispositivo objetivo
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

#-----------------------------------------------------------------------------------------------------------------
#Creamos el modelo de inferencia para las pruebas
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)
#-----------------------------------------------------------------------------------------------------------------
with open('/content/Deteccion-Automatica-Pinturas/train/via_region_data.json') as file:
    data = json.load(file)

with open('/content/Deteccion-Automatica-Pinturas/val/via_region_data.json') as file:
    data_val = json.load(file)

def esta_train(cuadro):
  train_dir =  '/content/Deteccion-Automatica-Pinturas/train/'
  esta = False
  for i in dataset_train.image_ids:
    if dataset_train.image_reference(i) == train_dir + cuadro:
      esta = True
  return esta

# Función para indicar el número de daños cogidos en un cuadro determinado
# Entrada: Nombre del cuadro
# Salida: Número de daños recogidos
def Daños_cogidos(nombre_cuadro):
  data_actual = data_val
  if esta_train(nombre_cuadro):
    data_actual = data
  daños=0
  for clave in data_actual:
    if data_actual[clave]['filename'] == nombre_cuadro:
      for clave2 in data_actual[clave]['regions']:
        for clave3 in data_actual[clave]['regions'][clave2]['region_attributes']:
          if clave3 == 'damage': 
            daños+=1
  
  return daños

#-----------------------------------------------------------------------------------------------------------------
#Sprint 1 : Prueba: Mostrar por pantalla algunos cuadros con los daños obtenidos para ver el conjunto de datos que he recogido.
def Sprint1(ruta):
  # Cogemos dos ids de imágenes de la carpeta train y dos de la carpeta val
  image_ids = np.random.choice(dataset.image_ids, 2)
  x = np.random.choice(dataset_train.image_ids,2)
  image_ids = np.append(image_ids,x)
  print(image_ids)

  #Variable para poner dos ejemplos de la carpeta train y dos ejemplos para poner de la  carpeta val
  cambio = 0
  aux = 0
  for image_id in image_ids:
    if cambio == 0:
      image = dataset.load_image(image_id)
      mask, class_ids = dataset.load_mask(image_id)
      # Compute Bounding box
      bbox = utils.extract_bboxes(mask)

      # Display image and additional stats
      print("image_id ", image_id, dataset.image_reference(image_id))
      log("image", image)
      log("mask", mask)
      log("class_ids", class_ids)
      log("bbox", bbox)
      # Display image and instances
      nombre_cuadro = str(image_id) + "_train.jpg"
      miVisualize.saveImage(ruta, nombre_cuadro, image, bbox, mask, class_ids, dataset_train.class_names,
                       figsize=(10,10))
      aux += 1

    else:      
      image = dataset_train.load_image(image_id)
      mask, class_ids = dataset_train.load_mask(image_id)
      # Compute Bounding box
      bbox = utils.extract_bboxes(mask)

      # Display image and additional stats
      print("image_id ", image_id, dataset_train.image_reference(image_id))
      log("image", image)
      log("mask", mask)
      log("class_ids", class_ids)
      log("bbox", bbox)
      # Display image and instances
      nombre_cuadro = str(image_id) + "_value.jpg"
      miVisualize.saveImage(ruta, nombre_cuadro, image, bbox, mask, class_ids, dataset_train.class_names,
                       figsize=(10,10))
      aux += 1

    if aux == 2:
      if cambio == 1:
        cambio = 0
      else:
        cambio = 1 

#-----------------------------------------------------------------------------------------------------------------
#Sprint 2 : Prueba : Validación del algoritmo con el primer entrenamiento realizado.
def Sprint2(ruta,cuadro):
  # Cargamos el peso
  weights_path = "/content/mask_rcnn_daño_0010.h5"
  print("Loading weights ", weights_path)
  model.load_weights(weights_path, by_name=True)

  # Comprobamos la eficiencia del entrenamiento
  if esta_train(nombre_cuadro):
          train_dir = '/content/Deteccion-Automatica-Pinturas/train/'
  else:
          train_dir = '/content/Deteccion-Automatica-Pinturas/val/'
  #Primero mostramos el cuadro con los daños que hemos recogido
  image_id = 0
  for i in dataset_train.image_ids:
    if dataset_train.image_reference(i) == train_dir + cuadro:
      image_id = i
  
  image, image_meta, gt_class_id, gt_bbox, gt_mask =\
      modellib.load_image_gt(dataset_train, config, image_id, use_mini_mask=False)
  info = dataset_train.image_info[image_id]
  print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                        dataset_train.image_reference(image_id)))

  # Run object detection
  results = model.detect([image], verbose=1)

  # Display results
  r = results[0]
  miVisualize.saveImage(ruta, "primer_entreno.jpg", image, r['rois'], r['masks'], r['class_ids'], 
                        dataset_train.class_names,
                        title="Predicción",figsize=(10,10))

  log("gt_class_id", gt_class_id)
  log("gt_bbox", gt_bbox)
  log("gt_mask", gt_mask)

#----------------------------------------------------------------------------------------------------------
# Sprint 3: Prueba : Comparamos el número de daños obtenidos con el entrenamiento y el número de daños que hemos recogido de un cuadro dado,
# si se obtiene el 90% del número de daños que hemos recogido el algoritmo está bien.
def Sprint3(ruta,cuadro):
  # Cargamos el peso
  weights_path = "/content/mask_rcnn_daño_0005 (7).h5"
  print("Loading weights ", weights_path)
  model.load_weights(weights_path, by_name=True)

  if esta_train(nombre_cuadro):
        train_dir = '/content/Deteccion-Automatica-Pinturas/train/'
  else:
        train_dir = '/content/Deteccion-Automatica-Pinturas/val/'
  #Primero mostramos el cuadro con los daños que hemos recogido
  image_id = 0
  for i in dataset_train.image_ids:
    if dataset_train.image_reference(i) == train_dir + cuadro:
      image_id = i

  image = dataset_train.load_image(image_id)
  mask, class_ids = dataset_train.load_mask(image_id)
  # Compute Bounding box
  bbox = utils.extract_bboxes(mask)

  # Display image and additional stats
  print("image_id ", image_id, dataset_train.image_reference(image_id))
  log("image", image)
  log("mask", mask)
  log("class_ids", class_ids)
  log("bbox", bbox)
  # Display image and instances
  #visualize.display_instances(image, bbox, mask, class_ids, dataset_train.class_names,figsize=(10,10))
  miVisualize.saveImage(ruta, "etiquetado.jpg", image, bbox, mask, class_ids, dataset_train.class_names,
                       figsize=(10,10))

  #Luego mostramos el cuadro con los daños que se detecta con el entrenamiento
  image_paths = []
  for filename in os.listdir(train_dir):
      if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
        if filename == cuadro :
          image_paths.append(os.path.join(train_dir, filename))

  for image_path in image_paths:
      img = skimage.io.imread(image_path)
      img_arr = np.array(img)
      results = model.detect([img_arr], verbose=1)
      r = results[0]
      #visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
                              #dataset_train.class_names,
                              #title="Predicción",figsize=(10,10))
      miVisualize.saveImage(ruta, "entrenado.jpg", img, r['rois'], r['masks'], r['class_ids'], 
                              dataset_train.class_names,
                              title="Predicción",figsize=(10,10))
      

  #Estudio de los daños que hemos obtenido
  daños_cogidos = Daños_cogidos(cuadro)
  daños_obtenidos = 0
  for i in r['scores']:
    daños_obtenidos+=1

  #Resultados obtenidos: Interpretar que el algoritmo debe detectar un 90% minimo de daños,
  # si no obtiene ese mínimo hay que seguir entrenando.

  daños_minimos = round((90*daños_cogidos)/100)

  print("Daños recogidos: " + str(daños_cogidos))
  print("Daños obtenidos: " + str(daños_obtenidos))
  print("Daños minimos que tiene que detectar: " + str(daños_minimos))

  if daños_obtenidos < daños_minimos:
    print("Hay que seguir entrenando el algoritmo!!!")

  else:
    print("Buen entrenamiento!!!")

#------------------------------------------------------------------------------------------------------
# Sprint 4
def Sprint4(nombre_cuadro):
    # Cargamos el peso
    weights_path = "/content/mask_rcnn_daño_0005 (7).h5"
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    # Vemos en que carpeta está
    if esta_train(nombre_cuadro):
          real_test_dir = '/content/Deteccion-Automatica-Pinturas/train/'
    else:
          real_test_dir = '/content/Deteccion-Automatica-Pinturas/val/'
    
    image_path = real_test_dir + nombre_cuadro

    # Realizamos la predicción
    imagen = skimage.io.imread(image_path)
    img_arr = np.array(imagen)
    results = model.detect([img_arr], verbose=1)
    r = results[0]

    #Obtenemos los poligonos
    poligonos = miVisualize.PoligonosDaños(img, r['rois'], r['masks'], r['class_ids'],dataset_train.class_names)

    # Obtenemos las áreas que encontramos y la sumamos
    areas_encontradas = []
    for pl in poligonos:
      poligono_encontrado = pl[1]
      coordenadas_encontradas = poligono_encontrado.get_xy()
      area_encontrada = abs(AreaPol(coordenadas_encontradas))
      areas_encontradas.append(area_encontrada)

    suma_areas = 0
    for i in areas_encontradas:
      suma_areas += i

    tamaño = imagen.shape[0]*imagen.shape[1]
    zona_no_dañada = tamaño - suma_areas
    zona_dañada = tamaño - zona_no_dañada
    porcentaje = (zona_dañada * 100)/tamaño


    print("Tamaño del cuadro:", tamaño)
    print("Suma de las áreas que hemos encontrados:",suma_areas)
    print("Zona no dañada:",zona_no_dañada)
    print("Zona dañada:",zona_dañada)
    print("Porcentaje de daño que presenta el cuadro:",porcentaje,"%")
      

############################################################
#  Sprines
############################################################
if __name__ == '__main__':
    import argparse

    # Argumentos de la linea de comandos
    parser = argparse.ArgumentParser(
        description='Elección de pruebas que se desea realizar')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'sprint1' or 'sprint2' or 'sprint3'")
    parser.add_argument('--cuadro', required=False,
                        #metavar="/path/to/custom/dataset/",
                        help='Nombre del cuadro que se quiere')

    parser.add_argument('--path', required=False,
                        #metavar="/path/to/custom/dataset/",
                        help='Nombre de la ruta donde se desea guardar')

    args = parser.parse_args()

    # Validando argumentos
    if args.command == "sprint2":
        assert args.cuadro,"Argumento --cuadro es requerido para ejecutar este sprint"
    elif args.command == "sprint3":
        assert args.cuadro,"Argumento --cuadro es requerido para ejecutar este sprint"
    elif args.command == "sprint4":
        assert args.cuadro,"Argumento --cuadro es requerido para ejecutar este sprint"

    print("Sprint: ", args.command)
    print("Dataset: ", args.cuadro)

    ruta = '/content/'
    if args.path:
      ruta = args.path


    #Realización de los sprines
    if args.command == "sprint1":
      Sprint1(ruta)
    elif args.command == "sprint2":
      Sprint2(ruta,args.cuadro)
    elif args.command == "sprint3":
      Sprint3(ruta,args.cuadro)
    elif args.command == "sprint4":
      Sprint4(args.cuadro)







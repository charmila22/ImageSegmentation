import numpy as np 
import streamlit as st 
import tensorflow_io as tfio
from tensorflow.keras.models import load_model 
import tensorflow as tf
import math
import time 
import os 

def convert_bytes(file_path, unit=None):
    size = os.path.getsize(file_path)
    if unit == "KB":
        return str(round(size / 1024, 3)) + ' Kilobytes'
    elif unit == "MB":
        return str(round(size / (1024 * 1024), 3)) + ' Megabytes'
    else:
        return str(size) + ' bytes'

def sigmoid(x):
    return 1 / (1 + math.exp(-x)) 

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)   

# this is the main function in which we define our webpage  
def main(): 
    st.title("Pneumothorax mask prediction") 
    image_path = st.selectbox('select an image',
              ('Images/1.2.276.0.7230010.3.1.4.8323329.10008.1517875220.957633.dcm', 
               'Images/1.2.276.0.7230010.3.1.4.8323329.10005.1517875220.958951.dcm', 
               'Images/1.2.276.0.7230010.3.1.4.8323329.10007.1517875220.954767.dcm',
               'Images/1.2.276.0.7230010.3.1.4.8323329.10035.1517875221.57974.dcm',
               'Images/1.2.276.0.7230010.3.1.4.8323329.10034.1517875221.47394.dcm',
               'Images/1.2.276.0.7230010.3.1.4.8323329.10026.1517875221.22915.dcm',
               'Images/1.2.276.0.7230010.3.1.4.8323329.10025.1517875221.19235.dcm',
               'Images/1.2.276.0.7230010.3.1.4.8323329.10016.1517875220.992175.dcm',
               'Images/1.2.276.0.7230010.3.1.4.8323329.10020.1517875221.5514.dcm',
               'Images/1.2.276.0.7230010.3.1.4.8323329.10023.1517875221.9111.dcm'))
    
    #st.write("option is ",option)
    #image_file = st.file_uploader("Upload Image",type=['dcm'])
    
    if image_path is not None:
        
        #image_path = image_file.name
        image_bytes = tf.io.read_file(image_path)
        image = tfio.image.decode_dicom_image(image_bytes, dtype=tf.float64)
        image = tf.image.resize(image, [256, 256])
        #preprocessing
        image = tf.cast(image, tf.float32) / 255.0  
        st.markdown('**_Input_ Image**')
        st.image(image.numpy())
        
        if st.button("Predict"):
          st.header("Predicting using unquantized model")
          class_model_size = convert_bytes('best_model.h5', "MB")
          st.write("size of unquantized classification model: ",class_model_size)
          seg_model_size = convert_bytes('BOOTSTRAP_model_20.h5', "MB")
          st.write("size of unquantized segmentation model: ",seg_model_size)
          
          dependencies = {'dice_coef': dice_coef}
          segmentation_model = load_model('BOOTSTRAP_model_20.h5',custom_objects=dependencies) 
          classification_model = load_model('best_model.h5') 

          start = time.time() 
          predicted_label =  classification_model.predict(image)
          
          if predicted_label < 0.3:
            st.markdown("**_Nothing_ to_ segment_ because_ this_ is_ not_ a_ pneumothorax_ image**")
            end = time.time()
            st.write("Time taken for prediction: ",round(end-start,2) ,"seconds.") 
          else:
            predicted_mask =  segmentation_model.predict(image) 
            predicted_mask = predicted_mask[0,:,:,0]
            sigmoid_v = np.vectorize(sigmoid)
            predicted_mask = sigmoid_v(predicted_mask) 
            predicted_mask[predicted_mask >= 0.5] = 1
            predicted_mask[predicted_mask < 0.5] = 0
            st.markdown('**_Predicted_ Mask**')
            st.image(predicted_mask)
            end = time.time()
            st.write("Time taken for prediction: ",round(end-start,2) ,"seconds.") 

          
          st.header("Predicting using quantized model")    
          
          tflite_class_model_name = "tflite_class_model.tflite"
          tflite_seg_model_name = "tflite_seg_model.tflite"

          class_model_size = convert_bytes(tflite_class_model_name, "MB")     
          st.write("size of quantized classification model: ",class_model_size)
          seg_model_size = convert_bytes(tflite_seg_model_name, "MB")
          st.write("size of quantized segmentation model: ",seg_model_size)
          
          seg_interpreter = tf.lite.Interpreter(model_path = tflite_seg_model_name)
          seg_interpreter.allocate_tensors()
          seg_input_details = seg_interpreter.get_input_details()
          seg_output_details = seg_interpreter.get_output_details()
          
          class_interpreter = tf.lite.Interpreter(model_path = tflite_class_model_name)
          class_interpreter.allocate_tensors()
          class_input_details = class_interpreter.get_input_details()
          class_output_details = class_interpreter.get_output_details()

          start = time.time()
          class_interpreter.set_tensor(class_input_details[0]['index'], image.numpy())
          class_interpreter.invoke()
          tflite_model_predictions = class_interpreter.get_tensor(class_output_details[0]['index'])
          if tflite_model_predictions < 0.3:
            st.markdown("**_Nothing_ to_ segment_ because_ this_ is_ not_ a_ pneumothorax_ image**")
            end = time.time()
            st.write("Time taken for prediction: ",round(end-start,2) ,"seconds.") 
          else:
            seg_interpreter.set_tensor(seg_input_details[0]['index'], image.numpy())
            seg_interpreter.invoke()
            tflite_model_predictions = seg_interpreter.get_tensor(seg_output_details[0]['index'])
            predicted_mask = tflite_model_predictions[0,:,:,0]
            sigmoid_v = np.vectorize(sigmoid)
            predicted_mask = sigmoid_v(predicted_mask) 
            predicted_mask[predicted_mask >= 0.5] = 1
            predicted_mask[predicted_mask < 0.5] = 0
            st.markdown('**_Predicted_ Mask**')
            st.image(predicted_mask)
            end = time.time()
            st.write("Time taken for prediction: ",round(end-start,2) ,"seconds.") 
     
if __name__=='__main__': 
    main()

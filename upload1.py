from os import name
from flask import *  
from  keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np
app = Flask("kapil")  
 
@app.route('/')  
def upload():  
    return render_template("file_upload_form.html")  
 
@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename) 
        from  keras.models import load_model
        img = cv2.imread(f.filename)
        img = np.array(img)
        res = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
        model = load_model('imagees_prediction.h5')
        test_image_arr_4d   = np.expand_dims(res, axis=0)
        print(model.predict(test_image_arr_4d)[0][0])
	
        if int(model.predict(test_image_arr_4d)[0][0]) == 0:
            return render_template("success.html", name = f.filename ,tee=type(img),ml=int(model.predict(test_image_arr_4d)[0][0]) ,data ="cat")
        else:
            return render_template("success.html", name = f.filename ,tee=type(img),ml=int(model.predict(test_image_arr_4d)[0][0]) ,data ="dog")
          
  
if "kapil" == "_main_":
    app.run(debug = True)    

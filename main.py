from google_images_search import GoogleImagesSearch #Library to scrape web for images to feed AI
import gi #GTK
import sys #sys library
import os #mkdir
import matplotlib.pyplot as plt #tf
import numpy as np #tf
import PIL #tf
import tensorflow as tf #tensorflow
import pathlib
from tensorflow import keras #image
from tensorflow.keras import layers #image (ignore errors [caused by "lazy" loading fashion])
from tensorflow.keras.models import Sequential #image (ignore errors [caused by "lazy" loading fashion])
gis = GoogleImagesSearch('AIzaSyAbGTmQB1fNEQER9pvojwxXyg18cYVZiWg', 'a760bac2838e54678', validate_images=False) #API Key, CX Key, validate_images disabled for faster scraping
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos.tar', origin=dataset_url, extract=True)
data_dir = pathlib.Path(data_dir).with_suffix('')
#gi.require_version('Gtk', '4.0')
#gi.require_version('Adw','1')
#from gi.repository import Gtk, Adw
#class MainWindow(Gtk.ApplicationWindow):
    #def __init__(self, *args, **kwargs):
     #   super().__init__(*args, **kwargs)
        #Window
     #   self.set_default_size(600, 600)
      #  self.set_title("AI Image Detector")
       #Boxes
        #self.box1=Gtk.Box()
       # self.box1 = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        # Things will go here
        #self.set_child(self.box1)
        #self.button = Gtk.Button(label="Hello")
        #self.entry=Gtk.Entry()
        #self.entry.set_placeholder_text("What object will it identify?")
        #spin_adjustment=0
        #self.spin=Gtk.SpinButton(adjustment=0,climb_rate=1,)
        #self.entry.connect("activate", self.uwu)
        #self.spin.connect('changed', self.owo)
        #self.button.connect('clicked',self.hello)
        #Adding things to boxes
        #self.box1.append(self.button)
        #self.box1.append(self.entry)
        #self.box1.append(self.spin)
        
    #def uwu(self,entry):
        #pass
   # def owo(self,spin):
       # pass
   # def hello(self,button):
      #  print("Hell")
      #  global search_query
       # search_query=self.entry.get_text()
#class MyApp(Adw.Application):
    #def __init__(self, **kwargs):
     #   super().__init__(**kwargs)
     #   self.connect('activate', self.on_activate)

    #def on_activate(self, app):
      #  self.win = MainWindow(application=app)
       # self.win.present()

#app = MyApp(application_id="com.example.GtkApplication")
#app.run(sys.argv)
search_query=''
data_amount=0
def process():
    global img_height
    global img_width
    global model
    global class_names
    #Loader params
    batch_size = 64
    img_height = 250
    img_width = 250
    #Developing model
    train_ds = tf.keras.utils.image_dataset_from_directory(
        'images/',
        validation_split=0.2, #percentage of imgs used for validation
        subset="training",
        seed=9857295,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        'images/',
        validation_split=0.2, #percentage of imgs used for validation
        subset="validation",
        seed=9857295,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    #Debug
    class_names = train_ds.class_names
    print(class_names)
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    #Optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    normalization_layer = layers.Rescaling(1./255) #Standardize colors
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))
    num_classes = len(class_names)
    #Augment data so our AI isnt dissappointment
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                            input_shape=(img_height,
                                        img_width,
                                        3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
)
    #Making model and including layers(i have no idea what this is)
    model = Sequential([ #ADD YOUR LAYERS HERE
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, name="outputs")
])
    #Compiling model (turning into runnable machine code ig)
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    #Debug
    model.summary()
    #Training the model
    epochs=15 #How many times AI looks at the images
    history = model.fit( #Include your layers here
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    #Plotting accurancy to see if our AI is a dissappointment or not
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    identify()

def identify():
    global img_height
    global img_width
    global class_names
    global model
    asdf=input("Url to image for identification: ")
    sunflower_url = asdf
    sunflower_path = tf.keras.utils.get_file(origin=sunflower_url)
    img = tf.keras.utils.load_img(
        sunflower_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    asdf= input("Again?: ")
    if asdf=="y":
        identify()
    else:
        exit()
    

def gather():
    global search_query
    global data_amount
    global _search_params
    asdf = input("Skip? (Saves API Usage): ")
    if asdf=="y":
        process()
    else:
        pass
    search_query=input("Search query: ")
    data_amount=int(input("How many images: "))
    _search_params = {
        'q': search_query,
        'num': data_amount,
        'fileType': 'jpg|jpeg|png',
        'rights': 'cc_publicdomain|cc_attribute|cc_sharealike|cc_noncommercial|cc_nonderived',
        'safe': 'safeUndefined', #cannot take multiple vals
        'imgType': 'photo', #cannot take multiple vals
        'imgSize': 'imgSizeUndefined', #cannot take multiple vals
        'imgDominantColor': 'imgDominantColorUndefined', #cannot take multiple vals
        'imgColorType': 'color' #cannot take multiple vals
}
    if os.path.exists('images/'+search_query+'/')==False:
        os.mkdir(os.path.join('images/', search_query) )
        gis.search(search_params=_search_params, path_to_dir='images/'+search_query)
    else:
        gis.search(search_params=_search_params, path_to_dir='images/'+search_query)
    asdf=input("More data? y/n: ")
    if asdf=="y":
        gather()
    if asdf=="n":
        process()
    else:
        print("Err unknown cmd")
        exit()


gather()

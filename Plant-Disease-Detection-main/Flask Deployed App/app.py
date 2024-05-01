import os
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd


disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model = CNN.CNN(39)    
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index


app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/nature_supplements/<string>')
def nature_supplements(string):
    print(string)
    d = [{'disease_name': 'Apple : Scab', 'details': {'nature_name': 'Sujala Foliar Fertilizer',
                                                      'img_link': 'https://m.media-amazon.com/images/I/41eww7SGxWL._SX300_SY300_QL70_FMwebp_.jpg',
                                                      'buying_link': 'https://www.amazon.in/Sujala-Foliar-Fertilizer-Plants-1kg/dp/B07MYYF2X5',
                                                      'video_link': 'https://www.youtube.com/watch?v=dhgaToMj4dE'}},
         {'disease_name': 'Apple : Black Rot', 'details': {'nature_name': 'EBS Bacillus Subtilis Bio Fungicide Powder',
                                                           'img_link': 'https://essentialbiosciences.com/assets/images/products/bacillus_powder.jpg',
                                                           'buying_link': 'https://agribegri.com/products/buy-ebs-bacillus-subtilis-bio-fungicide-online-1.php?utm_source=google&utm_medium=pmax&utm_campaign=PMax:+Smart+All+3000&utm_term=&utm_campaign=PMax:+Smart+All+3000&utm_source=adwords&utm_medium=ppc&hsa_acc=7428009962&hsa_cam=17668524651&hsa_grp=&hsa_ad=&hsa_src=x&hsa_tgt=&hsa_kw=&hsa_mt=&hsa_net=adwords&hsa_ver=3&gad_source=1&gclid=CjwKCAjw5v2wBhBrEiwAXDDoJSdwV9l0yAxPAFoBKwe-CbuBkno_wqcth8ILCd3Or9aW0RF6Je7ymxoCfCgQAvD_BwE',
                                                           'video_link': 'https://www.youtube.com/watch?v=fdZ8sdh0sYg'}},
         {'disease_name': 'Apple : Cedar Rust', 'details': {'nature_name': 'Horsetail Extract Powder',
                                                            'img_link': 'https://www.bulksupplements.com/cdn/shop/files/Horsetail-Extract-Powder-AMZ-500g.jpg?v=1694024116&width=800',
                                                            'buying_link': 'https://www.bulksupplements.com/en-IN/products/horsetail-extract-powder?variant=32133412946031&utm_source=google&utm_medium=organic&utm_campaign=India%20Multifeeds&utm_content=Horsetail%20Extract%20Powder&currency=INR&utm_campaign=20955427136&utm_source=g&utm_medium=cpc&utm_content=&utm_term=&ad_id=688165389117&wickedsource=google&wickedid=CjwKCAjw5v2wBhBrEiwAXDDoJaxUtt1KoD9qWj3kyg8640bgpNYqfo3RqfqSTSs5jL20ZxoG62p9IxoC8e8QAvD_BwE&wickedid=688165389117&wcid=20955427136&wv=4&gad_source=1&gclid=CjwKCAjw5v2wBhBrEiwAXDDoJaxUtt1KoD9qWj3kyg8640bgpNYqfo3RqfqSTSs5jL20ZxoG62p9IxoC8e8QAvD_BwE',
                                                            'video_link': 'https://www.youtube.com/watch?v=qTXil2XAWL0'}},
         {'disease_name': 'Cherry : Powdery_mildew', 'details': {'nature_name': 'Equisetum Arvense Extract Powder',
                                                                 'img_link': 'https://5.imimg.com/data5/SELLER/Default/2024/3/399391745/ER/KG/ZB/104938620/equisetum-arvense-extract-powder-250x250.jpg',
                                                                 'buying_link': 'https://www.indiamart.com/proddetail/equisetum-arvense-extract-powder-2849511336862.html',
                                                                 'video_link': 'https://www.youtube.com/watch?v=PmJQwZwB0xQ'}},
         {'disease_name': 'Cherry : Healthy',
          'details': {'nature_name': 'Biomass Lab Sampoorn Fasal Ahaar (Multipurpose Organic Fertilizer & Plant Food)',
                      'img_link': 'https://m.media-amazon.com/images/I/51VzsRSU+wL._SY300_SX300_.jpg',
                      'buying_link': 'https://www.amazon.in/Biomass-Lab-Sampoorn-Multipurpose-Fertilizer/dp/B07W8N4BVH',
                      'video_link': 'https://www.youtube.com/watch?v=WZ4LsQYDFnQ'}}, {'disease_name': 'Apple : Healthy',
                                                                                      'details': {
                                                                                          'nature_name': 'Tapti Booster Organic Fertilizer',
                                                                                          'img_link': 'https://rukminim1.flixcart.com/image/416/416/kc6jyq80/soil-manure/6/y/v/500-tapti-booster-500-ml-green-yantra-original-imaftd6rrgfhvshc.jpeg?q=70',
                                                                                          'buying_link': 'https://agribegri.com/products/tapti-booster-500-ml--organic-fertilizer-online-in-india.php',
                                                                                          'video_link': 'https://www.youtube.com/watch?v=dWTUeEwxL3w'}},
         {'disease_name': 'Blueberry : healthy', 'details': {'nature_name': 'GreenStix Fertilizer',
                                                             'img_link': 'https://lazygardener.in/cdn/shop/products/greenstix-all-purpose-plant-food-sticks-fertilizer-sticks-plant-food-sticks-lazygardener-821257.jpg?v=1695147465',
                                                             'buying_link': 'https://lazygardener.in/products/greenstix?variant=30920527020087&currency=INR&utm_medium=product_sync&utm_source=google&utm_content=sag_organic&utm_campaign=sag_organic&utm_campaign=gs-2019-10-29&utm_source=google&utm_medium=smart_campaign',
                                                             'video_link': 'https://www.youtube.com/watch?v=PjGdk7GVH0I'}},
         {'disease_name': 'Corn : healthy',
          'details': {'nature_name': 'Biomass Lab Sampoorn Fasal Ahaar (Multipurpose Organic Fertilizer & Plant Food)',
                      'img_link': 'https://m.media-amazon.com/images/I/51VzsRSU+wL._SY300_SX300_.jpg',
                      'buying_link': 'https://www.amazon.in/Biomass-Lab-Sampoorn-Multipurpose-Fertilizer/dp/B07W8N4BVH',
                      'video_link': 'https://www.youtube.com/watch?v=t-ax7AjNb1A'}},
         {'disease_name': 'Raspberry : healthy',
          'details': {'nature_name': "Karen's Naturals, Organic Just Raspberries",
                      'img_link': 'https://s3.images-iherb.com/jte/jte29001/v/6.jpg',
                      'buying_link': 'https://in.iherb.com/pr/Karen-s-Naturals-Organic-Just-Raspberries-1-5-oz-42-g/37730?gclsrc=ds',
                      'video_link': 'https://www.youtube.com/watch?v=wraYzHycDLA'}},
         {'disease_name': 'Soybean : healthy', 'details': {'nature_name': 'Max Crop Liquid Fertilizer',
                                                           'img_link': 'https://encrypted-tbn0.gstatic.com/shopping?q=tbn:ANd9GcRyFZFHGj1Wi0fabSOYVadCnsg8UdvhpKOHI6gxmItDLmz7RdC67HFFNyU7PYyD6SnlgXOs31yW_8-YWJKOeal7K4UQ3lAIxef-4x350ac&usqp=CAE',
                                                           'buying_link': 'https://agribegri.com/products/max-crop-1-litre.php',
                                                           'video_link': 'https://www.youtube.com/watch?v=6GjMUSlRJjo'}},
         {'disease_name': 'Tomato : healthy',
          'details': {'nature_name': 'Tomato Fertilizer Organic, for Home, Balcony, Terrace & Outdoor Gardening',
                      'img_link': 'https://casadeamor.in/cdn/shop/files/tomatofertilizer-900gm.jpg?v=1707295552&width=600',
                      'buying_link': 'https://www.casagardenshop.com/products/tomato-fertilizer-for-home-terrace-outdoor-gardening?variant=32106353131619&currency=INR&utm_medium=product_sync&utm_source=google&utm_content=sag_organic&utm_campaign=sag_organic&utm_campaign=gs-2020-02-08&utm_source=google&utm_medium=smart_campaign',
                      'video_link': 'https://www.youtube.com/watch?v=nnJ4W7aA2EQ'}},
         {'disease_name': 'Strawberry : healthy',
          'details': {'nature_name': 'SWISS GREEN ORGANIC PLANT GROWTH PROMOTER STRAWBERRY Fertilizer',
                      'img_link': 'https://rukminim1.flixcart.com/image/416/416/k2urhjk0/soil-manure/q/y/d/500-organic-plant-growth-promoter-strawberry-swiss-green-original-imafm3vx8mzyhxzg.jpeg?q=70',
                      'buying_link': 'https://www.flipkart.com/swiss-green-organic-plant-growth-promoter-strawberry-fertilizer-manure/p/itmcaf7955b802fa?pid=SMNFM3TT2RNZEQYD&lid=LSTSMNFM3TT2RNZEQYDIGMSRD&marketplace=FLIPKART&cmpid=content_soil-manure_8965229628_gmc',
                      'video_link': 'https://www.youtube.com/watch?v=fs4jB7IRTyE'}}, {'disease_name': 'Peach : healthy',
                                                                                      'details': {
                                                                                          'nature_name': 'Jeevamrut (Plant Growth Tonic)',
                                                                                          'img_link': 'https://nurserylive.com/cdn/shop/products/nurserylive-g-fertilizer-jeevamrut-plant-growth-tonic-500-ml-for-garden_jpg-660593_600x600.jpg?v=1680591234',
                                                                                          'buying_link': 'https://nurserylive.com/products/jeevamrut-plant-growth-tonic-250-ml-for-garden?variant=34163328811148',
                                                                                          'video_link': 'https://www.youtube.com/watch?v=DjMBuZTWqAM'}},
         {'disease_name': 'Potato : healthy', 'details': {'nature_name': 'Saosis Fertilizer for potato Fertilizer',
                                                          'img_link': 'https://rukminim1.flixcart.com/image/416/416/kfvfwy80/soil-manure/6/p/g/30-fertilizer-for-potato-saosis-original-imafw8b3sgcyrhh4.jpeg?q=70',
                                                          'buying_link': 'https://www.flipkart.com/saosis-fertilizer-potato/p/itmc40dfbfcd49f7?pid=SMNFW8B3DTY2V6PG&lid=LSTSMNFW8B3DTY2V6PGHC2BXS&marketplace=FLIPKART&cmpid=content_soil-manure_730597647_g_8965229628_gmc_pla&tgi=sem,1,G,11214002,g,search,,476044024748,,,,c,,,,,,,&ef_id=Cj0KCQjwsLWDBhCmARIsAPSL3_3TcmjyWFqJEZtCD0SQiDb9UFyESkiYtgU35W5Qs_KnTCkDS_Px2Y4aArmBEALw_wcB:G:s&s_kwcid=AL!739!3!476044024748!!!g!293946777986!&gclsrc=aw.ds',
                                                          'video_link': 'https://www.youtube.com/watch?v=F7febfoX3GI'}},
         {'disease_name': 'Jatropha : healthy', 'details': {'nature_name': 'Sansar Green Grapes Fertilizer',
                                                            'img_link': 'https://rukminim1.flixcart.com/image/416/416/k51cpe80/soil-manure/4/m/h/400-grapes-fertilizer-sansar-green-original-imafntdujntbdner.jpeg?q=70',
                                                            'buying_link': 'https://www.flipkart.com/sansar-green-grapes-fertilizer/p/itm90f06136b50c7?pid=SMNFNT8F7FCQJ4MH&lid=LSTSMNFNT8F7FCQJ4MHMXEELG&marketplace=FLIPKART&cmpid=content_soil-manure_730597647_g_8965229628_gmc_pla&tgi=sem,1,G,11214002,g,search,,476044024748,,,,c,,,,,,,&ef_id=Cj0KCQjwsLWDBhCmARIsAPSL3_0d3lBg14nXBbhJeTP7z5mqzpyHXmv4c7cL9mW4MqCBlnskpoMB8C0aArWREALw_wcB:G:s&s_kwcid=AL!739!3!476044024748!!!g!293946777986!&gclsrc=aw.ds',
                                                            'video_link': 'https://www.youtube.com/watch?v=mK0ZtCvxiag'}},
         {'disease_name': 'Corn : Cercospora_leaf_spot Gray_leaf_spot',
          'details': {'nature_name': 'Equisetum Arvense Extract Powder',
                      'img_link': 'https://cdn.moglix.com/p/eg9zLUnQd4a6X-xxlarge.jpg',
                      'buying_link': 'https://www.bighaat.com/products/katyayani-trichoderma-harzianum-bio-fungicide-powder?variant=40612781129751&currency=INR&utm_medium=product_sync&utm_source=google&utm_content=sag_organic&utm_campaign=sag_organic&utm_source=Google&utm_medium=CPC&utm_campaign=18815455412&utm_adgroup=&utm_term=&creative=&device=c&devicemodel=&matchtype=&feeditemid=&targetid=&network=x&placement=&adposition=&GA_loc_interest_ms=&GA_loc_physical_ms=9148884&gad_source=1&gclid=CjwKCAjw5v2wBhBrEiwAXDDoJXVmYnLuRxdQa0sS3OU9vWaf1d47ODWih78HE0ljUuzdruZXW4QbWhoCvI0QAvD_BwE',
                      'video_link': 'https://www.youtube.com/watch?v=PmJQwZwB0xQ&t=54s'}},
         {'disease_name': 'Corn : Common Rust', 'details': {'nature_name': 'Amibion Flower Booster',
                                                            'img_link': 'https://plantlane.com/cdn/shop/products/IMG-20191226-WA0007-1.jpg?v=1682037615',
                                                            'buying_link': 'https://plantlane.com/cdn/shop/products/IMG-20191226-WA0007-1.jpg?v=1682037615',
                                                            'video_link': 'https://www.youtube.com/watch?v=_dH6sZXGVmc'}},
         {'disease_name': 'Corn : Northern Leaf Blight',
          'details': {'nature_name': 'Shri Herbal Horsetail (Equisetum arvense)',
                      'img_link': 'https://m.media-amazon.com/images/I/11feZM+uD-L.jpg',
                      'buying_link': 'https://www.amazon.in/Shri-Horsetail-Equisetum-arvense-Powder-100/dp/B09BJQM5N8',
                      'video_link': 'https://www.youtube.com/watch?v=PxcXOmXNiB0'}},
         {'disease_name': 'Jatropha : Black_rot', 'details': {'nature_name': 'Real Sona',
                                                              'img_link': 'https://marketplace.agribazaar.com/s/6204bf2ecd8cdec2b3b45115/64c187ab77b05c51234a69a6/realpercent20sonapercent20boxpercent20_percent20bottle-480x480.png',
                                                              'buying_link': 'https://agribegri.com/products/buy-real-sona-bio-stimulant-online--buy-plant-promoter-online.php',
                                                              'video_link': 'https://www.youtube.com/watch?v=sSXfpNKq1OQ'}},
         {'disease_name': 'Jatropha : Esca(Black_Measles)', 'details': {'nature_name': 'Vine Vitality',
                                                                        'img_link': 'https://m.media-amazon.com/images/I/61myl2ok8ZL._SX385_.jpg',
                                                                        'buying_link': 'https://www.amazon.com/Vine-Vitality-Nutrient-Fertilizer-Concentrate/dp/B00VVVSI06',
                                                                        'video_link': 'https://www.youtube.com/watch?v=6izQfXMO9nY'}},
         {'disease_name': 'Jatropha : Leaf Blight', 'details': {'nature_name': 'Foliage Fortify',
                                                                'img_link': 'https://m.media-amazon.com/images/I/51au6MOHKoL._SX679_.jpg',
                                                                'buying_link': 'https://www.amazon.in/Naturally-Green-Purpose-Foliage-Fertilizer/dp/B096N4HY78?th=1',
                                                                'video_link': 'https://www.youtube.com/watch?v=xYhOjKwLznM'}},
         {'disease_name': 'Peach : Bacterial Spot', 'details': {
             'nature_name': 'Bloom - Growth booster, Flowering Stimulant, healthier plants, shoot growth, stem strength,increases food production capacity',
             'img_link': 'https://badikheti-production.s3.ap-south-1.amazonaws.com/products/202301171524051912935582.jpg',
             'buying_link': 'https://agribegri.com/products/buy-flowering-stimulant-online--flower-blooming-chemical.php?utm_source=google&utm_medium=pmax&utm_campaign=PMax:+Smart+All+3000&utm_term=&utm_campaign=PMax:+Smart+All+3000&utm_source=adwords&utm_medium=ppc&hsa_acc=7428009962&hsa_cam=17668524651&hsa_grp=&hsa_ad=&hsa_src=x&hsa_tgt=&hsa_kw=&hsa_mt=&hsa_net=adwords&hsa_ver=3&gad_source=1&gclid=CjwKCAjw5v2wBhBrEiwAXDDoJcBGeLnMP2_F9V43vOYf0kQCRyHv_vlp4RhAo4xqRAdh5G9wp4LDbRoCTxwQAvD_BwE',
             'video_link': 'https://www.youtube.com/watch?v=5ocZCkBcbbo'}},
         {'disease_name': 'Pepper Bell : Bacterial_spot',
          'details': {'nature_name': 'Grenera Turmeric Capsules (Haldi with Black Pepper)',
                      'img_link': 'https://m.media-amazon.com/images/I/61oGyzpqVpS._SX679_.jpg',
                      'buying_link': 'https://www.amazon.in/Grenera-Turmeric-Capsules-Curcumin-Supplement/dp/B097MHPHBL',
                      'video_link': 'https://www.youtube.com/watch?v=AcyLWvfFXBI'}},
         {'disease_name': 'Potato : Early Blight', 'details': {
             'nature_name': 'Casa De Amor Diatomaceous Earth | Eco Magic | Organic Eco Friendly Safe Pest Control',
             'img_link': 'https://casadeamor.in/cdn/shop/files/Diatomaceousearthpowder900gm.jpg?v=1697023632&width=600',
             'buying_link': 'https://casadeamor.in/products/eco-magic-dust-diatomaceous-earth-effective-organic-eco-friendly#:~:text=It%20contains%20trace%20minerals%20including,also%20a%20good%20soil%20conditioner.',
             'video_link': 'https://www.youtube.com/watch?v=mI4D8Vn6Lkg'}}, {'disease_name': 'Potato : Late Blight',
                                                                             'details': {
                                                                                 'nature_name': 'Garlic Liquid Extract',
                                                                                 'img_link': 'https://www.vedaoils.com/cdn/shop/products/GARLIC.jpg?v=1656656962',
                                                                                 'buying_link': 'https://www.vedaoils.com/products/garlic-liquid-extract?variant=37540251500712&currency=INR&utm_medium=product_sync&utm_source=google&utm_content=sag_organic&utm_campaign=sag_organic&gad_source=1&gclid=CjwKCAjw5v2wBhBrEiwAXDDoJVijUXNx9qwPw6mUfQfrwihEEsUV5f2O0gehmzA4-ff5Xb8-S3vCIRoCl0sQAvD_BwE',
                                                                                 'video_link': 'https://www.youtube.com/watch?v=rZ-i0dZuYlQ'}},
         {'disease_name': 'Squash : Powdery mildew', 'details': {
             'nature_name': 'GUARD Certified Organic Bio Fungus managment for Seeds and Young Plants Trichoderma viride',
             'img_link': 'https://m.media-amazon.com/images/I/51jXcenN2rL.jpg',
             'buying_link': 'https://www.amazon.in/Guard-Certified-Organic-Management-Trichoderma/dp/B0744KJHVZ',
             'video_link': 'https://www.youtube.com/watch?v=fdZ8sdh0sYg&t=38s'}},
         {'disease_name': 'Strawberry : Leaf Scorch',
          'details': {'nature_name': 'Bloom Nutrition Greens and Superfoods Powder',
                      'img_link': 'https://m.media-amazon.com/images/I/51s71JZVpAL._AC_SX679_.jpg',
                      'buying_link': 'https://www.amazon.com/Bloom-Nutrition-SuperGreens-Powder-Smoothie-Juice-Mix/dp/B098SFLC6Z/ref=sr_1_1_sspa?dib=eyJ2IjoiMSJ9.nNEeL2Tbm-x4zkWSx_Hqn0fGqQJAmQvw2-4pk3uECUx-ZMUfkO7a2ci71VdDj_1LKcCp7q7jfkZ8IAgxS26sVtKpN9Z-tkxlMYh4NSjkNh6wLWcxdtOfzuzMbz4NU9DI-gap2SGTZhrw6cMD4wzKGcOcu4CkbNs8YXXUglia6OWLKS-4t1-DhMYB-eqSlmBYbLZdoKfpoUpO42h5nnLa8IvKLUSCVJT4pZNzuWu4ZZEXULwwzG38fVLm35-8aq3RwppU7nRPFoGlHFNWUWy5v0ooPmjMbA6BKta4jxtJTVc.UFtUP_ug1KtAiihxBdVSLzDsXo5NBgP5YF_iBnvcZ8Q&dib_tag=se&keywords=berry+bloom&qid=1713415859&sr=8-1-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&psc=1',
                      'video_link': 'https://www.youtube.com/watch?v=Or23Pow-Og0'}},
         {'disease_name': 'Tomato : Bacterial_spot',
          'details': {'nature_name': 'THE WET TREE Pseudomonas Fluorescens Powder',
                      'img_link': 'https://m.media-amazon.com/images/I/51oM2Ho7erL._SY445_SX342_QL70_FMwebp_.jpg',
                      'buying_link': 'https://www.amazon.in/WET-TREE-Pseudomonas-Fluorescens-Fungicide/dp/B0CGPL1B9S',
                      'video_link': 'https://www.youtube.com/watch?v=qPNkzWSrLvQ'}},
         {'disease_name': 'Tomato : Early Blight', 'details': {'nature_name': 'Phyto Carb',
                                                               'img_link': 'https://i0.wp.com/geturpet.com/wp-content/uploads/2021/02/Phyto-Carb-Co2-Supplement-Algae-minus-100ml.jpg?resize=600%2C600&ssl=1',
                                                               'buying_link': 'https://greenmossaquarium.com/product/25235591/Phyto-Carb--Plant-Fertilizer---100-mL',
                                                               'video_link': 'https://www.youtube.com/watch?v=CG8Ux99FUqI'}},
         {'disease_name': 'Tomato : Late Blight', 'details': {'nature_name': 'Going Greens Tomato Plant Fertilizer',
                                                              'img_link': 'https://m.media-amazon.com/images/I/51leOzLR7tL._SX300_SY300_QL70_FMwebp_.jpg',
                                                              'buying_link': 'https://www.amazon.in/Going-Greens-Tomato-Fertilizer-200/dp/B0BGSRFJBN?th=1',
                                                              'video_link': 'https://www.youtube.com/watch?v=aC1B2ePY3Ag'}},
         {'disease_name': 'Tomato : Leaf Mold', 'details': {'nature_name': 'Nature Nourish Vermicompost Fertilizer',
                                                            'img_link': 'https://rukminim1.flixcart.com/image/400/400/kplisnk0/soil-manure/w/c/b/vermicompost-nature-nourish-original-imag3sg47ug8pyt2.jpeg?q=70',
                                                            'buying_link': 'https://www.flipkart.com/nature-nourish-vermicompost-fertilizer/p/itm71fdc8ccb72f1',
                                                            'video_link': 'https://www.youtube.com/watch?v=6Ejq4pbANww'}},
         {'disease_name': 'Tomato : Septoria_leaf_spot',
          'details': {'nature_name': 'Bioleaf MCG Specialized Growing Medium 60L',
                      'img_link': 'https://hydroponic.co.za/wp-content/uploads/2021/01/Bioleaf-MCG-Specialized-Growing-Medium.jpg',
                      'buying_link': 'https://hydroponic.co.za/hydroponics/bioleaf-mcg-specialized-growing-medium-60l/',
                      'video_link': 'https://www.youtube.com/watch?v=puyD8VPrV94'}},
         {'disease_name': 'Tomato : Spider_mites Two-spotted_spider_mitet',
          'details': {'nature_name': 'Alagaezyme Plant Micro Nutrient Elixir',
                      'img_link': 'https://m.media-amazon.com/images/I/61prHRpf2JL._SY500_.jpg',
                      'buying_link': 'https://www.amazon.in/Alagaezyme-Plant-Micro-Nutrient-Elixir/dp/B0CTSM1417',
                      'video_link': 'https://www.youtube.com/watch?v=XhdSdJi79Qo'}},
         {'disease_name': 'Tomato : Target_Spot',
          'details': {'nature_name': 'Tomato Plant Fertilizer For Overall Growth of Tomato',
                      'img_link': 'https://agriculturereview.com/wp-content/uploads/2024/04/tomato-plant-fertilizer-agriculture-review.webp',
                      'buying_link': 'https://agriculturereview.com/product/tomato-plant-fertilizer-for-overall-growth-of-tomato?srsltid=AfmBOoo6ARBuHRIGkXSWK4snTgkuiRTQdPdpexqhKGQdbRbqAUjIpPQcEMg',
                      'video_link': 'https://www.youtube.com/watch?v=nnJ4W7aA2EQ&t=63s'}},
         {'disease_name': 'Tomato : Mosaic_virus', 'details': {'nature_name': 'Green Leaf Nutra Plant Protector',
                                                               'img_link': 'https://5.imimg.com/data5/KK/RT/MY-33257937/nutra-plant-protector-100-ml-500x500.jpg',
                                                               'buying_link': 'https://www.indiamart.com/proddetail/nutra-plant-protector-100-ml-16480166955.html',
                                                               'video_link': 'https://www.youtube.com/watch?v=IRtsGvJ6wXg'}},
         {'disease_name': 'Tomato : Yellow_Leaf_Curl_Virus',
          'details': {'nature_name': 'Pure & Organic Super Vermicompost<',
                      'img_link': 'https://m.media-amazon.com/images/I/61Uj-Wd00hL._SX522_.jpg',
                      'buying_link': 'https://www.amazon.in/Organic-Super-Vermicompost-Types-Plants/dp/B0BTZ3JBT9?source=ps-sl-shoppingads-lpcontext&ref_=fplfs&psc=1&smid=A2MY2Y26SW41DQ',
                      'video_link': 'https://www.youtube.com/watch?v=Hl6HGZ-kXtM'}}]

    ans = {}
    fl = False;
    for i in d:
        if(i["disease_name"].lower() == string.lower()):
            ans = i["details"];
            fl = True;
            break;
    if(fl):
        print(ans);
    else:
        print("Not found")

    return render_template('nature_supplements.html',ans =ans,title = string)

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = prediction(file_path)
        print(pred)
        title = disease_info['disease_name'][pred]
        description =disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html' , title = title , desc = description , prevent = prevent , 
                               image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)

# Predict the seismic events

Task1: Design a learning method to distinguish Earthquake events from Glacial events or vice versa using the provided 2d image-like data.

Task2: Design a method to determine the source location, depth, and other source parameters from the given datasets. 

Task3: Categorize the events and maybe learn some new patterns other than glacial events.

--- 
├── README.md  
├── clustering.py  
├── config  
│   ├── config.py  
│   └── constant.py  
├── data  
│   ├── earthquakes_catalog.txt  
│   ├── glacial_catalog.txt  
│   └── stations_full.txt  
├── data_loader  
│   ├── dataloader_classifier.py  
│   └── dataloader_regression.py  
├── doc  
│   └── report.pdf  
├── eval.sh  
├── models  
│   ├── __init__.py  
│   ├── alexnet.py  
│   └── vgg16.py  
├── supervised_predict.py  
├── supervised_train.py  
├── train.sh  
├── unsupervised_train.py  
└── utils  
│   ├── iter.py  
│   ├── parse_catalog.py  
│   ├── preprocess.py  
│   └── tools.py  

6 directories, 26 files  

---

```
$ git clone https://github.com/chengwym/SeismicDetection.git
$ sh train.sh
```